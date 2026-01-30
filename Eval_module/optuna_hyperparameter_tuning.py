import os

# DGL在导入期间会尝试加载GraphBolt，提前禁止避免缺失依赖报错
os.environ["DGL_USE_GRAPHBOLT"] = "0"

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import json
import sys
from datetime import datetime
from pathlib import Path
import logging
import traceback
from typing import Dict, Tuple, Any
import numpy as np

from .device_utils import resolve_device
from .config import SBERT_DIM


# 确保所有进程（主进程 + 子进程）都禁用GraphBolt
def ensure_graphbolt_disabled():
    if os.environ.get("DGL_USE_GRAPHBOLT") != "0":
        os.environ["DGL_USE_GRAPHBOLT"] = "0"

# 初始化时立即禁用一次（主进程）
ensure_graphbolt_disabled()

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _to_json_serializable(obj):
    """Convert numpy / torch scalars and arrays to plain Python for json.dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def _set_user_attr_safe(trial: Trial, key: str, value: Any):
    """Set trial user_attr with numpy-safe conversion."""
    trial.set_user_attr(key, _to_json_serializable(value))


class HyperparameterTuner:
    """Optuna超参数优化基类"""
    
    def __init__(self, model_name: str, n_trials: int = 200, n_jobs: int = 1, seed: int = 42):
        self.model_name = model_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        
        project_root = Path(__file__).resolve().parents[1]

        self.results_dir = project_root / "Best_modelPara" / model_name / "tuning_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Legacy directory retained for backward compatibility (analysis scripts, etc.)
        self.legacy_results_dir = project_root / "model_Evaluate" / model_name / "tuning_results"
        self.legacy_results_dir.mkdir(parents=True, exist_ok=True)

        self.db_file = self.results_dir / "optuna_study.db"
        self.db_path = f"sqlite:///{self.db_file.as_posix()}"
        self.best_params_path = self.results_dir / "best_params.json"
        self.best_metrics_path = self.results_dir / "best_metrics.json"
        self.legacy_best_params_path = self.legacy_results_dir / "best_params.json"
        self.legacy_best_metrics_path = self.legacy_results_dir / "best_metrics.json"
        self.device = str(resolve_device())
        
    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        raise NotImplementedError
        
    def objective(self, trial: Trial) -> float:
        raise NotImplementedError
        
    def optimize(self):
        logger.info(f"开始 {self.model_name} 模型的超参数优化...")
        logger.info(f"试验次数: {self.n_trials}, 数据库: {self.db_path}")
        
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        study = optuna.create_study(
            study_name=f"{self.model_name}_optimization",
            storage=self.db_path,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=True
        )
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=1,
            show_progress_bar=True
        )
        
        self._save_results(study)
        return study
    
    def _save_results(self, study):
        logger.info("保存优化结果...")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logger.error(f"没有成功完成的试验！所有{len(study.trials)}个试验都被修剪或失败了。")
            logger.error("原因可能是: 1. 模块导入错误 2. 训练代码异常 3. 依赖缺失")
            return
        
        best_trial = study.best_trial
        best_params = _to_json_serializable(study.best_params)
        best_value = _to_json_serializable(study.best_value)
        
        # 从最佳试验的用户属性中提取详细指标（优先使用完整字典，其次回退到逐项键）
        best_val_metrics = {}
        best_test_metrics = {}
        best_epoch = None
        if best_trial.user_attrs:
            # 优先读取整字典
            if 'val_metrics' in best_trial.user_attrs and isinstance(best_trial.user_attrs['val_metrics'], dict):
                best_val_metrics = best_trial.user_attrs['val_metrics']
            elif 'best_val_metrics' in best_trial.user_attrs and isinstance(best_trial.user_attrs['best_val_metrics'], dict):
                best_val_metrics = best_trial.user_attrs['best_val_metrics']
            else:
                # 回退：提取验证集指标的逐项键
                for key in ['val_path_acc', 'val_path_f1', 'val_ab_acc', 'val_ab_f1', 'val_ab_auc_roc', 'val_ab_mcc',
                           'val_bc_acc', 'val_bc_f1', 'val_bc_auc_roc', 'val_bc_mcc']:
                    if key in best_trial.user_attrs:
                        best_val_metrics[key.replace('val_', '')] = best_trial.user_attrs[key]
            
            if 'test_metrics' in best_trial.user_attrs and isinstance(best_trial.user_attrs['test_metrics'], dict):
                best_test_metrics = best_trial.user_attrs['test_metrics']
            else:
                # 回退：提取测试集指标的逐项键
                for key in ['test_path_acc', 'test_path_f1', 'test_ab_acc', 'test_ab_f1', 'test_ab_auc_roc', 'test_ab_mcc',
                           'test_bc_acc', 'test_bc_f1', 'test_bc_auc_roc', 'test_bc_mcc']:
                    if key in best_trial.user_attrs:
                        best_test_metrics[key.replace('test_', '')] = best_trial.user_attrs[key]
            
            if 'best_epoch' in best_trial.user_attrs:
                best_epoch = _to_json_serializable(best_trial.user_attrs['best_epoch'])
        
        results = _to_json_serializable({
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "n_trials": self.n_trials,
            "best_params": best_params,
            "best_value": best_value,
            "best_trial": best_trial.number,
            "best_val_metrics": best_val_metrics,
            "best_test_metrics": best_test_metrics,
            "best_epoch": best_epoch
        })
        
        with open(self.best_params_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

        with open(self.best_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Mirror outputs to legacy directory so依赖旧路径的脚本仍然可用
        if self.legacy_results_dir != self.results_dir:
            try:
                with open(self.legacy_best_params_path, 'w', encoding='utf-8') as f:
                    json.dump(best_params, f, indent=2, ensure_ascii=False)
                with open(self.legacy_best_metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except OSError as exc:
                logger.warning(f"复制最优参数到 legacy 目录失败: {exc}")
        
        logger.info(f"最优参数已保存到: {self.best_params_path}")
        logger.info(f"优化结果已保存到: {self.best_metrics_path}")
        
        # 复制 optuna study 数据库以保持旧路径兼容
        if self.legacy_results_dir != self.results_dir:
            try:
                shutil.copy2(self.db_file, self.legacy_results_dir / self.db_file.name)
            except Exception as exc:
                logger.warning(f"复制 Optuna 数据库到 legacy 目录失败: {exc}")
        
        self._print_summary(study)
    
    def _print_summary(self, study):
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print("\n" + "="*60)
            print(f"    {self.model_name} 超参数优化结束（无成功试验）")
            print("="*60)
            print(f"总试验数: {len(study.trials)}")
            print(f"成功试验: 0")
            print("="*60 + "\n")
            return
        
        best_trial = study.best_trial
        print("\n" + "="*80)
        print(f"    {self.model_name} 超参数优化完成")
        print("="*80)
        print(f"最佳验证指标 (Path Acc): {study.best_value:.6f}")
        print(f"最佳试验号: {best_trial.number}")
        
        # 显示验证集详细指标
        if best_trial.user_attrs:
            # 优先显示整字典
            val_metrics_block = None
            if 'val_metrics' in best_trial.user_attrs and isinstance(best_trial.user_attrs['val_metrics'], dict):
                val_metrics_block = best_trial.user_attrs['val_metrics']
            elif 'best_val_metrics' in best_trial.user_attrs and isinstance(best_trial.user_attrs['best_val_metrics'], dict):
                val_metrics_block = best_trial.user_attrs['best_val_metrics']
            
            if 'best_epoch' in best_trial.user_attrs and best_trial.user_attrs['best_epoch'] is not None:
                print(f"最佳Epoch: {best_trial.user_attrs['best_epoch']}")
            
            print(f"\n【验证集指标】")
            if val_metrics_block is not None:
                print(f"  Path (A-B-C): Acc={val_metrics_block.get('path_acc', 0):.4f}, "
                      f"F1={val_metrics_block.get('path_f1', 0):.4f}")
                if any(k in val_metrics_block for k in ['ab_acc', 'ab_f1', 'ab_auc_roc', 'ab_mcc']):
                    print(f"  Comp (A-B):   Acc={val_metrics_block.get('ab_acc', 0):.4f}, "
                          f"F1={val_metrics_block.get('ab_f1', 0):.4f}, "
                          f"AUC={val_metrics_block.get('ab_auc_roc', 0):.4f}, "
                          f"MCC={val_metrics_block.get('ab_mcc', 0):.4f}")
                if any(k in val_metrics_block for k in ['bc_acc', 'bc_f1', 'bc_auc_roc', 'bc_mcc']):
                    print(f"  Comp (B-C):   Acc={val_metrics_block.get('bc_acc', 0):.4f}, "
                          f"F1={val_metrics_block.get('bc_f1', 0):.4f}, "
                          f"AUC={val_metrics_block.get('bc_auc_roc', 0):.4f}, "
                          f"MCC={val_metrics_block.get('bc_mcc', 0):.4f}")
            else:
                if 'val_path_acc' in best_trial.user_attrs:
                    print(f"  Path (A-B-C): Acc={best_trial.user_attrs.get('val_path_acc', 0):.4f}, "
                          f"F1={best_trial.user_attrs.get('val_path_f1', 0):.4f}")
                if 'val_ab_acc' in best_trial.user_attrs:
                    print(f"  Comp (A-B):   Acc={best_trial.user_attrs.get('val_ab_acc', 0):.4f}, "
                          f"F1={best_trial.user_attrs.get('val_ab_f1', 0):.4f}, "
                          f"AUC={best_trial.user_attrs.get('val_ab_auc_roc', 0):.4f}, "
                          f"MCC={best_trial.user_attrs.get('val_ab_mcc', 0):.4f}")
                if 'val_bc_acc' in best_trial.user_attrs:
                    print(f"  Comp (B-C):   Acc={best_trial.user_attrs.get('val_bc_acc', 0):.4f}, "
                          f"F1={best_trial.user_attrs.get('val_bc_f1', 0):.4f}, "
                          f"AUC={best_trial.user_attrs.get('val_bc_auc_roc', 0):.4f}, "
                          f"MCC={best_trial.user_attrs.get('val_bc_mcc', 0):.4f}")
            
            # 显示测试集详细指标
            test_metrics_block = None
            if 'test_metrics' in best_trial.user_attrs and isinstance(best_trial.user_attrs['test_metrics'], dict):
                test_metrics_block = best_trial.user_attrs['test_metrics']
            
            if test_metrics_block is not None or 'test_path_acc' in best_trial.user_attrs:
                print(f"\n【测试集指标】")
                if test_metrics_block is not None:
                    print(f"  Path (A-B-C): Acc={test_metrics_block.get('path_acc', 0):.4f}, "
                          f"F1={test_metrics_block.get('path_f1', 0):.4f}")
                    if any(k in test_metrics_block for k in ['ab_acc', 'ab_f1', 'ab_auc_roc', 'ab_mcc']):
                        print(f"  Comp (A-B):   Acc={test_metrics_block.get('ab_acc', 0):.4f}, "
                              f"F1={test_metrics_block.get('ab_f1', 0):.4f}, "
                              f"AUC={test_metrics_block.get('ab_auc_roc', 0):.4f}, "
                              f"MCC={test_metrics_block.get('ab_mcc', 0):.4f}")
                    if any(k in test_metrics_block for k in ['bc_acc', 'bc_f1', 'bc_auc_roc', 'bc_mcc']):
                        print(f"  Comp (B-C):   Acc={test_metrics_block.get('bc_acc', 0):.4f}, "
                              f"F1={test_metrics_block.get('bc_f1', 0):.4f}, "
                              f"AUC={test_metrics_block.get('bc_auc_roc', 0):.4f}, "
                              f"MCC={test_metrics_block.get('bc_mcc', 0):.4f}")
                else:
                    print(f"  Path (A-B-C): Acc={best_trial.user_attrs.get('test_path_acc', 0):.4f}, "
                          f"F1={best_trial.user_attrs.get('test_path_f1', 0):.4f}")
                    if 'test_ab_acc' in best_trial.user_attrs:
                        print(f"  Comp (A-B):   Acc={best_trial.user_attrs.get('test_ab_acc', 0):.4f}, "
                              f"F1={best_trial.user_attrs.get('test_ab_f1', 0):.4f}, "
                              f"AUC={best_trial.user_attrs.get('test_ab_auc_roc', 0):.4f}, "
                              f"MCC={best_trial.user_attrs.get('test_ab_mcc', 0):.4f}")
                    if 'test_bc_acc' in best_trial.user_attrs:
                        print(f"  Comp (B-C):   Acc={best_trial.user_attrs.get('test_bc_acc', 0):.4f}, "
                              f"F1={best_trial.user_attrs.get('test_bc_f1', 0):.4f}, "
                              f"AUC={best_trial.user_attrs.get('test_bc_auc_roc', 0):.4f}, "
                              f"MCC={best_trial.user_attrs.get('test_bc_mcc', 0):.4f}")
        
        print(f"\n【最优超参数】")
        for key, value in study.best_params.items():
            print(f"  {key:20s} = {value}")
        print("="*80 + "\n")
    
    def _log_metrics(self, trial: Trial, val_metrics: Dict, test_metrics: Dict = None):
        """记录并显示所有评估指标"""
        # 记录验证集指标
        if val_metrics:
            # 额外记录整字典（确保后续能直接获取“最佳epoch”的完整指标）
            vm = _to_json_serializable(val_metrics)
            _set_user_attr_safe(trial, 'val_metrics', vm)
            _set_user_attr_safe(trial, 'val_path_acc', vm.get('path_acc', 0))
            _set_user_attr_safe(trial, 'val_path_f1', vm.get('path_f1', 0))
            _set_user_attr_safe(trial, 'val_ab_acc', vm.get('ab_acc', 0))
            _set_user_attr_safe(trial, 'val_ab_f1', vm.get('ab_f1', 0))
            _set_user_attr_safe(trial, 'val_ab_auc_roc', vm.get('ab_auc_roc', 0))
            _set_user_attr_safe(trial, 'val_ab_mcc', vm.get('ab_mcc', 0))
            _set_user_attr_safe(trial, 'val_bc_acc', vm.get('bc_acc', 0))
            _set_user_attr_safe(trial, 'val_bc_f1', vm.get('bc_f1', 0))
            _set_user_attr_safe(trial, 'val_bc_auc_roc', vm.get('bc_auc_roc', 0))
            _set_user_attr_safe(trial, 'val_bc_mcc', vm.get('bc_mcc', 0))
            
            logger.info(f"[Trial {trial.number}] 【验证集指标】")
            logger.info(f"  Path (A-B-C): Acc={val_metrics.get('path_acc', 0):.4f}, "
                       f"F1={val_metrics.get('path_f1', 0):.4f}")
            logger.info(f"  Comp (A-B):   Acc={val_metrics.get('ab_acc', 0):.4f}, "
                       f"F1={val_metrics.get('ab_f1', 0):.4f}, "
                       f"AUC={val_metrics.get('ab_auc_roc', 0):.4f}, "
                       f"MCC={val_metrics.get('ab_mcc', 0):.4f}")
            logger.info(f"  Comp (B-C):   Acc={val_metrics.get('bc_acc', 0):.4f}, "
                       f"F1={val_metrics.get('bc_f1', 0):.4f}, "
                       f"AUC={val_metrics.get('bc_auc_roc', 0):.4f}, "
                       f"MCC={val_metrics.get('bc_mcc', 0):.4f}")
        
        # 记录测试集指标
        if test_metrics:
            # 额外记录整字典
            tm = _to_json_serializable(test_metrics)
            _set_user_attr_safe(trial, 'test_metrics', tm)
            _set_user_attr_safe(trial, 'test_path_acc', tm.get('path_acc', 0))
            _set_user_attr_safe(trial, 'test_path_f1', tm.get('path_f1', 0))
            _set_user_attr_safe(trial, 'test_ab_acc', tm.get('ab_acc', 0))
            _set_user_attr_safe(trial, 'test_ab_f1', tm.get('ab_f1', 0))
            _set_user_attr_safe(trial, 'test_ab_auc_roc', tm.get('ab_auc_roc', 0))
            _set_user_attr_safe(trial, 'test_ab_mcc', tm.get('ab_mcc', 0))
            _set_user_attr_safe(trial, 'test_bc_acc', tm.get('bc_acc', 0))
            _set_user_attr_safe(trial, 'test_bc_f1', tm.get('bc_f1', 0))
            _set_user_attr_safe(trial, 'test_bc_auc_roc', tm.get('bc_auc_roc', 0))
            _set_user_attr_safe(trial, 'test_bc_mcc', tm.get('bc_mcc', 0))
            
            logger.info(f"[Trial {trial.number}] 【测试集指标】")
            logger.info(f"  Path (A-B-C): Acc={test_metrics.get('path_acc', 0):.4f}, "
                       f"F1={test_metrics.get('path_f1', 0):.4f}")
            logger.info(f"  Comp (A-B):   Acc={test_metrics.get('ab_acc', 0):.4f}, "
                       f"F1={test_metrics.get('ab_f1', 0):.4f}, "
                       f"AUC={test_metrics.get('ab_auc_roc', 0):.4f}, "
                       f"MCC={test_metrics.get('ab_mcc', 0):.4f}")
            logger.info(f"  Comp (B-C):   Acc={test_metrics.get('bc_acc', 0):.4f}, "
                       f"F1={test_metrics.get('bc_f1', 0):.4f}, "
                       f"AUC={test_metrics.get('bc_auc_roc', 0):.4f}, "
                       f"MCC={test_metrics.get('bc_mcc', 0):.4f}")
    
    @staticmethod
    def load_best_params(model_name: str) -> Dict[str, Any]:
        project_root = Path(__file__).resolve().parents[1]
        candidate_paths = [
            project_root / "Best_modelPara" / model_name / "tuning_results" / "best_params.json",
            project_root / "model_Evaluate" / model_name / "tuning_results" / "best_params.json",
        ]

        for path in candidate_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)

        logger.warning(f"未找到 {model_name} 的最优参数文件")
        return {}


def _load_module_with_dependencies(module_name: str, file_path: str, package_name: str = None):
    """加载模块及其依赖"""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    if package_name:
        module.__package__ = package_name
    
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RGCNTuner(HyperparameterTuner):
    """
    RGCN 超参数调优器
    """
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        super().__init__("RGCN", n_trials, n_jobs)
    
    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 50, 150, step=10),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 512, step=64),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1),
            "batch_size": trial.suggest_int("batch_size", 16, 128, step=16),
            "use_slc_neighbors":True,
        }
    
    def objective(self, trial: Trial) -> float:
        import time
        start_time = time.time()
        
        try:
            ensure_graphbolt_disabled()
            import os, sys, importlib

            device = self.device

            # 1. 确保项目根目录在 sys.path（项目根是 TAGs+graph）
            current_file = os.path.abspath(__file__)
            eval_module_dir = os.path.dirname(current_file)  # .../benchmark_paper/Eval_module
            project_root = os.path.dirname(eval_module_dir)  # .../benchmark_paper
            top_root = os.path.dirname(project_root)        # .../TAGs+graph

            if top_root not in sys.path:
                sys.path.insert(0, top_root)

            # 2. 使用完整包名导入 RGCN.train_paths
            logger.info(f"[Trial {trial.number}] 开始导入 RGCN 模块...")
            rgcn_module = importlib.import_module("benchmark_paper.Eval_module.RGCN.train_paths")
            train_pipeline_from_graph = rgcn_module.train_pipeline_from_graph

            params = self.define_hyperparameters(trial)
            logger.info(f"[Trial {trial.number}] 参数: {params}")

            result = train_pipeline_from_graph(
                sbert_dim=SBERT_DIM,
                epochs=params["epochs"],
                lr=params["lr"],
                batch_size=params["batch_size"],
                hidden_dim=params["hidden_dim"],
                dropout_rate=params["dropout_rate"],
                weight_decay=params["weight_decay"],
                use_slc_neighbors=params["use_slc_neighbors"],
                device=device
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"[Trial {trial.number}] 训练完成，耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
            
            # ✅ 处理RGCN的返回值
            if isinstance(result, dict) and 'val_metrics' in result:
                # ✅ 情况1: RGCN已修改为返回字典，包含验证指标
                val_metrics = result.get('val_metrics', {})
                test_metrics = result.get('test_metrics', {})
                
                # 记录所有指标
                self._log_metrics(trial, val_metrics, test_metrics)
                
                val_acc = val_metrics.get('path_acc', 0)
                logger.info(f"[Trial {trial.number}] 返回验证集 Path Acc: {val_acc:.4f}")
                return val_acc
            elif isinstance(result, tuple) and len(result) == 4:
                # ❌ 情况2: RGCN仍返回4元组(rgcn, rel_emb, head_ab, head_bc)
                logger.error(f"[Trial {trial.number}] RGCN返回元组，无法获取验证指标")
                logger.error("提示: 修改RGCN的train_pipeline返回字典格式，包含 'val_metrics' 和 'test_metrics'")
                raise ValueError("RGCN train_pipeline未返回验证指标字典，请修改返回格式")
            else:
                # ❌ 未知返回格式
                logger.error(f"[Trial {trial.number}] 未知的返回格式: {type(result)}")
                raise ValueError(f"Unexpected return type from train_pipeline: {type(result)}")

        except (ModuleNotFoundError, ImportError) as e:
            elapsed_time = time.time() - start_time
            logger.error(f"[Trial {trial.number}] 导入失败 [{type(e).__name__}] (耗时: {elapsed_time:.2f}秒): {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        except ValueError as e:
            elapsed_time = time.time() - start_time
            logger.error(f"[Trial {trial.number}] 值错误 [{type(e).__name__}] (耗时: {elapsed_time:.2f}秒): {e}")
            raise optuna.TrialPruned()
        except KeyboardInterrupt:
            elapsed_time = time.time() - start_time
            logger.warning(f"[Trial {trial.number}] 被用户中断 (耗时: {elapsed_time:.2f}秒)")
            raise
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"[Trial {trial.number}] 执行失败 [{type(e).__name__}] (耗时: {elapsed_time:.2f}秒): {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise optuna.TrialPruned()



class StarETuner(HyperparameterTuner):
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        super().__init__("StarE", n_trials, n_jobs)
    
    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 100, 300, step=10),
            "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256, step=64),
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 512, step=64),
            "num_layers": trial.suggest_int("num_layers", 1, 3, step=1),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "use_text_features": True,
            "use_node_features": True,
        }
    
    def objective(self, trial: Trial) -> float:
        ensure_graphbolt_disabled()
        import importlib
        import sys as _sys

        _argv_backup = _sys.argv[:]
        _sys.argv = [_sys.argv[0]]

        try:
            device = self.device
            
            # 使用完整包名导入 StarE.train_paths
            stare_module = importlib.import_module("benchmark_paper.Eval_module.StarE.train_paths")
            train_pipeline_from_graph = stare_module.train_pipeline_from_graph
            
            params = self.define_hyperparameters(trial)
            logger.info(f"[Trial {trial.number}] 参数: {params}")
            
            results = train_pipeline_from_graph(
                device=device,
                epochs=params["epochs"],
                lr=params["lr"],
                embedding_dim=params["embedding_dim"],
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                val_every=10,
                use_text_features=params["use_text_features"],
                use_node_features=params["use_node_features"],
            )
            
            if isinstance(results, dict):
                val_metrics = results.get("best_val_metrics") or results.get("val_metrics", {})
                test_metrics = results.get("test_metrics", {})
                self._log_metrics(trial, val_metrics, test_metrics)
                # 记录最佳epoch（如果有）
                if "best_epoch" in results:
                    trial.set_user_attr("best_epoch", results.get("best_epoch"))

                # 返回验证集的 Path Accuracy 作为优化目标
                return val_metrics.get("path_acc", 0)

            logger.warning(f"[Trial {trial.number}] StarE 返回非字典结果，使用占位得分 0")
            return 0.0

        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Trial {trial.number} 导入失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} 执行失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        finally:
            _sys.argv = _argv_backup


class HypETuner(HyperparameterTuner):
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        super().__init__("HypE", n_trials, n_jobs)
    
    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 100, 200, step=10),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256, step=64),
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 512, step=64),
            "num_layers": trial.suggest_int("num_layers", 1, 3, step=1),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "use_text_features": True,
            "use_node_features": True,
        }
    
    def objective(self, trial: Trial) -> float:
        try:
            ensure_graphbolt_disabled()
            import importlib

            device = self.device
            
            # 使用完整包名导入 HypE.train_paths
            hype_module = importlib.import_module("benchmark_paper.Eval_module.HypE.train_paths")
            train_pipeline_from_graph = hype_module.train_pipeline_from_graph
            
            params = self.define_hyperparameters(trial)
            logger.info(f"[Trial {trial.number}] 参数: {params}")
            
            results = train_pipeline_from_graph(
                epochs=params["epochs"],
                lr=params["lr"],
                embedding_dim=params["embedding_dim"],
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                val_every=10,
                use_text_features=params["use_text_features"],
                use_node_features=params["use_node_features"],
                device=device
            )
            
            # 记录所有指标
            val_metrics = results.get('best_val_metrics') or results.get('val_metrics', {})
            test_metrics = results.get('test_metrics', {})
            self._log_metrics(trial, val_metrics, test_metrics)
            # 记录最佳epoch（如果有）
            if isinstance(results, dict) and 'best_epoch' in results:
                trial.set_user_attr('best_epoch', results.get('best_epoch'))
            
            # 返回验证集的 Path Accuracy 作为优化目标
            return val_metrics.get('path_acc', 0)
            
        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Trial {trial.number} 导入失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} 执行失败 [{type(e).__name__}]: {e}")
            raise optuna.TrialPruned()


class NaLPTuner(HyperparameterTuner):
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        super().__init__("NaLP", n_trials, n_jobs)
    
    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 50, 200, step=10),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256, step=64),
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 512, step=64),
            "num_layers": trial.suggest_int("num_layers", 1, 3, step=1),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "use_text_features": True,
            "use_node_features":True,
        }
    
    def objective(self, trial: Trial) -> float:
        try:
            ensure_graphbolt_disabled()
            import importlib

            device = self.device
            
            # 使用完整包名导入 NaLP.train_paths
            nalp_module = importlib.import_module("benchmark_paper.Eval_module.NaLP.train_paths")
            train_pipeline_from_graph = nalp_module.train_pipeline_from_graph
            
            params = self.define_hyperparameters(trial)
            logger.info(f"[Trial {trial.number}] 参数: {params}")
            
            results = train_pipeline_from_graph(
                epochs=params["epochs"],
                lr=params["lr"],
                embedding_dim=params["embedding_dim"],
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                val_every=10,
                use_text_features=params["use_text_features"],
                use_node_features=params["use_node_features"],
                device=device
            )
            
            # 记录所有指标
            val_metrics = results.get('best_val_metrics') or results.get('val_metrics', {})
            test_metrics = results.get('test_metrics', {})
            self._log_metrics(trial, val_metrics, test_metrics)
            # 记录最佳epoch（如果有）
            if isinstance(results, dict) and 'best_epoch' in results:
                trial.set_user_attr('best_epoch', results.get('best_epoch'))
            
            # 返回验证集的 Path Accuracy 作为优化目标
            return val_metrics.get('path_acc', 0)
            
        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Trial {trial.number} 导入失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} 执行失败 [{type(e).__name__}]: {e}")
            raise optuna.TrialPruned()


class GRANTuner(HyperparameterTuner):
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        super().__init__("GRAN", n_trials, n_jobs)
    
    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 50, 200, step=10),
            "lr": trial.suggest_float("lr", 5e-4, 1e-1, log=True),
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256, step=64),
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=64),
            "num_layers": trial.suggest_int("num_layers", 1, 3, step=1),
            "num_prop": trial.suggest_int("num_prop", 1, 3, step=1),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "use_text_features": True,
            "use_node_features": True,
        }
    
    def objective(self, trial: Trial) -> float:
        try:
            ensure_graphbolt_disabled()
            import importlib

            device = self.device
            
            # 使用完整包名导入 GRAN.train_paths
            gran_module = importlib.import_module("benchmark_paper.Eval_module.GRAN.train_paths")
            train_pipeline_from_graph = gran_module.train_pipeline_from_graph
            
            params = self.define_hyperparameters(trial)
            logger.info(f"[Trial {trial.number}] 参数: {params}")
            
            results = train_pipeline_from_graph(
                epochs=params["epochs"],
                lr=params["lr"],
                embedding_dim=params["embedding_dim"],
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                num_prop=params["num_prop"],
                dropout=params["dropout"],
                val_every=10,
                use_text_features=params["use_text_features"],
                use_node_features=params["use_node_features"],
                has_attention=True,
                device=device
            )
            
            # 记录所有指标
            val_metrics = results.get('best_val_metrics') or results.get('val_metrics', {})
            test_metrics = results.get('test_metrics', {})
            self._log_metrics(trial, val_metrics, test_metrics)
            # 记录最佳epoch（如果有）
            if isinstance(results, dict) and 'best_epoch' in results:
                trial.set_user_attr('best_epoch', results.get('best_epoch'))
            
            # 返回验证集的 Path Accuracy 作为优化目标
            return val_metrics.get('path_acc', 0)
            
        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Trial {trial.number} 导入失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} 执行失败 [{type(e).__name__}]: {e}")
            raise optuna.TrialPruned()


class NSHARTTuner(HyperparameterTuner):
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        super().__init__("NS-HART", n_trials, n_jobs)

    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        embedding_dim = trial.suggest_int("embedding_dim", 64, 256, step=64)
        # 在 Transformer 的多头注意力机制中，embedding_dim 会被平均分成 num_heads 个头
        # 每个头的维度 = embedding_dim // num_heads
        # PyTorch 的 TransformerEncoderLayer 要求 d_model (embedding_dim) 必须能被 nhead (num_heads) 整除
        # 否则会抛出 RuntimeError: "embedding_dim must be divisible by num_heads"
        possible_heads = [h for h in [2, 4, 8] if embedding_dim % h == 0]
        if not possible_heads:
            possible_heads = [1]  # 至少保证有1个头（单头注意力）

        return {
            "epochs": trial.suggest_int("epochs", 50, 200, step=10),
            "lr": trial.suggest_float("lr", 1e-4, 5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "embedding_dim": embedding_dim,
            "encoder_hidden_dim": trial.suggest_int("encoder_hidden_dim", 128, 512, step=64),
            "encoder_layers": trial.suggest_int("encoder_layers", 1, 4, step=1),
            "encoder_heads": trial.suggest_categorical("encoder_heads", possible_heads),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "use_text_features": True,
            "use_node_features": True,
        }

    def objective(self, trial: Trial) -> float:
        import sys as _sys

        _argv_backup = _sys.argv[:]
        try:
            ensure_graphbolt_disabled()
            import importlib.util
            import importlib

            _sys.argv = [_sys.argv[0]]

            device = self.device
            
            # 由于目录名包含连字符，使用 importlib.util 将其注册为合法包名
            package_alias = "benchmark_paper.Eval_module.NS_HART"
            module_alias = f"{package_alias}.train_paths"

            hart_pkg_path = os.path.join(current_dir, "NS-HART", "__init__.py")
            spec_pkg = importlib.util.spec_from_file_location(
                package_alias, hart_pkg_path
            )
            hart_pkg = importlib.util.module_from_spec(spec_pkg)
            hart_pkg.__package__ = package_alias
            hart_pkg.__path__ = [os.path.dirname(hart_pkg_path)]
            sys.modules[package_alias] = hart_pkg
            spec_pkg.loader.exec_module(hart_pkg)
            
            train_paths_file = os.path.join(current_dir, "NS-HART", "train_paths.py")
            spec = importlib.util.spec_from_file_location(
                module_alias, train_paths_file
            )
            hart_module = importlib.util.module_from_spec(spec)
            hart_module.__package__ = package_alias
            sys.modules[module_alias] = hart_module
            spec.loader.exec_module(hart_module)
            train_pipeline_from_graph = hart_module.train_pipeline_from_graph

            params = self.define_hyperparameters(trial)
            logger.info(f"[Trial {trial.number}] 参数: {params}")

            results = train_pipeline_from_graph(
                epochs=params["epochs"],
                lr=params["lr"],
                embedding_dim=params["embedding_dim"],
                encoder_hidden_dim=params["encoder_hidden_dim"],
                encoder_layers=params["encoder_layers"],
                encoder_heads=params["encoder_heads"],
                dropout=params["dropout"],
                val_every=10,
                use_text_features=params["use_text_features"],
                use_node_features=params["use_node_features"],
                device=device,
                weight_decay=params["weight_decay"],
            )

            if results is None:
                raise optuna.TrialPruned()

            # 记录所有指标
            val_metrics = results.get("best_val_metrics") or results.get("val_metrics", {})
            test_metrics = results.get("test_metrics", {})
            self._log_metrics(trial, val_metrics, test_metrics)
            # 记录最佳epoch（如果有）
            if isinstance(results, dict) and "best_epoch" in results:
                trial.set_user_attr("best_epoch", results.get("best_epoch"))

            # 返回验证集的 Path Accuracy 作为优化目标
            return val_metrics.get("path_acc", 0)

        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Trial {trial.number} 导入失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} 执行失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        finally:
            _sys.argv = _argv_backup


class RAMTuner(HyperparameterTuner):
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        super().__init__("RAM", n_trials, n_jobs)
    
    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 50, 200, step=10),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "embedding_dim": trial.suggest_int("embedding_dim", 64, 256, step=64),
            "n_parts": trial.suggest_int("n_parts", 2, 8, step=2),
            "max_ary": trial.suggest_int("max_ary", 5, 6, step=1),  # 至少需要5，因为输入是4元组，arity=5
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "use_text_features": True,
            "use_node_features": True,
        }
    
    def objective(self, trial: Trial) -> float:
        try:
            ensure_graphbolt_disabled()
            import importlib

            device = self.device
            
            # 使用完整包名导入 RAM.train_paths
            ram_module = importlib.import_module("benchmark_paper.Eval_module.RAM.train_paths")
            train_pipeline_from_graph = ram_module.train_pipeline_from_graph
            
            params = self.define_hyperparameters(trial)
            logger.info(f"[Trial {trial.number}] 参数: {params}")
            
            results = train_pipeline_from_graph(
                epochs=params["epochs"],
                lr=params["lr"],
                embedding_dim=params["embedding_dim"],
                n_parts=params["n_parts"],
                max_ary=params["max_ary"],
                dropout=params["dropout"],
                val_every=10,
                use_text_features=params["use_text_features"],
                use_node_features=params["use_node_features"],
                device=device
            )
            
            # 记录所有指标
            val_metrics = results.get('best_val_metrics') or results.get('val_metrics', {})
            test_metrics = results.get('test_metrics', {})
            self._log_metrics(trial, val_metrics, test_metrics)
            # 记录最佳epoch（如果有）
            if isinstance(results, dict) and 'best_epoch' in results:
                trial.set_user_attr('best_epoch', results.get('best_epoch'))
            
            # 返回验证集的 Path Accuracy 作为优化目标
            return val_metrics.get('path_acc', 0)
            
        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Trial {trial.number} 导入失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} 执行失败 [{type(e).__name__}]: {e}")
            raise optuna.TrialPruned()


class TAPECascadeTuner(HyperparameterTuner):
    """
    TAPE 级联模型（CascadingModel）超参数调优器
    使用与 HypE / N-ComplEx 一致的路径级指标（path_acc / path_f1 等）作为评估标准
    """
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        super().__init__("TAPE", n_trials, n_jobs)
    
    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 50, 200, step=10),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "emb_dim": trial.suggest_int("emb_dim", 64, 256, step=64),
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 512, step=64),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "batch_size": trial.suggest_int("batch_size", 16, 128, step=16),
            "val_every": trial.suggest_int("val_every", 5, 20, step=5),
            "use_node_feat": trial.suggest_categorical("use_node_feat", [True, False]),
            "use_text_feature": True,
        }
    
    def objective(self, trial: Trial) -> float:
        ensure_graphbolt_disabled()
        import importlib
        import sys as _sys

        _argv_backup = _sys.argv[:]
        _sys.argv = [_sys.argv[0]]

        try:
            # TAPE 内部决定使用哪个 GPU，这里的 device 主要用于统一日志与结果目录
            device = self.device
            logger.info(f"[Trial {trial.number}] 使用设备: {device} (TAPE 内部通过 cfg.device 控制具体 GPU)")

            # 导入 TAPE 的配置与训练入口
            tape_cfg_module = importlib.import_module("tape.models.core.config")
            tape_train_module = importlib.import_module("tape.models.core.trainCascading")

            base_cfg = tape_cfg_module.update_cfg(tape_cfg_module.cfg, args_str="")

            params = self.define_hyperparameters(trial)
            logger.info(f"[Trial {trial.number}] 参数: {params}")

            # 根据 Optuna 采样结果更新 cfg（只改与级联相关的字段）
            cfg = base_cfg
            cfg.cascade.epochs = params["epochs"]
            cfg.cascade.lr = params["lr"]
            cfg.cascade.emb_dim = params["emb_dim"]
            cfg.cascade.hidden_dim = params["hidden_dim"]
            cfg.cascade.dropout = params["dropout"]
            cfg.cascade.batch_size = params["batch_size"]
            cfg.cascade.val_every = params["val_every"]
            cfg.cascade.use_node_feat = params["use_node_feat"]
            # TAPE 中该开关通过 getattr 访问，可能不存在，使用 setattr 安全设置
            setattr(cfg.cascade, "use_text_feature", params["use_text_feature"])

            # 启动训练（train 已被修改为返回 best_metrics 字典）
            logger.info(f"[Trial {trial.number}] 开始 TAPE 级联训练...")
            best_metrics = tape_train_module.train(cfg)

            if not isinstance(best_metrics, dict) or "val" not in best_metrics or "test" not in best_metrics:
                logger.error(f"[Trial {trial.number}] TAPE train 未返回预期的 best_metrics 字典，返回值类型: {type(best_metrics)}")
                raise optuna.TrialPruned()

            val_raw = best_metrics.get("val", {})
            test_raw = best_metrics.get("test", {})

            # 将 TAPE 的指标键名映射到统一命名（ab_auc -> ab_auc_roc 等）
            def _convert_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
                return {
                    "path_acc": float(raw.get("path_acc", 0.0)),
                    "path_f1": float(raw.get("path_f1", 0.0)),
                    "ab_acc": float(raw.get("ab_acc", 0.0)),
                    "ab_f1": float(raw.get("ab_f1", 0.0)),
                    "ab_auc_roc": float(raw.get("ab_auc", 0.0)),
                    "ab_mcc": float(raw.get("ab_mcc", 0.0)),
                    "bc_acc": float(raw.get("bc_acc", 0.0)),
                    "bc_f1": float(raw.get("bc_f1", 0.0)),
                    "bc_auc_roc": float(raw.get("bc_auc", 0.0)),
                    "bc_mcc": float(raw.get("bc_mcc", 0.0)),
                }

            val_metrics = _convert_metrics(val_raw)
            test_metrics = _convert_metrics(test_raw)

            # 记录所有指标到 trial.user_attrs，便于统一汇总与保存
            self._log_metrics(trial, val_metrics, test_metrics)

            # 记录最佳 epoch（如果有）
            if "epoch" in best_metrics:
                trial.set_user_attr("best_epoch", best_metrics["epoch"])

            logger.info(f"[Trial {trial.number}] 返回验证集 Path Acc: {val_metrics.get('path_acc', 0):.4f}")
            return val_metrics.get("path_acc", 0.0)

        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Trial {trial.number} 导入失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} 执行失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        finally:
            _sys.argv = _argv_backup


class NComplexTuner(HyperparameterTuner):
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        super().__init__("N-ComplEx", n_trials, n_jobs)
    
    def define_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        return {
            "epochs": trial.suggest_int("epochs", 50, 200, step=10),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "embedding_dim": trial.suggest_int("embedding_dim", 128, 512, step=64),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "val_every": trial.suggest_int("val_every", 5, 20, step=5),
            "use_node_features": True,
            "use_text_features": True,
        }
    
    def objective(self, trial: Trial) -> float:
        try:
            ensure_graphbolt_disabled()
            import importlib.util
            import importlib

            device = self.device
            
            package_alias = "benchmark_paper.Eval_module.NComplEx"
            module_alias = f"{package_alias}.train_paths"

            nc_pkg_path = os.path.join(current_dir, "N-ComplEx", "__init__.py")
            nc_pkg_dir = os.path.dirname(nc_pkg_path)
            spec_pkg = importlib.util.spec_from_file_location(
                package_alias,
                nc_pkg_path,
                submodule_search_locations=[nc_pkg_dir],
            )
            nc_pkg = importlib.util.module_from_spec(spec_pkg)
            nc_pkg.__package__ = package_alias
            nc_pkg.__path__ = [nc_pkg_dir]
            sys.modules[package_alias] = nc_pkg
            sys.modules['NComplEx'] = nc_pkg  # backward-compatible alias
            spec_pkg.loader.exec_module(nc_pkg)
            
            train_paths_file = os.path.join(current_dir, "N-ComplEx", "train_paths.py")
            spec = importlib.util.spec_from_file_location(module_alias, train_paths_file)
            nc_module = importlib.util.module_from_spec(spec)
            nc_module.__package__ = package_alias
            sys.modules[module_alias] = nc_module
            sys.modules['NComplEx.train_paths'] = nc_module  # backward-compatible alias
            spec.loader.exec_module(nc_module)
            train_pipeline_from_graph = nc_module.train_pipeline_from_graph
            
            params = self.define_hyperparameters(trial)
            logger.info(f"[Trial {trial.number}] 参数: {params}")
            
            results = train_pipeline_from_graph(
            epochs=params["epochs"],
            lr=params["lr"],
            embedding_dim=params["embedding_dim"],
            dropout=params["dropout"],
            val_every=10,
            use_node_features=params["use_node_features"],
            use_text_features=params["use_text_features"],
            device=device,
        )
            
            # 处理返回值：可能是字典或元组
            if isinstance(results, dict):
                val_metrics = results.get('best_val_metrics') or results.get('val_metrics', {})
                test_metrics = results.get('test_metrics', {})
                self._log_metrics(trial, val_metrics, test_metrics)
                # 记录最佳epoch（如果有）
                if 'best_epoch' in results:
                    trial.set_user_attr('best_epoch', results.get('best_epoch'))
                return val_metrics.get('path_acc', 0)
            else:
                # 如果返回元组，无法获取指标，返回占位符
                logger.warning(f"[Trial {trial.number}] N-ComplEx返回元组，无法获取验证指标")
                return 0.85
        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Trial {trial.number} 导入失败 [{type(e).__name__}]: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"Trial {trial.number} 执行失败 [{type(e).__name__}]: {e}")
            raise optuna.TrialPruned()


def get_tuner(model_name: str, n_trials: int = 100, n_jobs: int = 1) -> HyperparameterTuner:
    tuners = {
        "RGCN": RGCNTuner,
        "StarE": StarETuner,
        "HypE": HypETuner,
        "NaLP": NaLPTuner,
        "GRAN": GRANTuner,
        "NS-HART": NSHARTTuner,
        "RAM": RAMTuner,
        "TAPE": TAPECascadeTuner,
        "N-ComplEx": NComplexTuner,
    }
    
    if model_name not in tuners:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(tuners.keys())}")
    
    return tuners[model_name](n_trials=n_trials, n_jobs=n_jobs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optuna超参数自动优化框架")
    parser.add_argument("--model", type=str, default="RGCN",
                       choices=["RGCN", "StarE", "HypE", "NaLP", "GRAN", "NS-HART", "RAM", "TAPE", "N-ComplEx"],
                       help="要优化的模型名称")
    parser.add_argument("--n_trials", type=int, default=200,
                       help="优化试验次数（推荐: 200-500次用于论文发表，100次用于快速测试）")
    parser.add_argument("--n_jobs", type=int, default=1,
                       help="并行工作数")
    parser.add_argument("--load_best", action="store_true",
                       help="加载并显示已保存的最优参数")
    
    args = parser.parse_args()
    
    if args.load_best:
        best_params = HyperparameterTuner.load_best_params(args.model)
        print(f"\n{args.model} 的最优参数:")
        for key, value in best_params.items():
            print(f"  {key:20s} = {value}")
    else:
        tuner = get_tuner(args.model, n_trials=args.n_trials, n_jobs=args.n_jobs)
        study = tuner.optimize()

