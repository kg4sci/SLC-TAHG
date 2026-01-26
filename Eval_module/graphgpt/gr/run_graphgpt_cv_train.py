"""
GraphGPT GRAN 10-fold cross validation runner (Stage2 only, fixed Stage1).

流程（每折）：
1) 调用 build_gran_instruct.py 生成本折的 train/val/test instruct（固定随机划分）。
2) 用 Stage1 固定权重训练 Stage2（超参取自 Best_modelPara/GraphGPT/tuning_results/best_params.json）。
3) 用本折模型跑 LP 推理（run_graphgpt_LP.py），再用 eval_gran_cascading.py 计算指标。
4) 汇总所有折的 metrics，输出 avg/std + 逐折指标。

说明：
- Stage1 不重训，共用 --stage1_path。
- graph_data / graph_content 共用同一份（不随折变化）。
- ablation_mode 通过环境变量 GRAPHGPT_ABLATION 传递（none/text/node），下游自行识别。
- 输出目录：{output_root}/abl_<mode>/fold_<i>/ 下保存 instruct / stage2 模型 / 预测与指标。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

def _run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("[CMD]", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(cwd), env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed with code {ret.returncode}: {' '.join(cmd)}")


def _load_best_params(best_params_file: Path) -> Dict[str, Any]:
    with open(best_params_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate(all_metrics: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    聚合逐折指标，尽量保留 NS-HART 等输出的所有字段：
    - 标量数值：求均值/标准差
    - 一维数值列表：按位置求均值/标准差
    - 二维数值列表（矩阵，如混淆矩阵）：按元素求均值/标准差
    - 其他类型：保留首个非空值
    """
    import numpy as np

    def _is_number(x):
        return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)

    def _is_vector(x):
        return isinstance(x, list) and all(_is_number(v) for v in x)

    def _is_matrix(x):
        return (
            isinstance(x, list)
            and all(isinstance(r, list) for r in x)
            and all(all(_is_number(v) for v in r) for r in x)
        )

    avg: Dict[str, Any] = {}
    std: Dict[str, Any] = {}
    keys = set().union(*[m.keys() for m in all_metrics])

    for k in sorted(keys):
        vals = [m[k] for m in all_metrics if k in m]
        if not vals:
            continue
        if all(_is_number(v) for v in vals):
            arr = np.asarray(vals, dtype=float)
            avg[k] = float(arr.mean())
            std[k] = float(arr.std())
        elif all(_is_vector(v) for v in vals):
            lengths = {len(v) for v in vals}
            if len(lengths) == 1:
                arr = np.asarray(vals, dtype=float)
                avg[k] = arr.mean(axis=0).tolist()
                std[k] = arr.std(axis=0).tolist()
            else:
                avg[k] = vals[0]
        elif all(_is_matrix(v) for v in vals):
            shapes = {(len(v), len(v[0]) if len(v) > 0 else 0) for v in vals}
            if len(shapes) == 1:
                arr = np.asarray(vals, dtype=float)
                avg[k] = arr.mean(axis=0).tolist()
                std[k] = arr.std(axis=0).tolist()
            else:
                avg[k] = vals[0]
        else:
            avg[k] = vals[0]
    return avg, std


def main():
    ap = argparse.ArgumentParser(description="GraphGPT GRAN 10-fold CV runner")
    ap.add_argument("--k_folds", type=int, default=10)
    ap.add_argument("--stage1_path", type=str, required=True, help="已训练好的 Stage1 权重目录")
    ap.add_argument("--graph_data_path", type=str, required=True)
    ap.add_argument("--graph_content", type=str, required=True)
    ap.add_argument("--pretrain_graph_mlp_adapter", type=str, default="", help="Stage1 projector 路径")
    ap.add_argument("--best_params_file", type=str, required=True, help="Best_modelPara/GraphGPT/tuning_results/best_params.json")
    ap.add_argument("--output_root", type=str, required=True, help="CV 输出根目录，如 ./graphgpt/gr/folds")
    ap.add_argument("--ablation_modes", type=str, default="none", help="逗号分隔：none,text,node")
    ap.add_argument("--force_label_decode", action="store_true", help="推理时启用约束解码")
    ap.add_argument("--python_exec", type=str, default=sys.executable or "python")
    args = ap.parse_args()

    best_params = _load_best_params(Path(args.best_params_file))
    modes = [m.strip() for m in args.ablation_modes.split(",") if m.strip()]

    eval_module_root = Path(__file__).resolve().parents[2]
    repo_root = eval_module_root.parent

    def _abs(path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (repo_root / p)

    stage1_path = _abs(args.stage1_path)
    projector_path = _abs(args.pretrain_graph_mlp_adapter) if args.pretrain_graph_mlp_adapter else None
    graph_data_path = _abs(args.graph_data_path)
    graph_content_path = _abs(args.graph_content)
    best_params_file = _abs(args.best_params_file)
    output_root = _abs(args.output_root)

    for mode in modes:
        mode_root = output_root / f"abl_{mode}"
        for fold in range(args.k_folds):
            fold_root = mode_root / f"fold_{fold}"
            fold_root.mkdir(parents=True, exist_ok=True)

            # 1) 构建本折 instruct
            data_dir = fold_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            build_cmd = [
                args.python_exec,
                "-m",
                "Eval_module.graphgpt.gr.data.build_gran_instruct",
                "--output_dir",
                str(data_dir),
                "--k_folds",
                str(args.k_folds),
                "--fold_idx",
                str(fold),
                "--path_cache_file",
                str(eval_module_root / "graphgpt" / "gr" / "data" / "stage_2" / "cached_eval_module_paths.json"),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [env.get("PYTHONPATH", ""), str(repo_root), str(eval_module_root), str(eval_module_root / "graphgpt")]
            )
            env["GRAPHGPT_ABLATION"] = mode
            _run_cmd(build_cmd, cwd=repo_root, env=env)

            train_json = data_dir / "gran_train_instruct.json"
            val_json = data_dir / "gran_val_instruct.json"
            test_json = data_dir / "gran_test_instruct.json"

            # 2) 训练 Stage2
            stage2_out = fold_root / "stage2"
            stage2_out.mkdir(parents=True, exist_ok=True)
            train_cmd = [
                args.python_exec,
                "-m",
                "Eval_module.graphgpt.gr.graphgpt.train.train_mem",
                "--model_name_or_path",
                str(stage1_path),
                "--version",
                "v1",
                "--data_path",
                str(train_json),
                "--graph_content",
                str(graph_content_path),
                "--graph_data_path",
                str(graph_data_path),
                "--graph_tower",
                "clip_gt_arxiv",
            ]
            if projector_path:
                train_cmd += ["--pretrain_graph_mlp_adapter", str(projector_path)]
            train_cmd += [
                "--tune_graph_mlp_adapter",
                "False",
                "--freeze_graph_mlp_adapter",
                "False",
                "--lora_enable",
                "True",
                "--lora_r",
                "64",
                "--lora_alpha",
                "128",
                "--lora_dropout",
                "0.05",
                "--lora_bias",
                "none",
                "--graph_select_layer",
                "-2",
                "--use_graph_start_end",
                "True",
                "--bf16",
                "False",
                "--fp16",
                "False",
                "--double_quant",
                "True",
                "--quant_type",
                "nf4",
                "--output_dir",
                str(stage2_out),
                "--num_train_epochs",
                str(best_params.get("num_train_epochs", 4)),
                "--per_device_train_batch_size",
                str(best_params.get("per_device_train_batch_size", 1)),
                "--per_device_eval_batch_size",
                "1",
                "--gradient_accumulation_steps",
                str(best_params.get("gradient_accumulation_steps", 1)),
                "--eval_strategy",
                "no",
                "--save_strategy",
                "steps",
                "--save_steps",
                "50000",
                "--save_total_limit",
                "1",
                "--learning_rate",
                str(best_params.get("learning_rate", 2e-5)),
                "--max_grad_norm",
                "0.3",
                "--warmup_ratio",
                "0.03",
                "--lr_scheduler_type",
                "cosine",
                "--logging_steps",
                "1",
                "--tf32",
                "True",
                "--model_max_length",
                "2048",
                "--gradient_checkpointing",
                "True",
                "--dataloader_num_workers",
                "0",
                "--lazy_preprocess",
                "True",
            ]
            _run_cmd(train_cmd, cwd=repo_root, env=env)

            # 3) 推理
            pred_dir = fold_root / "pred"
            pred_dir.mkdir(parents=True, exist_ok=True)
            lp_cmd = [
                args.python_exec,
                "Eval_module/graphgpt/gr/graphgpt/eval/run_graphgpt_LP.py",
                "--model-name",
                str(stage2_out),
                "--prompting_file",
                str(test_json),
                "--graph_data_path",
                str(graph_data_path),
                "--output_res_path",
                str(pred_dir),
            ]
            if projector_path:
                lp_cmd += ["--projector_path", str(projector_path)]
            if args.force_label_decode:
                lp_cmd.append("--force_label_decode")
            _run_cmd(lp_cmd, cwd=repo_root, env=env)

            pred_file = pred_dir / "arxiv_test_res_all.json"
            metrics_file = pred_dir / "arxiv_test_res_all_metrics.json"
            eval_cmd = [
                args.python_exec,
                "-m",
                "Eval_module.graphgpt.gr.eval_gran_cascading",
                "--model_output_file",
                str(pred_file),
                "--save_path",
                str(metrics_file),
            ]
            _run_cmd(eval_cmd, cwd=repo_root, env=env)

    # 汇总
    all_metrics: List[Dict[str, Any]] = []
    for mode in modes:
        for fold in range(args.k_folds):
            metrics_path = output_root / f"abl_{mode}" / f"fold_{fold}" / "pred" / "arxiv_test_res_all_metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            m["_ablation"] = mode
            m["_fold"] = fold
            all_metrics.append(m)

    if not all_metrics:
        print("No metrics found. Abort aggregation.")
        return

    avg, std = _aggregate(all_metrics)
    summary = {
        "fold_metrics": all_metrics,
        "avg_metrics": avg,
        "std_metrics": std,
    }
    summary_path = output_root / "cv_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved CV summary to {summary_path}")


if __name__ == "__main__":
    main()
