"""
GraphGPT 级联预测任务的 Optuna 超参数调优脚本（独立于 Eval_module 的 optuna_hyperparameter_tuning.py）。

使用流程（在 Eval_module 根目录下运行，本脚本路径为 graphgpt/gr/graphgpt_optuna_tuning.py）：

    python -m graphgpt.gr.graphgpt_optuna_tuning --n_trials 10

前提：
- Stage1（gran_stage_1）已通过 run_stage1.sh 或等价命令训练好，对应 projector 保存在 ./graphgpt_output/gran_stage_1/graph_projector.bin
- 本脚本只对 Stage2（gran_stage_2） + GraphGPT 评估阶段的关键超参数做搜索，并使用 eval_gran_cascading.py 输出的 path_acc 作为优化目标。
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import optuna


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Eval_module 根目录


def _run_cmd(cmd, cwd=None, env=None):
    """简单的子进程封装，失败时抛异常。"""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")


def _build_stage2_train_cmd(params: Dict[str, Any]) -> list:
    """
    构造 Stage2 训练（gran_stage_2）的命令行。
    参照 graphgpt/gr/train_gran_stage2.sh，但把一部分超参数暴露给 Optuna。
    """
    python_exe = sys.executable or "python"

    model_path = "/mnt/data/lxy/benchmark_paper/vicuna-7b-v1.5-16k"
    instruct_ds = "./graphgpt/gr/data/stage_2/gran_train_instruct.json"
    graph_data_path = "./graphgpt/gr/graph_data/gran_graph_data.pt"
    graph_content = "./graphgpt/gr/data/stage_2/gran_graph_content.json"
    pretra_gnn = "clip_gt_arxiv"
    tuned_proj = "./graphgpt_output/gran_stage_1/graph_projector.bin"
    output_model = "./graphgpt_output/gran_stage_2"

    num_train_epochs = params.get("num_train_epochs", 3)
    learning_rate = params.get("learning_rate", 1e-5)
    per_device_train_batch_size = params.get("per_device_train_batch_size", 1)
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 4)

    cmd = [
        python_exe,
        "graphgpt/gr/graphgpt/train/train_mem.py",
        "--model_name_or_path",
        model_path,
        "--version",
        "v1",
        "--data_path",
        instruct_ds,
        "--graph_content",
        graph_content,
        "--graph_data_path",
        graph_data_path,
        "--graph_tower",
        pretra_gnn,
        "--pretrain_graph_mlp_adapter",
        tuned_proj,
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
        "--output_dir",
        output_model,
        "--num_train_epochs",
        str(num_train_epochs),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--eval_strategy",
        "no",
        "--save_strategy",
        "steps",
        "--save_steps",
        "500",
        "--save_total_limit",
        "2",
        "--learning_rate",
        str(learning_rate),
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
        "--report_to",
        "none",
    ]
    return cmd


def _build_lp_inference_cmd(output_dir: Path, *, force_label_decode: bool = True) -> list:
    """
    构造 run_graphgpt_LP 的命令；输出预测文件到 output_dir/arxiv_test_res_all.json。
    """
    python_exe = sys.executable or "python"

    model_name = "./graphgpt_output/gran_stage_2"
    prompting_file = "./graphgpt/gr/data/stage_2/gran_test_instruct.json"
    graph_data_path = "./graphgpt/gr/graph_data/gran_graph_data.pt"

    cmd = [
        python_exe,
        "graphgpt/gr/graphgpt/eval/run_graphgpt_LP.py",
        "--model-name",
        model_name,
        "--prompting_file",
        prompting_file,
        "--graph_data_path",
        graph_data_path,
        "--output_res_path",
        str(output_dir),
    ]
    if force_label_decode:
        cmd.append("--force_label_decode")
    return cmd


def _build_eval_cascading_cmd(pred_file: Path, metrics_file: Path) -> list:
    """
    构造 eval_gran_cascading 的命令，对预测结果进行级联指标评估，并把指标写入 metrics_file。
    """
    python_exe = sys.executable or "python"
    cmd = [
        python_exe,
        "-m",
        "graphgpt.gr.eval_gran_cascading",
        "--model_output_file",
        str(pred_file),
        "--save_path",
        str(metrics_file),
    ]
    return cmd


def objective(trial: optuna.Trial) -> float:
    """
    单次 Optuna 试验：
    1）用给定超参数训练 Stage2（gran_stage_2）
    2）使用 run_graphgpt_LP.py 生成测试集预测
    3）调用 eval_gran_cascading.py 计算级联指标
    4）读取 metrics.json，并返回 path_acc 作为优化目标
    """
    # 1. 定义搜索空间（可按需扩展）
    params = {
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", 1, 2),
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 8),
    }

    print(f"[Trial {trial.number}] Params: {params}")

    # 为每个 trial 创建独立的输出目录
    trial_dir = PROJECT_ROOT / "graphgpt" / "gr" / "optuna_trials" / f"trial_{trial.number}"
    pred_dir = trial_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # 确保 PYTHONPATH 包含项目根与 graphgpt/gr，避免 ModuleNotFoundError
    extra_paths = [str(PROJECT_ROOT), str(PROJECT_ROOT / "graphgpt" / "gr")]
    existing_pp = env.get("PYTHONPATH", "")
    for p in extra_paths:
        if p not in existing_pp.split(os.pathsep):
            existing_pp = (existing_pp + os.pathsep + p).strip(os.pathsep)
    env["PYTHONPATH"] = existing_pp
    # 可以在这里根据需要指定 GPU，例如：env["CUDA_VISIBLE_DEVICES"] = "0"

    # 2. 训练 Stage2
    train_cmd = _build_stage2_train_cmd(params)
    _run_cmd(train_cmd, cwd=PROJECT_ROOT, env=env)

    # 3. 运行 LP 推理，生成预测文件
    lp_cmd = _build_lp_inference_cmd(pred_dir, force_label_decode=True)
    _run_cmd(lp_cmd, cwd=PROJECT_ROOT, env=env)

    pred_file = pred_dir / "arxiv_test_res_all.json"
    if not pred_file.exists():
        raise RuntimeError(f"预测文件不存在: {pred_file}")

    # 4. 运行级联评估
    metrics_file = pred_dir / "arxiv_test_res_all_metrics.json"
    eval_cmd = _build_eval_cascading_cmd(pred_file, metrics_file)
    _run_cmd(eval_cmd, cwd=PROJECT_ROOT, env=env)

    if not metrics_file.exists():
        raise RuntimeError(f"指标文件不存在: {metrics_file}")

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    path_acc = float(metrics.get("path_acc", 0.0))
    print(f"[Trial {trial.number}] path_acc = {path_acc:.4f}")
    return path_acc


def main():
    parser = argparse.ArgumentParser(description="Optuna 调优 GraphGPT GRAN 级联任务 (Stage2+Eval)")
    parser.add_argument("--n_trials", type=int, default=10, help="本次追加的试验次数（可叠加到已有 study 上）")
    parser.add_argument("--study_name", type=str, default="graphgpt_gran_stage2_optuna", help="Study 名称")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///graphgpt/gr/optuna_trials/graphgpt_gran_optuna.db",
        help="Optuna storage，例如 sqlite:///graphgpt/gr/optuna_trials/graphgpt_gran_optuna.db",
    )
    args = parser.parse_args()

    # 确保 storage 目录存在（仅针对 sqlite）
    if args.storage.startswith("sqlite:///"):
        storage_path = args.storage.replace("sqlite:///", "")
        storage_dir = Path(storage_path).parent
        storage_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state.is_finished()])
    print(f"[Resume] Loaded study '{args.study_name}' with {completed} finished trials.")

    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Optuna Finished ===")
    print(f"Best value (path_acc): {study.best_value:.6f}")
    print(f"Best trial params: {study.best_trial.params}")


if __name__ == "__main__":
    main()


