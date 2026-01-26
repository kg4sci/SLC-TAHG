"""
GraphGPT few-shot runner (no CV): repeat over seeds and k list, train Stage2 with limited shots,
run inference, evaluate, aggregate. Defaults: ks = [1,5,10,20,50,100], seeds=[42].

Process per seed:
 1) Build instruct (train/val/test) via build_gran_instruct.py (full data, deterministic seed).
 2) For each k: subsample train set to k examples (shuffle by seed), train Stage2, infer, eval.
 3) Collect metrics with _seed/_k tags, aggregate avg/std per k across seeds.

Note: ablation is not included here. Use run_graphgpt_cv_train.py for ablation CV.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("[CMD]", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(cwd), env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed with code {ret.returncode}: {' '.join(cmd)}")


def _load_best_params(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate(all_metrics: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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


def _subset_json(src: Path, dst: Path, k: int, seed: int) -> None:
    """Few-shot 子集：仅对正样本截取 k，负样本按原始比例截取。

    假设 build_gran_instruct 产物中：
      - 正样本 split == "train"
      - 负样本 split == "train_neg"
    若无 split 字段，则全部视为正样本。
    """
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)
    positives = [d for d in data if d.get("split") == "train"]
    negatives = [d for d in data if d.get("split") == "train_neg"]

    if not positives and not negatives:
        rng.shuffle(data)
        subset = data[:k]
    else:
        if not positives:
            positives = []
        else:
            rng.shuffle(positives)
        pos_subset = positives[:k]

        # 负样本按原始比例截取
        if positives:
            ratio = len(negatives) / float(len(positives))
            neg_keep = int(round(ratio * k))
        else:
            neg_keep = 0
        rng.shuffle(negatives)
        neg_subset = negatives[:neg_keep]
        subset = pos_subset + neg_subset

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)


def parse_args():
    ap = argparse.ArgumentParser(description="GraphGPT few-shot runner")
    ap.add_argument("--ks", type=str, default="1,5,10,20,50,100", help="Comma-separated k shots")
    ap.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    ap.add_argument("--stage1_path", type=str, required=True)
    ap.add_argument("--graph_data_path", type=str, required=True)
    ap.add_argument("--graph_content", type=str, required=True)
    ap.add_argument("--pretrain_graph_mlp_adapter", type=str, default="")
    ap.add_argument("--best_params_file", type=str, required=True)
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--force_label_decode", action="store_true")
    ap.add_argument("--python_exec", type=str, default=sys.executable or "python")
    ap.add_argument("--split_seed", type=int, default=42, help="Seed for train/val/test split in build_gran_instruct")
    return ap.parse_args()


def main():
    args = parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    eval_module_root = Path(__file__).resolve().parents[2]
    repo_root = eval_module_root.parent

    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (repo_root / path)

    stage1_path = _abs(args.stage1_path)
    projector_path = _abs(args.pretrain_graph_mlp_adapter) if args.pretrain_graph_mlp_adapter else None
    graph_data_path = _abs(args.graph_data_path)
    graph_content_path = _abs(args.graph_content)
    best_params_file = _abs(args.best_params_file)
    output_root = _abs(args.output_root)

    best_params = _load_best_params(best_params_file)

    results_all: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)

        # 1) build instruct (full train/val/test) for this seed
        data_dir = seed_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        build_cmd = [
            args.python_exec,
            "-m",
            "Eval_module.graphgpt.gr.data.build_gran_instruct",
            "--output_dir",
            str(data_dir),
            "--path_cache_file",
            str(eval_module_root / "graphgpt" / "gr" / "data" / "stage_2" / "cached_eval_module_paths.json"),
            "--split_seed",
            str(seed if args.split_seed is None else args.split_seed),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [env.get("PYTHONPATH", ""), str(repo_root), str(eval_module_root), str(eval_module_root / "graphgpt")]
        )
        _run_cmd(build_cmd, cwd=repo_root, env=env)

        train_json = data_dir / "gran_train_instruct.json"
        val_json = data_dir / "gran_val_instruct.json"
        test_json = data_dir / "gran_test_instruct.json"

        seed_runs: List[Dict[str, Any]] = []

        for k in ks:
            k_root = seed_root / f"k_{k}"
            k_root.mkdir(parents=True, exist_ok=True)

            # few-shot subset of train
            train_subset = k_root / "gran_train_instruct.json"
            _subset_json(train_json, train_subset, k, seed)

            # 2) train Stage2
            stage2_out = k_root / "stage2"
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
                str(train_subset),
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

            # 3) inference
            pred_dir = k_root / "pred"
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

            # 4) eval
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

            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            metrics["_seed"] = seed
            metrics["_k"] = k
            results_all.append(metrics)
            seed_runs.append(metrics)

        per_seed.append({"seed": seed, "metrics": seed_runs})

    # aggregate per k across seeds
    per_k: Dict[int, Dict[str, Any]] = {}
    for k in ks:
        k_metrics = [m for m in results_all if m.get("_k") == k]
        if not k_metrics:
            continue
        avg, std = _aggregate(k_metrics)
        per_k[k] = {"combined_avg_metrics": avg, "combined_std_metrics": std}

    summary = {
        "seeds": seeds,
        "ks": ks,
        "runs": results_all,
        "per_seed": per_seed,
        "per_k": per_k,
    }
    summary_path = output_root / "few_shot_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved few-shot summary to {summary_path}")


if __name__ == "__main__":
    main()
