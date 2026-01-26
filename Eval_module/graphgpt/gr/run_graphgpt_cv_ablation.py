"""
GraphGPT 10-fold 评测 + 文本/节点特征消融的简单调度脚本。

功能：
- 读取 best_params.json（optuna 最优）并仅用于记录；目前推理阶段不需要动态调参。
- 将 prompting_file 均分为 K 份（默认 10 折），使用 --start_id/--end_id 控制 run_graphgpt_LP 子集评测。
- 支持 ablation_mode：none / text / node，可在环境变量里告知下游（需要下游脚本自行识别）。
- 可选 force_label_decode（默认开启，避免闲聊）。

使用示例（Eval_module 根目录执行）：
    python -m graphgpt.gr.run_graphgpt_cv_ablation \
      --best_params_file Best_modelPara/GraphGPT/tuning_results/best_params.json \
      --model_name ./graphgpt_output/gran_stage_2_merged \
      --prompting_file ./graphgpt/gr/data/stage_2/gran_test_instruct.json \
      --graph_data_path ./graphgpt/gr/graph_data/gran_graph_data.pt \
      --projector_path ./graphgpt_output/gran_stage_2_merged/graph_projector.fp16.bin \
      --output_root ./graphgpt_output/gran_stage_2_eval_cv \
      --folds 10 --force_label_decode

注：
- 当前脚本仅做子集切分 + 调度，不会重新训练；如需按折重训，请改写 train 命令。
- ablation_mode 仅通过环境变量 GRAPHGPT_ABLATION 提示下游；若需真正消融，请在数据加载处识别该变量。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _run_cmd(cmd: List[str], env=None):
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def split_indices(total: int, folds: int):
    fold_sizes = [math.ceil(total / folds) for _ in range(folds)]
    # 调整最后一折避免越界
    for i in range(folds):
        start = i * fold_sizes[i]
        end = min(total, (i + 1) * fold_sizes[i])
        yield start, end


def main():
    ap = argparse.ArgumentParser(description="GraphGPT K-fold eval + ablation dispatcher")
    ap.add_argument("--best_params_file", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--prompting_file", type=str, required=True)
    ap.add_argument("--graph_data_path", type=str, required=True)
    ap.add_argument("--projector_path", type=str, default="")
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--force_label_decode", action="store_true", help="Use constrained decoding")
    ap.add_argument("--ablation_modes", type=str, default="none,text,node", help="Comma-separated ablations")
    args = ap.parse_args()

    with open(args.prompting_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    total = len(prompts)
    print(f"Total samples: {total}")

    try:
        with open(args.best_params_file, "r", encoding="utf-8") as f:
            best_params = json.load(f)
        print(f"Loaded best params: {best_params}")
    except Exception as e:
        print(f"WARNING: failed to load best params: {e}")
        best_params = {}

    ablation_list = [m.strip() for m in args.ablation_modes.split(",") if m.strip()]

    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    for ablation in ablation_list:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [env.get("PYTHONPATH", ""), str(Path(__file__).resolve().parents[1])]
        )
        env["GRAPHGPT_ABLATION"] = ablation  # 下游可按需识别 text/node/none

        for fold_id, (s, e) in enumerate(split_indices(total, args.folds)):
            out_dir = Path(args.output_root) / f"abl_{ablation}" / f"fold_{fold_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable or "python",
                "graphgpt/gr/graphgpt/eval/run_graphgpt_LP.py",
                "--model-name",
                args.model_name,
                "--prompting_file",
                args.prompting_file,
                "--graph_data_path",
                args.graph_data_path,
                "--output_res_path",
                str(out_dir),
                "--start_id",
                str(s),
                "--end_id",
                str(e),
            ]
            if args.projector_path:
                cmd += ["--projector_path", args.projector_path]
            if args.force_label_decode:
                cmd.append("--force_label_decode")

            print(f"[Fold {fold_id}] ablation={ablation}, start={s}, end={e}")
            _run_cmd(cmd, env=env)

    print("All folds done.")


if __name__ == "__main__":
    main()
