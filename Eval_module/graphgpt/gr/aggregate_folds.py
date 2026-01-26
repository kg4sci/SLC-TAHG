"""
Aggregate fold metrics produced by run_graphgpt_cv_train.py.

Scans output_root/abl_<mode>/fold_<i>/pred/arxiv_test_res_all_metrics.json,
then computes avg/std across folds (numeric scalars/vectors/matrices) and
keeps the first value for non-numeric fields. Saves to output_root/cv_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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


def aggregate(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    return {"avg_metrics": avg, "std_metrics": std, "fold_metrics": all_metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True, help="Root with abl_* folders")
    args = ap.parse_args()

    root = Path(args.output_root)
    all_metrics: List[Dict[str, Any]] = []

    for abl_dir in root.glob("abl_node"):
        for fold_dir in abl_dir.glob("fold_*"):
            metrics_path = fold_dir / "pred" / "arxiv_test_res_all_metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            m["_ablation"] = abl_dir.name.replace("abl_", "")
            m["_fold"] = fold_dir.name.replace("fold_", "")
            all_metrics.append(m)

    if not all_metrics:
        print("No metrics found under", root)
        return

    summary = aggregate(all_metrics)
    summary_path = root / "node_10_fold.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Saved", summary_path)


if __name__ == "__main__":
    main()
