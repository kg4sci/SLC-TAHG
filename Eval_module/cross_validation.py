"""Utilities to run k-fold cross validation for N-ary models with tuned hyperparameters.

Features
--------
* Supports standard pipelines implemented in ``Eval_module/<Model>/train_paths.py``.
* Provides a specialised runner for the TAPE cascading model (whose training API
  differs from the other models).
* Allows selecting a subset of models (e.g. run TAPE separately from the others).
"""

import argparse
import json
import os
import importlib
import inspect
import copy
from typing import Dict, Any, List, Callable, Optional, Tuple
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from .config import SBERT_DIM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_PARAMS_DIR = PROJECT_ROOT / "Best_modelPara"

MODEL_MODULE_MAP = {
    "RGCN": "Eval_module.RGCN.train_paths",
    "StarE": "Eval_module.StarE.train_paths",
    "NaLP": "Eval_module.NaLP.train_paths",
    "GRAN": "Eval_module.GRAN.train_paths",
    "NS-HART": "Eval_module.NS-HART.train_paths",
    "HypE": "Eval_module.HypE.train_paths",
    "RAM": "Eval_module.RAM.train_paths",
    "N-ComplEx": "Eval_module.N-ComplEx.train_paths",
}


METRIC_KEYS = [
    "path_acc",
    "path_f1",
    "ab_acc",
    "ab_f1",
    "ab_auc_roc",
    "ab_mcc",
    "bc_acc",
    "bc_f1",
    "bc_auc_roc",
    "bc_mcc",
]


MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # StarE and HypE expect the caller to provide feature toggles.k=1，5，10，20，50，100
    "StarE": {"use_text_features": False, "use_node_features": True},
    "HypE": {"use_text_features": True, "use_node_features": True},
    "NaLP": {"use_text_features": True, "use_node_features": True},
    "NS-HART": {"use_text_features": True, "use_node_features":True},
    "N-ComplEx": {"use_text_features": False, "use_node_features": True},
    "RGCN": {"use_slc_neighbors": True},
    "GRAN": {"use_text_features": True, "use_node_features": False},
    "RAM": {"use_text_features": True, "use_node_features": True},
    # TAPE few-shot 默认不截断，由 CLI/外部参数控制
    "TAPE": {"few_shot_k": None, "few_shot_balance": None},
}


TEXT_FEATURE_FLAG_KEYS: Dict[str, str] = {
    "StarE": "use_text_features",
    "HypE": "use_text_features",
    "NaLP": "use_text_features",
    "NS-HART": "use_text_features",
    "RAM": "use_text_features",
    "N-ComplEx": "use_text_features",
    "TAPE": "use_text_feature",
}

0
def _ensure_text_feature_flag(model_name: str, params: Dict[str, Any]) -> None:
    """Ensure models that rely on text features always see the flag present."""

    flag_key = TEXT_FEATURE_FLAG_KEYS.get(model_name)
    if flag_key is None:
        return
    # 对 TAPE，默认关闭文本特征，除非显式开启；其余模型保持原有默认开启
    default_value = False if model_name == "TAPE" else True
    params.setdefault(flag_key, default_value)


def _tape_metric_projection(raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise metric keys coming from the TAPE pipeline to match ``METRIC_KEYS``."""

    def _get(key: str, fallback_key: Optional[str] = None) -> float:
        value = raw_metrics.get(key) if fallback_key is None else raw_metrics.get(key, raw_metrics.get(fallback_key))
        return float(value) if value is not None else 0.0

    def _coerce(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _coerce(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_coerce(v) for v in value]
        return value

    projected: Dict[str, Any] = {
        "path_acc": _get("path_acc"),
        "path_f1": _get("path_f1"),
        "ab_acc": _get("ab_acc"),
        "ab_f1": _get("ab_f1"),
        "ab_auc_roc": _get("ab_auc_roc", "ab_auc"),
        "ab_mcc": _get("ab_mcc"),
        "bc_acc": _get("bc_acc"),
        "bc_f1": _get("bc_f1"),
        "bc_auc_roc": _get("bc_auc_roc", "bc_auc"),
        "bc_mcc": _get("bc_mcc"),
    }

    for key, value in raw_metrics.items():
        if key in projected:
            continue
        if key in {"ab_auc", "bc_auc"}:
            continue
        projected[key] = _coerce(value)

    return projected


def load_best_params(model_name: str) -> Dict[str, Any]:
    params_path = BEST_PARAMS_DIR / model_name / "tuning_results" / "best_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Best params not found for model {model_name} at {params_path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    _ensure_text_feature_flag(model_name, params)
    return params


def _load_existing_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

import numpy as np

def to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    else:
        return obj

def _write_results(results: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        results = to_json_serializable(results)
        json.dump(results, f, indent=2, ensure_ascii=False)


def enumerate_paths(train_module) -> List[Dict[str, Any]]:
    path_data_module = importlib.import_module("Eval_module.path_data")
    if hasattr(path_data_module, "enumerate_graph_paths"):
        return path_data_module.enumerate_graph_paths()
    raise RuntimeError("enumerate_graph_paths not found in path_data module")


def _aggregate_fold_metrics(all_metrics: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    avg_metrics: Dict[str, Any] = {key: float(np.mean([m.get(key, 0.0) for m in all_metrics])) for key in METRIC_KEYS}
    std_metrics: Dict[str, Any] = {key: float(np.std([m.get(key, 0.0) for m in all_metrics])) for key in METRIC_KEYS}

    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)

    def _is_list_of_numbers(value: Any) -> bool:
        return isinstance(value, list) and all(_is_number(v) for v in value)

    def _is_matrix_of_numbers(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(isinstance(row, list) for row in value)
            and all(all(_is_number(v) for v in row) for row in value)
        )

    for key in sorted(all_keys):
        if key in avg_metrics or key in std_metrics:
            continue
        values = [m.get(key) for m in all_metrics if key in m]
        if not values:
            continue

        if key.endswith("_label_names") or key.endswith("_labels"):
            first = values[0]
            if all(v == first for v in values):
                avg_metrics[key] = first
            else:
                try:
                    counts = Counter([json.dumps(v, ensure_ascii=False, sort_keys=True) for v in values])
                    most_common = counts.most_common(1)[0][0]
                    avg_metrics[key] = json.loads(most_common)
                except Exception:
                    avg_metrics[key] = first
            continue

        if all(_is_number(v) for v in values):
            arr = np.asarray(values, dtype=float)
            avg_metrics[key] = float(arr.mean())
            std_metrics[key] = float(arr.std())
            continue

        if all(_is_list_of_numbers(v) for v in values):
            lengths = {len(v) for v in values}
            if len(lengths) == 1:
                arr = np.asarray(values, dtype=float)
                avg_metrics[key] = arr.mean(axis=0).tolist()
                std_metrics[key] = arr.std(axis=0).tolist()
            continue

        if all(_is_matrix_of_numbers(v) for v in values):
            shapes = {(len(v), len(v[0]) if len(v) > 0 else 0) for v in values}
            if len(shapes) == 1:
                arr = np.asarray(values, dtype=float)
                avg_metrics[key] = arr.mean(axis=0).tolist()
                std_metrics[key] = arr.std(axis=0).tolist()
            continue

    return avg_metrics, std_metrics


def run_k_fold(model_name: str, train_module, params: Dict[str, Any], k: int, seed: int = 42):
    pipeline_signature = inspect.signature(train_module.train_pipeline_from_graph)
    supports_skip_negatives = "skip_negatives" in pipeline_signature.parameters

    paths = enumerate_paths(train_module)
    if not paths:
        raise RuntimeError(f"No paths found for model {model_name}")

    for key in ["rel_AB", "rel_BC"]:
        if any(key not in path for path in paths):
            raise ValueError(f"Missing {key} in paths for model {model_name}")

    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    fold_size = len(paths) // k

    all_metrics = []

    base_params = dict(params)
    if model_name == "GRAN":
        base_params.setdefault("val_every", 10)
        base_params.setdefault("has_attention", True)
    elif model_name in {"NaLP", "HypE", "NS-HART", "StarE", "RAM", "N-ComplEx"}:
        base_params.setdefault("val_every", 10)

    if model_name == "RGCN":
        base_params.setdefault("sbert_dim", SBERT_DIM)

    for key, value in MODEL_DEFAULTS.get(model_name, {}).items():
        base_params.setdefault(key, value)

    if "device" not in base_params:
        base_params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Use external splits without regenerating negatives unless explicitly requested otherwise
    if supports_skip_negatives:
        base_params.setdefault("skip_negatives", True)
    else:
        base_params.pop("skip_negatives", None)

    # Some pipelines (e.g., RAM) define required positional arguments without defaults.
    # When tuned params are missing these keys, provide a safe fallback so CV does not crash.
    required_missing: List[str] = []
    for param_name, param in pipeline_signature.parameters.items():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        if param.default is not inspect._empty:
            continue
        if param_name in {"train_paths", "val_paths", "test_paths"}:
            continue
        if param_name in base_params:
            continue

        if param_name == "n_parts":
            base_params[param_name] = 6
        elif param_name == "max_ary":
            base_params[param_name] = 5
        else:
            required_missing.append(param_name)

    if required_missing:
        raise ValueError(
            f"Best params for {model_name} missing required keys for train_pipeline_from_graph: {required_missing}"
        )

    for fold_idx in range(k):
        start = fold_idx * fold_size
        end = len(paths) if fold_idx == k - 1 else start + fold_size
        val_paths = paths[start:end]
        train_paths = paths[:start] + paths[end:]

        call_kwargs = dict(base_params)
        if not supports_skip_negatives:
            call_kwargs.pop("skip_negatives", None)
        if "seed" in pipeline_signature.parameters:
            call_kwargs["seed"] = seed

        results = train_module.train_pipeline_from_graph(
            **call_kwargs,
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=val_paths,
        )

        val_metrics = results.get("val_metrics", {})
        if not val_metrics:
            raise RuntimeError(f"No validation metrics returned for fold {fold_idx} of model {model_name}")

        val_metrics_snapshot = copy.deepcopy(val_metrics)
        val_metrics_snapshot.setdefault("_model", model_name)
        val_metrics_snapshot.setdefault("_seed", seed)
        val_metrics_snapshot.setdefault("_fold", fold_idx)
        all_metrics.append(val_metrics_snapshot)

    avg_metrics, std_metrics = _aggregate_fold_metrics(all_metrics)

    return {
        "model": model_name,
        "fold_metrics": all_metrics,
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
    }


def run_tape_k_fold(params: Dict[str, Any], k: int = 10, seed: int = 42):
    """Run k-fold CV for the TAPE cascading model using its bespoke training pipeline."""

    try:
        from Eval_module.tape.models.core.trainCascading import train as tape_train
        from Eval_module.tape.models.core.config import cfg as tape_cfg_template
    except ImportError as exc:  # pragma: no cover - import error is informative already
        raise RuntimeError("Failed to import TAPE modules. Ensure TAPE dependencies are installed.") from exc

    paths = enumerate_paths(None)
    if not paths:
        raise RuntimeError("No paths found for TAPE")

    # apply defaults (few-shot etc.) when missing
    tape_base_params = dict(MODEL_DEFAULTS.get("TAPE", {}))
    tape_base_params.update(params)

    for key in ("epochs", "lr", "emb_dim", "hidden_dim", "dropout", "batch_size", "val_every"):
        if key not in tape_base_params:
            raise ValueError(f"Best params for TAPE missing required key: {key}")

    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    fold_size = len(paths) // k

    base_cfg = tape_cfg_template.clone()
    # Ensure output paths are rooted inside the repository to avoid hard-coded absolute paths.
    tape_output_root = PROJECT_ROOT / "Eval_module" / "tape" / "models" / "output_data"
    base_cfg.paths.output_base = str(tape_output_root)
    force_cpu = os.environ.get("TAPE_FORCE_CPU")
    requested_device = os.environ.get("TAPE_DEVICE")

    # Align with the device used for the rest of cross-validation when no overrides are present.
    default_device = base_params.get("device") if 'base_params' in locals() else None
    if isinstance(default_device, str) and default_device.startswith("cuda"):
        try:
            default_device = int(default_device.split(":", 1)[1])
        except (IndexError, ValueError):
            default_device = 0
    elif isinstance(default_device, torch.device) and default_device.type == "cuda":
        default_device = default_device.index or 0
    elif isinstance(default_device, (int, float)):
        default_device = int(default_device)
    else:
        default_device = getattr(base_cfg, "device", 0)

    if requested_device is not None:
        try:
            desired_device = int(requested_device)
        except ValueError:
            print(f"[WARN] Invalid TAPE_DEVICE='{requested_device}', defaulting to {default_device}")
            desired_device = default_device
    else:
        desired_device = default_device

    if force_cpu is not None and force_cpu.strip() != "0":
        desired_device = -1
    elif torch.cuda.is_available():
        visible_devices = torch.cuda.device_count()
        if visible_devices == 0:
            desired_device = -1
        else:
            desired_device = max(0, min(desired_device, visible_devices - 1))
    else:
        desired_device = -1

    base_cfg.device = desired_device

    base_cfg.cascade.emb_dim = tape_base_params["emb_dim"]
    base_cfg.cascade.hidden_dim = tape_base_params["hidden_dim"]
    base_cfg.cascade.dropout = tape_base_params["dropout"]
    base_cfg.cascade.batch_size = tape_base_params["batch_size"]
    base_cfg.cascade.lr = tape_base_params["lr"]
    base_cfg.cascade.epochs = tape_base_params["epochs"]
    base_cfg.cascade.val_every = tape_base_params["val_every"]
    base_cfg.cascade.use_node_feat = tape_base_params.get("use_node_feat", base_cfg.cascade.use_node_feat)
    base_cfg.cascade.use_text_feature = tape_base_params.get("use_text_feature", True)

    few_shot_k = tape_base_params.get("few_shot_k")
    few_shot_balance = tape_base_params.get("few_shot_balance")

    all_metrics: List[Dict[str, float]] = []

    for fold_idx in range(k):
        start = fold_idx * fold_size
        end = len(paths) if fold_idx == k - 1 else start + fold_size
        val_paths = paths[start:end]
        train_paths = paths[:start] + paths[end:]

        cfg_fold = base_cfg.clone()
        cfg_fold.seed = seed + fold_idx

        results = tape_train(
            cfg_fold,
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=val_paths,
            few_shot_k=few_shot_k,
            few_shot_balance=few_shot_balance,
        )

        if not results or "val" not in results:
            raise RuntimeError(f"TAPE training did not return validation metrics for fold {fold_idx}")

        val_metrics_raw = results["val"]
        fold_metrics = _tape_metric_projection(val_metrics_raw)
        all_metrics.append(fold_metrics)

    avg_metrics, std_metrics = _aggregate_fold_metrics(all_metrics)

    return {
        "model": "TAPE",
        "fold_metrics": all_metrics,
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
    }


SPECIAL_MODEL_RUNNERS: Dict[str, Callable[[Dict[str, Any], int, int], Dict[str, Any]]] = {
    "TAPE": run_tape_k_fold,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k-fold cross validation for configured models.")
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Names of models to evaluate (e.g. StarE RAM). "
            "Use 'TAPE' to run the TAPE pipeline. Defaults to all standard models."
        ),
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling paths")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional list of seeds; if provided, overrides --seed and aggregates all seeds × folds.",
    )
    parser.add_argument(
        "--few_shot_k",
        type=int,
        default=None,
        help="Override few-shot k for ALL models (including RAM); None means do not truncate.",
    )
    parser.add_argument(
        "--few_shot_balance",
        type=str,
        default=None,
        help="Optional stratified sampling key when applying few-shot (relation/disease/pathway).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "Best_modelPara" / "Compl_none.json"),
        help="Path to JSON file storing aggregated results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results for a model in the output JSON (default: skip if already present).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    requested_models = (
        args.models
        if args.models is not None
        else list(MODEL_MODULE_MAP.keys())
    )

    seeds = args.seeds if args.seeds is not None else [args.seed]
    output_path = Path(args.output)
    results: Dict[str, Any] = _load_existing_results(output_path)

    for model_name in requested_models:
        if not args.overwrite and model_name in results:
            print(f"[INFO] Found existing results for {model_name} in {output_path}; skipping. Use --overwrite to rerun.")
            continue
        combined_fold_metrics: List[Dict[str, Any]] = []
        per_seed_results: List[Dict[str, Any]] = []

        if model_name in MODEL_MODULE_MAP:
            module_path = MODEL_MODULE_MAP[model_name]
            try:
                params = load_best_params(model_name)
            except FileNotFoundError:
                continue
            # apply CLI few-shot override uniformly to all models
            if args.few_shot_k is not None:
                params = dict(params)
                params["few_shot_k"] = args.few_shot_k
                if args.few_shot_balance is not None:
                    params["few_shot_balance"] = args.few_shot_balance
            train_module = importlib.import_module(module_path)

            for seed in seeds:
                model_results = run_k_fold(model_name, train_module, params, k=args.folds, seed=seed)
                per_seed_results.append(model_results)
                combined_fold_metrics.extend(model_results.get("fold_metrics", []))

            combined_avg, combined_std = _aggregate_fold_metrics(combined_fold_metrics)
            results[model_name] = {
                "model": model_name,
                "seeds": seeds,
                "folds": args.folds,
                "per_seed": per_seed_results,
                "combined_avg_metrics": combined_avg,
                "combined_std_metrics": combined_std,
            }

        elif model_name in SPECIAL_MODEL_RUNNERS:
            try:
                params = load_best_params(model_name)
            except FileNotFoundError:
                continue
            runner = SPECIAL_MODEL_RUNNERS[model_name]

            if args.few_shot_k is not None:
                params = dict(params)
                params["few_shot_k"] = args.few_shot_k
                if args.few_shot_balance is not None:
                    params["few_shot_balance"] = args.few_shot_balance

            for seed in seeds:
                model_results = runner(params, k=args.folds, seed=seed)
                per_seed_results.append(model_results)
                combined_fold_metrics.extend(model_results.get("fold_metrics", []))

            combined_avg, combined_std = _aggregate_fold_metrics(combined_fold_metrics)
            results[model_name] = {
                "model": model_name,
                "seeds": seeds,
                "folds": args.folds,
                "per_seed": per_seed_results,
                "combined_avg_metrics": combined_avg,
                "combined_std_metrics": combined_std,
            }
        else:
            print(f"[WARN] Unknown model '{model_name}' requested for cross-validation; skipping.")

        _write_results(results, output_path)

    _write_results(results, output_path)


if __name__ == "__main__":
    main()
