"""
评估GraphGPT在级联预测任务上的表现，统一复用Eval_module的数据接口与指定预测任务的级联指标。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    balanced_accuracy_score,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# 确保 graphgpt/gr 包可被解析
GRAPHGPT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "graphgpt"))
if GRAPHGPT_ROOT not in sys.path:
    sys.path.insert(0, GRAPHGPT_ROOT)

from path_data import enumerate_graph_paths, split_paths  # noqa: E402
from graphgpt.gr.gran_utils import build_relation_type_map_from_paths  # noqa: E402

INVALID_REL_ID = -1
REL_TEXT_KEYS = ("output", "res", "prediction", "answer", "text")


def load_json_records(file_path: str) -> List[Dict[str, Any]]:
    """兼容列表/单条/JSONL格式的加载函数。"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            if "predictions" in data and isinstance(data["predictions"], list):
                return data["predictions"]
            return [data]
    except json.JSONDecodeError:
        pass

    # fallback: 按行解析
    records: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def parse_model_output(output_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从模型输出中解析rel_AB和rel_BC。
    支持多种格式：
    1. "Step 1 - A-B Relationship: XXX\nStep 2 - B-C Relationship: YYY"
    2. "Relationship between A and B: XXX\nRelationship between B and C: YYY"
    3. "1. Relationship: XXX\n2. Relationship: YYY"
    """
    if not output_text:
        return None, None

    rel_ab = None
    rel_bc = None

    # 模式1: "Step 1 - A-B Relationship: XXX" 格式（truth 字段的格式）
    pattern_step1_ab = r"Step\s*1[^:]*A[^:]*B[^:]*Relationship[^:]*:\s*([^\n]+)"
    pattern_step2_bc = r"Step\s*2[^:]*B[^:]*C[^:]*Relationship[^:]*:\s*([^\n]+)"
    match_ab = re.search(pattern_step1_ab, output_text, re.IGNORECASE)
    match_bc = re.search(pattern_step2_bc, output_text, re.IGNORECASE)
    if match_ab:
        rel_ab = match_ab.group(1).strip()
    if match_bc:
        rel_bc = match_bc.group(1).strip()

    # 模式2: "Relationship between A and B: XXX" 格式
    if not rel_ab or not rel_bc:
        pattern1_ab = r"Relationship\s+between\s+A[^:]*and\s+B[^:]*:\s*([^\n]+)"
        pattern1_bc = r"Relationship\s+between\s+B[^:]*and\s+C[^:]*:\s*([^\n]+)"
        match_ab2 = re.search(pattern1_ab, output_text, re.IGNORECASE)
        match_bc2 = re.search(pattern1_bc, output_text, re.IGNORECASE)
        if match_ab2 and not rel_ab:
            rel_ab = match_ab2.group(1).strip()
        if match_bc2 and not rel_bc:
            rel_bc = match_bc2.group(1).strip()

    # 模式3: "1. Relationship: XXX" 格式
    if not rel_ab or not rel_bc:
        pattern2_ab = r"1\.\s*[^:]*Relationship[^:]*:\s*([^\n]+)"
        pattern2_bc = r"2\.\s*[^:]*Relationship[^:]*:\s*([^\n]+)"
        match_ab3 = re.search(pattern2_ab, output_text, re.IGNORECASE)
        match_bc3 = re.search(pattern2_bc, output_text, re.IGNORECASE)
        if match_ab3 and not rel_ab:
            rel_ab = match_ab3.group(1).strip()
        if match_bc3 and not rel_bc:
            rel_bc = match_bc3.group(1).strip()

    # 清理结果：移除末尾的标点符号
    if rel_ab:
        rel_ab = rel_ab.rstrip(".,;:").strip()
    if rel_bc:
        rel_bc = rel_bc.rstrip(".,;:").strip()
    return rel_ab, rel_bc


def extract_truth_from_instruction(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """从GraphGPT指令样本中提取真值标签。"""
    gpt_response = ""
    for conv in item.get("conversations", []):
        if conv.get("from") == "gpt":
            gpt_response = conv.get("value", "") or ""
            break
    return parse_model_output(gpt_response)


def load_ground_truth_from_instructions(file_path: str) -> List[Dict[str, Any]]:
    data = load_json_records(file_path)
    ground_truth: List[Dict[str, Any]] = []
    for idx, sample in enumerate(data):
        rel_ab, rel_bc = extract_truth_from_instruction(sample)
        ground_truth.append(
            {
                "sample_id": sample.get("id", f"sample_{idx}"),
                "true_rel_AB": rel_ab,
                "true_rel_BC": rel_bc,
            }
        )
    return ground_truth


def load_ground_truth_from_eval_module(
    split_name: str,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[int, str]]:
    all_paths = enumerate_graph_paths()
    if not all_paths:
        raise RuntimeError("Eval_module数据接口返回为空，请检查Neo4j/路径构建。")

    name_to_id, id_to_name = build_relation_type_map_from_paths(all_paths)
    train_split, val_split, test_split = split_paths(
        all_paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
    )
    split_map = {"train": train_split, "val": val_split, "test": test_split}
    if split_name not in split_map:
        raise ValueError(f"split_name 必须为 train/val/test，当前为 {split_name}")
    target_paths = split_map[split_name]

    ground_truth: List[Dict[str, Any]] = []
    for idx, path in enumerate(target_paths):
        ground_truth.append(
            {
                "sample_id": f"gran_{split_name}_{idx}_LP",
                "true_rel_AB": path.get("rel_AB"),
                "true_rel_BC": path.get("rel_BC"),
            }
        )
    return ground_truth, name_to_id, id_to_name


def build_mapping_from_samples(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    relation_names = set()
    for sample in samples:
        rel_ab = sample.get("true_rel_AB")
        rel_bc = sample.get("true_rel_BC")
        if rel_ab:
            relation_names.add(rel_ab)
        if rel_bc:
            relation_names.add(rel_bc)
    sorted_names = sorted(relation_names)
    name_to_id = {name: idx for idx, name in enumerate(sorted_names)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return name_to_id, id_to_name


def load_relation_mapping(
    relation_mapping_file: Optional[str],
    fallback_samples: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    if relation_mapping_file and os.path.exists(relation_mapping_file):
        with open(relation_mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return mapping.get("name_to_id", {}), mapping.get("id_to_name", {})
    return build_mapping_from_samples(fallback_samples)


def extract_prediction_text(record: Any) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return ""
    for key in REL_TEXT_KEYS:
        text = record.get(key)
        if isinstance(text, str) and text.strip():
            return text
    # 兼容 `{'choices': [{'message': {'content': ...}}]}`
    maybe_choices = record.get("choices")
    if isinstance(maybe_choices, list) and maybe_choices:
        message = maybe_choices[0]
        if isinstance(message, dict):
            content = message.get("message", {}).get("content") if isinstance(message.get("message"), dict) else message.get("text")
            if isinstance(content, str):
                return content
    return ""


def align_predictions_with_truth(
    ground_truth: List[Dict[str, Any]],
    model_outputs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    id_to_indices: Dict[str, List[int]] = {}
    for idx, record in enumerate(model_outputs):
        sample_id = record.get("id") or record.get("sample_id")
        if isinstance(sample_id, str):
            id_to_indices.setdefault(sample_id, []).append(idx)

    used = [False] * len(model_outputs)
    aligned: List[Dict[str, Any]] = []
    seq_ptr = 0

    for sample in ground_truth:
        sample_id = sample["sample_id"]
        record = None

        candidate_indices = id_to_indices.get(sample_id, [])
        while candidate_indices:
            candidate_idx = candidate_indices.pop(0)
            if not used[candidate_idx]:
                record = model_outputs[candidate_idx]
                used[candidate_idx] = True
                break

        if record is None:
            while seq_ptr < len(model_outputs):
                if not used[seq_ptr]:
                    record = model_outputs[seq_ptr]
                    used[seq_ptr] = True
                    seq_ptr += 1
                    break
                seq_ptr += 1

        aligned.append(
            {
                **sample,
                "raw_prediction": record,
            }
        )
    return aligned


def prepare_prediction_records(
    aligned: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    parsed: List[Dict[str, Any]] = []
    missing = 0
    for sample in aligned:
        raw_prediction = sample.get("raw_prediction")
        pred_text = extract_prediction_text(raw_prediction)
        pred_rel_ab, pred_rel_bc = parse_model_output(pred_text)
        if pred_rel_ab is None or pred_rel_bc is None:
            missing += 1
        parsed.append(
            {
                **sample,
                "pred_rel_AB": pred_rel_ab,
                "pred_rel_BC": pred_rel_bc,
                "parsed_prediction_text": pred_text,
            }
        )
    return parsed, missing


def evaluate_cascading_predictions(
    predictions: List[Dict[str, Any]],
    name_to_id: Dict[str, int],
) -> Dict[str, Any]:
    if not predictions:
        return {
            "path_acc": 0.0,
            "path_f1": 0.0,
            "ab_acc": 0.0,
            "ab_f1": 0.0,
            "ab_auc_roc": 0.0,
            "ab_mcc": 0.0,
            "bc_acc": 0.0,
            "bc_f1": 0.0,
            "bc_auc_roc": 0.0,
            "bc_mcc": 0.0,
        }

    def to_true_id(label: Optional[str]) -> int:
        if label is None:
            return INVALID_REL_ID
        if label not in name_to_id:
            normalized_label = label.strip().upper()
            if normalized_label in name_to_id:
                return name_to_id[normalized_label]
            print(f"WARNING: 未在关系映射中找到真实标签: {label} (normalized: {normalized_label})")
            return INVALID_REL_ID
        return name_to_id[label]

    def to_pred_id(label: Optional[str]) -> int:
        if label is None:
            return INVALID_REL_ID
        return name_to_id.get(label, INVALID_REL_ID)

    # 过滤掉无效的样本（true_rel_AB 或 true_rel_BC 为 None 的样本）
    valid_predictions = []
    for p in predictions:
        true_ab_id = to_true_id(p["true_rel_AB"])
        true_bc_id = to_true_id(p["true_rel_BC"])
        # 只保留两个关系都有效的样本
        if true_ab_id != INVALID_REL_ID and true_bc_id != INVALID_REL_ID:
            valid_predictions.append(p)
    
    if len(valid_predictions) == 0:
        print("WARNING: No valid predictions found after filtering!")
        return {
            "path_acc": 0.0,
            "path_f1": 0.0,
            "ab_acc": 0.0,
            "ab_f1": 0.0,
            "ab_auc_roc": 0.0,
            "ab_mcc": 0.0,
            "bc_acc": 0.0,
            "bc_f1": 0.0,
            "bc_auc_roc": 0.0,
            "bc_mcc": 0.0,
        }
    
    print(f"Using {len(valid_predictions)} valid predictions out of {len(predictions)} total.")
    
    all_true_ab = [to_true_id(p["true_rel_AB"]) for p in valid_predictions]
    all_true_bc = [to_true_id(p["true_rel_BC"]) for p in valid_predictions]
    all_pred_ab = [to_pred_id(p.get("pred_rel_AB")) for p in valid_predictions]
    all_pred_bc = [to_pred_id(p.get("pred_rel_BC")) for p in valid_predictions]

    classes_ab = sorted(set(all_true_ab) | {cls for cls in all_pred_ab if cls != INVALID_REL_ID})
    classes_bc = sorted(set(all_true_bc) | {cls for cls in all_pred_bc if cls != INVALID_REL_ID})

    def compute_auc_macro(y_true: List[int], y_pred: List[int], classes: List[int], fallback_acc: float) -> float:
        if len(set(y_true)) <= 1 or len(classes) <= 1:
            return fallback_acc
        y_true_bin = label_binarize(y_true, classes=classes)
        y_pred_scores = np.zeros((len(y_pred), len(classes)))
        class_to_pos = {cls: idx for idx, cls in enumerate(classes)}
        for row, cls in enumerate(y_pred):
            col = class_to_pos.get(cls)
            if col is not None:
                y_pred_scores[row, col] = 1.0
        try:
            return float(
                roc_auc_score(
                    y_true_bin,
                    y_pred_scores,
                    average="macro",
                    multi_class="ovo",
                )
            )
        except ValueError:
            return fallback_acc

    y_true_path = []
    y_pred_path = []
    for pred_ab, pred_bc, true_ab, true_bc in zip(all_pred_ab, all_pred_bc, all_true_ab, all_true_bc):
        y_true_path.append(0)
        is_ab_correct = pred_ab == true_ab
        is_bc_correct = pred_bc == true_bc
        if is_ab_correct and is_bc_correct:
            y_pred_path.append(0)
        elif is_ab_correct and not is_bc_correct:
            y_pred_path.append(1)
        elif (not is_ab_correct) and is_bc_correct:
            y_pred_path.append(2)
        else:
            y_pred_path.append(3)

    acc_ab = accuracy_score(all_true_ab, all_pred_ab)
    acc_bc = accuracy_score(all_true_bc, all_pred_bc)
    f1_ab = f1_score(all_true_ab, all_pred_ab, average="macro", zero_division=0)
    f1_bc = f1_score(all_true_bc, all_pred_bc, average="macro", zero_division=0)
    mcc_ab = matthews_corrcoef(all_true_ab, all_pred_ab)
    mcc_bc = matthews_corrcoef(all_true_bc, all_pred_bc)
    auc_roc_ab = compute_auc_macro(all_true_ab, all_pred_ab, classes_ab, acc_ab)
    auc_roc_bc = compute_auc_macro(all_true_bc, all_pred_bc, classes_bc, acc_bc)

    path_acc = accuracy_score(y_true_path, y_pred_path)
    path_f1 = f1_score(
        y_true_path,
        y_pred_path,
        labels=[0],
        average="macro",
        zero_division=0,
    )

    # 详细指标
    id_to_name = {v: k for k, v in name_to_id.items()}

    ab_cm = confusion_matrix(all_true_ab, all_pred_ab, labels=classes_ab).tolist()
    bc_cm = confusion_matrix(all_true_bc, all_pred_bc, labels=classes_bc).tolist()

    ab_prec, ab_rec, ab_f1_cls, ab_sup = precision_recall_fscore_support(
        all_true_ab, all_pred_ab, labels=classes_ab, zero_division=0
    )
    bc_prec, bc_rec, bc_f1_cls, bc_sup = precision_recall_fscore_support(
        all_true_bc, all_pred_bc, labels=classes_bc, zero_division=0
    )

    ab_bal_acc = balanced_accuracy_score(all_true_ab, all_pred_ab) if len(classes_ab) > 1 else acc_ab
    bc_bal_acc = balanced_accuracy_score(all_true_bc, all_pred_bc) if len(classes_bc) > 1 else acc_bc

    ab_label_names = [str(id_to_name.get(int(lbl), str(lbl))) for lbl in classes_ab]
    bc_label_names = [str(id_to_name.get(int(lbl), str(lbl))) for lbl in classes_bc]

    return {
        "path_acc": path_acc,
        "path_f1": path_f1,
        "ab_acc": acc_ab,
        "ab_f1": f1_ab,
        "ab_auc_roc": auc_roc_ab,
        "ab_mcc": mcc_ab,
        "bc_acc": acc_bc,
        "bc_f1": f1_bc,
        "bc_auc_roc": auc_roc_bc,
        "bc_mcc": mcc_bc,
        "ab_labels": classes_ab,
        "ab_label_names": ab_label_names,
        "ab_confusion_matrix": ab_cm,
        "ab_precision_by_class": ab_prec.tolist(),
        "ab_recall_by_class": ab_rec.tolist(),
        "ab_f1_by_class": ab_f1_cls.tolist(),
        "ab_support_by_class": ab_sup.tolist(),
        "ab_balanced_acc": float(ab_bal_acc),
        "bc_labels": classes_bc,
        "bc_label_names": bc_label_names,
        "bc_confusion_matrix": bc_cm,
        "bc_precision_by_class": bc_prec.tolist(),
        "bc_recall_by_class": bc_rec.tolist(),
        "bc_f1_by_class": bc_f1_cls.tolist(),
        "bc_support_by_class": bc_sup.tolist(),
        "bc_balanced_acc": float(bc_bal_acc),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GraphGPT Cascading Performance (Direct Mode)")
    parser.add_argument("--model_output_file", type=str, required=True, help="Path to the model prediction json file")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save metrics json")
    parser.add_argument("--split_name", type=str, default="test", help="Ignored in direct mode")
    
    args = parser.parse_args()

    # 1. 加载模型输出文件
    print(f"Loading data from: {args.model_output_file}")
    records = load_json_records(args.model_output_file)
    print(f"Loaded {len(records)} records.")

    if len(records) == 0:
        print("Error: No records found.")
        sys.exit(1)

    # 2. 直接从文件中构建 Ground Truth 和 Predictions
    # 既然文件里已经包含了 'truth' 和 'res' (或 'prediction')，我们直接解析它
    ground_truth = []
    
    print("Extracting Ground Truth and Predictions directly from file...")
    
    for idx, item in enumerate(records):
        # A. 提取真值 (Ground Truth)
        # 优先查找 'truth' 字段 (run_graphgpt_LP.py 生成的字段)
        raw_truth = item.get("truth")
        if not raw_truth:
            # 如果没有 'truth' 字段，尝试从 conversations 里的 gpt 回复提取
            raw_truth_ab, raw_truth_bc = extract_truth_from_instruction(item)
        else:
            raw_truth_ab, raw_truth_bc = parse_model_output(raw_truth)
            
        # B. 构建 Ground Truth 对象
        ground_truth.append({
            "sample_id": item.get("id", f"sample_{idx}"),
            "true_rel_AB": raw_truth_ab,
            "true_rel_BC": raw_truth_bc,
            "raw_prediction": item  # 将整个对象存入，后续 align 步骤会用到
        })

    # 3. 构建关系映射 (Mapping)
    # 根据提取到的所有真实标签动态构建 ID 映射
    name_to_id, id_to_name = build_mapping_from_samples(ground_truth)
    print(f"Built relation mapping with {len(name_to_id)} relations.")

    # 4. 数据对齐 (Align)
    # 因为数据来源是同一个文件，这里主要起到格式标准化的作用
    aligned_data = align_predictions_with_truth(ground_truth, records)

    # 5. 解析预测文本
    print("Parsing prediction text...")
    parsed_data, missing_count = prepare_prediction_records(aligned_data)
    if missing_count > 0:
        print(f"Warning: {missing_count} samples could not be parsed (format error).")

    # 6. 计算指标
    print("Calculating metrics...")
    metrics = evaluate_cascading_predictions(parsed_data, name_to_id)

    # 7. 输出与保存
    print("\n" + "="*30)
    print("Final Evaluation Results:")
    print(json.dumps(metrics, indent=4))
    print("="*30)

    # 自动生成保存路径
    if args.save_path:
        out_path = args.save_path
    else:
        base_dir = os.path.dirname(args.model_output_file)
        base_name = os.path.splitext(os.path.basename(args.model_output_file))[0]
        out_path = os.path.join(base_dir, f"{base_name}_metrics.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to: {out_path}")

