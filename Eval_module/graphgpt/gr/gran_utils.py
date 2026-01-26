"""
本文件提供面向级联预测任务的本地工具函数，参考 GRAN 管线实现，在 `graphgpt/gr` 内独立使用。

主要功能：
- 从 MongoDB 加载 A-B / B-C 实体对的文本证据并编码为向量
- 根据路径样本构建关系类型的 name<->id 映射
- （可选）对给定 GRAN 风格模型计算级联路径指标
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
)


# 确保 Eval_module 根目录可导入（包含 config / db_mongo / device_utils 等）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import SBERT_MODEL_NAME  # type: ignore  # noqa: E402
from db_mongo import get_pair_evidence_embedding  # type: ignore  # noqa: E402
from device_utils import resolve_device  # type: ignore  # noqa: E402


try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - 可选依赖
    SentenceTransformer = None  # type: ignore


def load_text_features_for_pairs(
    paths: List[Dict],
    name_to_id: Dict[str, int],
    embed_dim: int = 384,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    从 MongoDB 加载并编码 A-B / B-C 实体对的文本证据。

    Args:
        paths: N-ary 路径列表，每个元素应至少包含 A_name / B_name / C_name。
        name_to_id: 关系名称到 ID 的映射（保留接口一致性，内部不直接使用）。
        embed_dim: 期望的文本向量维度。

    Returns:
        text_features_ab: {样本索引 -> A-B 文本向量}
        text_features_bc: {样本索引 -> B-C 文本向量}
    """
    text_features_ab: Dict[int, torch.Tensor] = {}
    text_features_bc: Dict[int, torch.Tensor] = {}

    encoder = None
    if SentenceTransformer is not None:
        try:
            encoder = SentenceTransformer(SBERT_MODEL_NAME)
        except Exception as exc:  # pragma: no cover
            print(f"[gran_utils] Warning: init SBERT encoder failed: {exc}")
            encoder = None

    for idx, p in enumerate(paths):
        try:
            a_name = p.get("A_name", "")
            b_name = p.get("B_name", "")
            c_name = p.get("C_name", "")

            # A-B: slc_pathway
            if a_name and b_name:
                pair_type = "slc_pathway"
                _, emb_ab = get_pair_evidence_embedding(
                    pair_type,
                    a_name,
                    b_name,
                    encoder=encoder,
                    reencode=True,
                )
                if emb_ab is None:
                    emb_ab = [0.0] * embed_dim
                text_features_ab[idx] = torch.tensor(emb_ab, dtype=torch.float32)

            # B-C: pathway_disease
            if b_name and c_name:
                pair_type = "pathway_disease"
                _, emb_bc = get_pair_evidence_embedding(
                    pair_type,
                    b_name,
                    c_name,
                    encoder=encoder,
                    reencode=True,
                )
                if emb_bc is None:
                    emb_bc = [0.0] * embed_dim
                text_features_bc[idx] = torch.tensor(emb_bc, dtype=torch.float32)

        except Exception as exc:
            print(f"[gran_utils] Warning: failed to fetch text for path {idx}: {exc}")
            if idx not in text_features_ab:
                text_features_ab[idx] = torch.zeros(embed_dim, dtype=torch.float32)
            if idx not in text_features_bc:
                text_features_bc[idx] = torch.zeros(embed_dim, dtype=torch.float32)

    print(f"[gran_utils] ✓ Loaded text features for {len(text_features_ab)} A-B pairs")
    print(f"[gran_utils] ✓ Loaded text features for {len(text_features_bc)} B-C pairs")

    return text_features_ab, text_features_bc


def build_relation_type_map_from_paths(
    paths: List[Dict],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    根据路径样本中出现的 rel_AB / rel_BC 构建关系名称与 ID 的映射。
    """
    relation_types = set()
    for p in paths:
        if "rel_AB" in p and p["rel_AB"] is not None:
            relation_types.add(p["rel_AB"])
        if "rel_BC" in p and p["rel_BC"] is not None:
            relation_types.add(p["rel_BC"])

    sorted_types = sorted(relation_types)
    name_to_id = {name: idx for idx, name in enumerate(sorted_types)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return name_to_id, id_to_name


def evaluate_path_metrics(
    gran_model,
    g,
    samples: List[Dict],
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
    use_text_features: bool,
    text_features_ab: Optional[Dict[int, torch.Tensor]] = None,
    text_features_bc: Optional[Dict[int, torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """
    复制自 GRAN 评估逻辑的级联路径指标计算函数，用于与 GRAN 结果对齐。
    这里只依赖抽象的 gran_model 接口，而不依赖 GRAN 包本身。
    """
    device = resolve_device(device)

    if len(samples) == 0:
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

    gran_model.eval()

    num_entities = gran_model.num_entities
    num_nodes = g.num_nodes()
    node_features_cpu = g.ndata.get("feat", None)
    node_features = None

    if node_features_cpu is not None:
        node_feat_dim = node_features_cpu.shape[1]
        if num_entities > num_nodes:
            expanded = torch.zeros(
                num_entities,
                node_feat_dim,
                dtype=node_features_cpu.dtype,
                device=torch.device("cpu"),
            )
            expanded[:num_nodes] = node_features_cpu
            node_features_cpu = expanded
        node_features = node_features_cpu.to(device)
    elif hasattr(gran_model, "node_feat_encoder") and gran_model.node_feat_encoder is not None:
        node_feat_dim = gran_model.node_feat_encoder[0].in_features
        node_features = torch.zeros(
            num_entities,
            node_feat_dim,
            dtype=torch.float32,
            device=device,
        )

    all_true_ab, all_pred_ab, all_scores_ab = [], [], []
    all_true_bc, all_pred_bc, all_scores_bc = [], [], []
    y_true_path_4class: List[int] = []
    y_pred_path_4class: List[int] = []

    with torch.no_grad():
        for idx, path in enumerate(samples):
            A, B, C = path["A"], path["B"], path["C"]
            Event = path.get("Event", A)
            true_ab = name_to_id[path["rel_AB"]]
            true_bc = name_to_id[path["rel_BC"]]

            a_ids = torch.tensor([A], dtype=torch.long, device=device)
            b_ids = torch.tensor([B], dtype=torch.long, device=device)
            c_ids = torch.tensor([C], dtype=torch.long, device=device)
            event_ids = torch.tensor([Event], dtype=torch.long, device=device)

            text_ab = None
            text_bc = None
            if use_text_features and text_features_ab and idx in text_features_ab:
                text_ab = text_features_ab[idx].unsqueeze(0).to(device)
            if use_text_features and text_features_bc and idx in text_features_bc:
                text_bc = text_features_bc[idx].unsqueeze(0).to(device)

            logits_ab, logits_bc, _ = gran_model(
                a_ids,
                b_ids,
                c_ids,
                event_ids,
                training=False,
                node_features=node_features,
                text_features_ab=text_ab,
                text_features_bc=text_bc,
            )

            scores_ab = F.softmax(logits_ab, dim=-1).cpu().numpy()[0]
            pred_ab = int(torch.argmax(logits_ab, dim=-1).item())
            all_true_ab.append(true_ab)
            all_pred_ab.append(pred_ab)
            all_scores_ab.append(scores_ab[1] if len(scores_ab) > 1 else scores_ab[0])

            scores_bc = F.softmax(logits_bc, dim=-1).cpu().numpy()[0]
            pred_bc = int(torch.argmax(logits_bc, dim=-1).item())
            all_true_bc.append(true_bc)
            all_pred_bc.append(pred_bc)
            all_scores_bc.append(scores_bc[1] if len(scores_bc) > 1 else scores_bc[0])

            is_ab_correct = (pred_ab == true_ab)
            is_bc_correct = (pred_bc == true_bc)
            y_true_path_4class.append(0)
            if is_ab_correct and is_bc_correct:
                y_pred_path_4class.append(0)
            elif is_ab_correct and not is_bc_correct:
                y_pred_path_4class.append(1)
            elif (not is_ab_correct) and is_bc_correct:
                y_pred_path_4class.append(2)
            else:
                y_pred_path_4class.append(3)

    acc_ab = accuracy_score(all_true_ab, all_pred_ab)
    f1_ab = f1_score(all_true_ab, all_pred_ab, average="macro", zero_division=0)
    mcc_ab = matthews_corrcoef(all_true_ab, all_pred_ab)
    auc_roc_ab = 0.0
    try:
        if len(np.unique(all_true_ab)) > 1:
            auc_roc_ab = roc_auc_score(all_true_ab, all_scores_ab)
        else:
            auc_roc_ab = acc_ab
    except ValueError:
        pass

    acc_bc = accuracy_score(all_true_bc, all_pred_bc)
    f1_bc = f1_score(all_true_bc, all_pred_bc, average="macro", zero_division=0)
    mcc_bc = matthews_corrcoef(all_true_bc, all_pred_bc)
    auc_roc_bc = 0.0
    try:
        if len(np.unique(all_true_bc)) > 1:
            auc_roc_bc = roc_auc_score(all_true_bc, all_scores_bc)
        else:
            auc_roc_bc = acc_bc
    except ValueError:
        pass

    path_acc = accuracy_score(y_true_path_4class, y_pred_path_4class)
    path_f1 = f1_score(
        y_true_path_4class,
        y_pred_path_4class,
        labels=[0],
        average="macro",
        zero_division=0,
    )

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
    }


__all__ = [
    "load_text_features_for_pairs",
    "build_relation_type_map_from_paths",
    "evaluate_path_metrics",
]


