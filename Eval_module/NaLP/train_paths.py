"""
Training pipeline for NaLP (PyTorch) on N-ary Knowledge Graphs (CASCADING).

与 StarE/HypE 的管线保持一致：
- 路径样本: (A, Event, B) 和 (Event, C)，预测 rel_AB 与 rel_BC
- 级联评估: 先预测 rel_AB，再基于“预测的 rel_AB”预测 rel_BC
- 节点特征: 可选编码
- 文本特征: 提供兼容接口（可选）
"""

from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np

from .models import NaLPBackbone, NaLPCascadingHead
from ..graph_build import build_dgl_graph
from ..path_data import enumerate_graph_paths, split_paths, generate_negatives, select_few_shot
from ..db_mongo import get_pair_evidence_embedding
from ..config import NODE_NAME_FIELD, NEIGHBOR_AGG_METHOD, LABEL_SLC, SBERT_DIM, SBERT_MODEL_NAME
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from ..device_utils import resolve_device


def load_text_features_for_pairs(
    paths: List[Dict], 
    name_to_id: Dict[str, int],
    embed_dim: int = SBERT_DIM
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    加载并嵌入文本特征（从MongoDB）用于A-B和B-C实体对。
    与StarE/HypE保持一致：调用get_pair_evidence_embedding
    """
    text_features_ab: Dict[int, torch.Tensor] = {}
    text_features_bc: Dict[int, torch.Tensor] = {}

    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(SBERT_MODEL_NAME)
    except Exception:
        encoder = None  # 回退：使用缓存或返回零

    for idx, p in enumerate(paths):
        try:
            # A-B: slc_pathway using names
            a_name = p.get("A_name", "")
            b_name = p.get("B_name", "")
            if a_name and b_name:
                pair_type = "slc_pathway"
                _, emb_ab = get_pair_evidence_embedding(pair_type, a_name, b_name, encoder=encoder, reencode=True)
                if emb_ab is None:
                    emb_ab = [0.0] * embed_dim
                text_features_ab[idx] = torch.tensor(emb_ab, dtype=torch.float32)

            # B-C: pathway_disease using names
            b_name2 = p.get("B_name", "")
            c_name = p.get("C_name", "")
            if b_name2 and c_name:
                pair_type = "pathway_disease"
                _, emb_bc = get_pair_evidence_embedding(pair_type, b_name2, c_name, encoder=encoder, reencode=True)
                if emb_bc is None:
                    emb_bc = [0.0] * embed_dim
                text_features_bc[idx] = torch.tensor(emb_bc, dtype=torch.float32)

        except Exception as e:
            print(f"  Warning: Failed to fetch text for path {idx}: {e}")
            if idx not in text_features_ab:
                text_features_ab[idx] = torch.zeros(embed_dim, dtype=torch.float32)
            if idx not in text_features_bc:
                text_features_bc[idx] = torch.zeros(embed_dim, dtype=torch.float32)
            continue

    print(f"  ✓ Loaded text features for {len(text_features_ab)} A-B pairs")
    print(f"  ✓ Loaded text features for {len(text_features_bc)} B-C pairs")

    return text_features_ab, text_features_bc


def tensorize_text_features(
    text_features: Dict[int, torch.Tensor] | None,
    length: int,
    device: Union[str, torch.device],
    embed_dim: int = SBERT_DIM,
) -> torch.Tensor | None:
    if text_features is None:
        return None
    tensor = torch.zeros(length, embed_dim, dtype=torch.float32, device=device)
    for idx in range(length):
        if idx in text_features:
            tensor[idx] = text_features[idx].to(device)
    return tensor


def build_relation_type_map_from_paths(paths: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    relation_types = set()
    for p in paths:
        if "rel_AB" in p:
            relation_types.add(p["rel_AB"])
        if "rel_BC" in p:
            relation_types.add(p["rel_BC"])
    sorted_types = sorted(list(relation_types))
    name_to_id = {name: idx for idx, name in enumerate(sorted_types)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return name_to_id, id_to_name


def prepare_batches(paths: List[Dict], name_to_id: Dict[str, int]) -> Dict[str, torch.Tensor]:
    if len(paths) == 0:
        return {
            "A": torch.tensor([], dtype=torch.long),
            "B": torch.tensor([], dtype=torch.long),
            "C": torch.tensor([], dtype=torch.long),
            "Event": torch.tensor([], dtype=torch.long),
            "y_ab": torch.tensor([], dtype=torch.long),
            "y_bc": torch.tensor([], dtype=torch.long),
        }
    A_ids = [p["A"] for p in paths]
    B_ids = [p["B"] for p in paths]
    C_ids = [p["C"] for p in paths]
    # Event ID应该从路径数据中获取，如果不存在则使用A作为fallback
    Event_ids = [p.get("Event", p["A"]) for p in paths]
    y_ab = [name_to_id[p["rel_AB"]] for p in paths]
    y_bc = [name_to_id[p["rel_BC"]] for p in paths]
    return {
        "A": torch.tensor(A_ids, dtype=torch.long),
        "B": torch.tensor(B_ids, dtype=torch.long),
        "C": torch.tensor(C_ids, dtype=torch.long),
        "Event": torch.tensor(Event_ids, dtype=torch.long),
        "y_ab": torch.tensor(y_ab, dtype=torch.long),
        "y_bc": torch.tensor(y_bc, dtype=torch.long),
    }


def evaluate_path_metrics(
    backbone: NaLPBackbone,
    head: NaLPCascadingHead,
    samples: List[Dict],
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
    use_text_features: bool,
    node_features: torch.Tensor | None,
    text_features_ab: Dict[int, torch.Tensor] | None = None,
    text_features_bc: Dict[int, torch.Tensor] | None = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
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

    backbone.eval()
    head.eval()

    batch = prepare_batches(samples, name_to_id)
    A = batch["A"].to(device)
    B = batch["B"].to(device)
    C = batch["C"].to(device)
    Event = batch["Event"].to(device)

    text_ab_tensor = (
        tensorize_text_features(text_features_ab, len(samples), device) if use_text_features else None
    )
    text_bc_tensor = (
        tensorize_text_features(text_features_bc, len(samples), device) if use_text_features else None
    )

    text_ab_proj = backbone.project_text(text_ab_tensor)
    text_bc_proj = backbone.project_text(text_bc_tensor)

    with torch.no_grad():
        repr_ab = backbone.encode_stage(
            "ab",
            A,
            B,
            C,
            Event,
            node_features=node_features,
            text_features=text_ab_proj,
        )
        logits_ab, cond_signal, rel_ab_pred, cond_onehot = head.forward_ab(repr_ab, training=False)
        cond_rel_emb = backbone.relation_value_from_onehot(cond_onehot.to(device))

        repr_bc = backbone.encode_stage(
            "bc",
            A,
            B,
            C,
            Event,
            node_features=node_features,
            cond_rel_emb=cond_rel_emb,
            text_features=text_bc_proj,
        )
        logits_bc = head.forward_bc(repr_bc, cond_signal)

    probs_ab = F.softmax(logits_ab, dim=-1).cpu().numpy()
    probs_bc = F.softmax(logits_bc, dim=-1).cpu().numpy()
    all_pred_ab = probs_ab.argmax(axis=-1)
    all_pred_bc = probs_bc.argmax(axis=-1)
    all_true_ab = batch["y_ab"].cpu().numpy()
    all_true_bc = batch["y_bc"].cpu().numpy()

    path_true = np.zeros(len(samples), dtype=np.int64)
    path_pred = []
    for pa, pb in zip(all_pred_ab == all_true_ab, all_pred_bc == all_true_bc):
        if pa and pb:
            path_pred.append(0)
        elif pa and not pb:
            path_pred.append(1)
        elif (not pa) and pb:
            path_pred.append(2)
        else:
            path_pred.append(3)

    acc_ab = accuracy_score(all_true_ab, all_pred_ab)
    f1_ab = f1_score(all_true_ab, all_pred_ab, average="macro", zero_division=0)
    mcc_ab = matthews_corrcoef(all_true_ab, all_pred_ab)

    acc_bc = accuracy_score(all_true_bc, all_pred_bc)
    f1_bc = f1_score(all_true_bc, all_pred_bc, average="macro", zero_division=0)
    mcc_bc = matthews_corrcoef(all_true_bc, all_pred_bc)

    path_acc = accuracy_score(path_true, path_pred)
    path_f1 = f1_score(path_true, path_pred, labels=[0], average="macro", zero_division=0)

    try:
        ab_auc_roc = (
            roc_auc_score(all_true_ab, probs_ab[:, 1]) if probs_ab.shape[1] > 1 else acc_ab
        )
    except ValueError:
        ab_auc_roc = 0.0

    try:
        bc_auc_roc = (
            roc_auc_score(all_true_bc, probs_bc[:, 1]) if probs_bc.shape[1] > 1 else acc_bc
        )
    except ValueError:
        bc_auc_roc = 0.0

    labels_ab = sorted(set(all_true_ab) | set(all_pred_ab))
    ab_cm = confusion_matrix(all_true_ab, all_pred_ab, labels=labels_ab).tolist()
    ab_prec, ab_rec, ab_f1_cls, ab_sup = precision_recall_fscore_support(
        all_true_ab,
        all_pred_ab,
        labels=labels_ab,
        zero_division=0,
    )
    ab_bal_acc = balanced_accuracy_score(all_true_ab, all_pred_ab) if len(labels_ab) > 1 else acc_ab
    ab_label_names = [str(id_to_name.get(int(lbl), str(lbl))) for lbl in labels_ab]

    labels_bc = sorted(set(all_true_bc) | set(all_pred_bc))
    bc_cm = confusion_matrix(all_true_bc, all_pred_bc, labels=labels_bc).tolist()
    bc_prec, bc_rec, bc_f1_cls, bc_sup = precision_recall_fscore_support(
        all_true_bc,
        all_pred_bc,
        labels=labels_bc,
        zero_division=0,
    )
    bc_bal_acc = balanced_accuracy_score(all_true_bc, all_pred_bc) if len(labels_bc) > 1 else acc_bc
    bc_label_names = [str(id_to_name.get(int(lbl), str(lbl))) for lbl in labels_bc]

    return {
        "path_acc": float(path_acc),
        "path_f1": float(path_f1),
        "ab_acc": float(acc_ab),
        "ab_f1": float(f1_ab),
        "ab_auc_roc": float(ab_auc_roc),
        "ab_mcc": float(mcc_ab),
        "bc_acc": float(acc_bc),
        "bc_f1": float(f1_bc),
        "bc_auc_roc": float(bc_auc_roc),
        "bc_mcc": float(mcc_bc),
        "ab_labels": labels_ab,
        "ab_label_names": ab_label_names,
        "ab_confusion_matrix": ab_cm,
        "ab_precision_by_class": ab_prec.tolist(),
        "ab_recall_by_class": ab_rec.tolist(),
        "ab_f1_by_class": ab_f1_cls.tolist(),
        "ab_support_by_class": ab_sup.tolist(),
        "ab_balanced_acc": float(ab_bal_acc),
        "bc_labels": labels_bc,
        "bc_label_names": bc_label_names,
        "bc_confusion_matrix": bc_cm,
        "bc_precision_by_class": bc_prec.tolist(),
        "bc_recall_by_class": bc_rec.tolist(),
        "bc_f1_by_class": bc_f1_cls.tolist(),
        "bc_support_by_class": bc_sup.tolist(),
        "bc_balanced_acc": float(bc_bal_acc),
    }


def train_pipeline_from_graph(
    epochs: int,
    lr: float,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    val_every: int,
    use_text_features: bool,
    use_node_features: bool,
    nalp_n_filters: int = None,    # 事实级模型：卷积filter数量
    nalp_n_gfcn: int = None,       # 事实级模型：g-FCN输出维度
    device: Optional[Union[str, torch.device]] = None,
    *,
    train_paths: Optional[List[Dict]] = None,
    val_paths: Optional[List[Dict]] = None,
    test_paths: Optional[List[Dict]] = None,
    skip_negatives: bool = False,
    few_shot_k: Optional[int] = None,
    few_shot_balance: Optional[str] = None,
    seed: int = 0,
):
    """
    训练管线，支持两种模式：use_fact_style=True: 原生NaLP事实级模型（NaLPCascadingFactClassifier）
    """
    device = resolve_device(device)
    print(f"[Device] Using {device} for NaLP pipeline")

    # 1) 构图
    print("\n[1/8] Building DGL graph from Neo4j...")
    # 图保持在CPU上，避免DGL CUDA依赖问题
    print(f"[Device] Graph will stay on CPU (to avoid DGL CUDA dependency)")
    g = build_dgl_graph()[0]  # 图保持在CPU上
    print(f"  ✓ Graph built: {g.num_nodes()} nodes, {g.num_edges()} edges")

    # 2) 路径处理（支持外部折叠）
    if train_paths is None and val_paths is None and test_paths is None:
        print("\n[2/8] Enumerating N-ary paths...")
        all_paths = enumerate_graph_paths()
        print(f"  ✓ Found {len(all_paths)} paths")
        if len(all_paths) == 0:
            print("  ✗ No paths found!")
            return None

        # 3) 关系类型映射
        print("\n[3/8] Building relation type mappings from paths...")
        name_to_id, id_to_name = build_relation_type_map_from_paths(all_paths)
        num_relation_classes = len(name_to_id)
        print(f"  ✓ {num_relation_classes} classes")
        print("[4/8] Splitting data...")
        train_pos, val_pos, test_pos = split_paths(all_paths, train_ratio=0.7, val_ratio=0.15)
        if few_shot_k:
            train_pos = select_few_shot(train_pos, few_shot_k, seed=seed, balance_by=few_shot_balance)
        print(f"  ✓ Train {len(train_pos)}, Val {len(val_pos)}, Test {len(test_pos)}")
    else:
        if train_paths is None or val_paths is None:
            raise ValueError("train_paths and val_paths must be provided when using external splits")
        print("[2/8] Using externally provided train/val/test splits...")
        train_pos = list(train_paths)
        val_pos = list(val_paths)
        test_pos = list(test_paths) if test_paths is not None else list(val_pos)
        print(f"  ✓ Provided splits - Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}")

        combined_paths = train_pos + val_pos + [p for p in test_pos if p not in val_pos]
        if len(combined_paths) == 0:
            print("  ✗ No paths supplied by caller!")
            return None
        print("[3/8] Building relation mappings from provided paths...")
        name_to_id, id_to_name = build_relation_type_map_from_paths(combined_paths)
        num_relation_classes = len(name_to_id)
        print(f"  ✓ {num_relation_classes} classes")
        all_paths = combined_paths
        if few_shot_k:
            train_pos = select_few_shot(train_pos, few_shot_k, seed=seed, balance_by=few_shot_balance)

    # 5) 生成负样本（可选）
    if skip_negatives:
        print("\n[5/8] Skipping negative sampling (skip_negatives=True)...")
        train_neg = []
    else:
        print("\n[5/8] Generating negative samples...")
        train_neg = generate_negatives(train_pos, all_paths)
        print(f"  ✓ Generated {len(train_neg)} negative samples")

    # 6) 初始化模型
    print("\n[6/8] Initializing NaLP model...")
    num_nodes = g.num_nodes()
    num_edge_types = len(torch.unique(g.edata["rel_type"]))

    # 从所有路径中提取所有实体ID，找到最大值和最小值，确保embedding表大小足够
    all_entity_ids = []
    for p in all_paths:
        all_entity_ids.append(p["A"])
        all_entity_ids.append(p["B"])
        all_entity_ids.append(p["C"])
        if "Event" in p and p["Event"] is not None:
            all_entity_ids.append(p["Event"])
    
    if not all_entity_ids:
        print("  ✗ No entity IDs found in paths!")
        return None
    
    min_entity_id = min(all_entity_ids)
    max_entity_id = max(all_entity_ids)
    
    # 检查实体ID是否从0开始
    if min_entity_id < 0:
        print(f"  ⚠ Warning: Found negative entity IDs (min={min_entity_id}). This may cause issues.")
    
    # 确保embedding表大小至少为 max_entity_id + 1
    # 同时也要考虑图的节点数量，取两者最大值
    num_entities = max(num_nodes, max_entity_id + 1)
    print(f"  ✓ Graph nodes: {num_nodes}, Entity ID range: [{min_entity_id}, {max_entity_id}], Using num_entities: {num_entities}")

    node_feat_dim = 0
    if use_node_features and "feat" in g.ndata:
        node_feat_dim = g.ndata["feat"].shape[1]
        print(f"  ✓ Node features: dim={node_feat_dim}")

    # 加载文本特征
    text_features_ab_train = None
    text_features_bc_train = None
    text_features_ab_val = None
    text_features_bc_val = None
    text_features_ab_test = None
    text_features_bc_test = None
    
    if use_text_features:
        print("\n[5.5/8] Loading text features from MongoDB...")
        print("  Loading training set text features...")
        text_features_ab_train, text_features_bc_train = load_text_features_for_pairs(
            train_pos, name_to_id, embed_dim=SBERT_DIM
        )
        print("  Loading validation set text features...")
        text_features_ab_val, text_features_bc_val = load_text_features_for_pairs(
            val_pos, name_to_id, embed_dim=SBERT_DIM
        )
        print("  Loading test set text features...")
        text_features_ab_test, text_features_bc_test = load_text_features_for_pairs(
            test_pos, name_to_id, embed_dim=SBERT_DIM
        )
        train_text_ab_tensor = tensorize_text_features(text_features_ab_train, len(train_pos), device)
        train_text_bc_tensor = tensorize_text_features(text_features_bc_train, len(train_pos), device)
    else:
        train_text_ab_tensor = None
        train_text_bc_tensor = None

    n_filters = nalp_n_filters if nalp_n_filters is not None else hidden_dim
    n_gfcn = nalp_n_gfcn if nalp_n_gfcn is not None else max(256, 2 * embedding_dim)

    backbone = NaLPBackbone(
        num_entities=num_entities,
        num_relation_classes=num_relation_classes,
        embedding_dim=embedding_dim,
        n_filters=n_filters,
        n_gfcn=n_gfcn,
        dropout=dropout,
        node_feat_dim=node_feat_dim,
        text_dim=SBERT_DIM if use_text_features else 0,
        use_text_features=use_text_features,
    ).to(device)
    head = NaLPCascadingHead(repr_dim=n_gfcn, num_relation_classes=num_relation_classes, dropout=dropout).to(device)

    print(f"  ✓ NaLPBackbone (n_filters={n_filters}, n_gfcn={n_gfcn})")
    print(f"  ✓ Cascading head initialized")
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)

    # 7) 训练
    train_batch_pos = prepare_batches(train_pos, name_to_id)
    train_batch_neg = prepare_batches(train_neg, name_to_id) if len(train_neg) > 0 else None

    # 处理节点特征：扩展节点特征张量以匹配实体ID范围
    node_features = g.ndata.get("feat", None)
    if node_features is not None:
        node_features = node_features.to(device)
        # 如果num_entities大于图的节点数，需要扩展节点特征张量
        if num_entities > num_nodes:
            # 创建扩展的节点特征张量，对于超出图的节点使用零向量
            expanded_node_features = torch.zeros(
                num_entities, node_feat_dim, 
                dtype=node_features.dtype, 
                device=device
            )
            # 将原有节点特征复制到扩展张量的前num_nodes行
            expanded_node_features[:num_nodes] = node_features
            node_features = expanded_node_features
            print(f"  ✓ Expanded node features from [{num_nodes}, {node_feat_dim}] to [{num_entities}, {node_feat_dim}]")
    elif node_feat_dim > 0:
        # 即使图中没有节点特征，但如果需要节点特征，创建一个零张量
        node_features = torch.zeros(num_entities, node_feat_dim, dtype=torch.float32, device=device)
        print(f"  ✓ Created zero node features tensor: [{num_entities}, {node_feat_dim}]")

    print("\n[7/8] Training with cascading...")
    best_val_path_acc = 0.0
    best_epoch = 0
    best_val_metrics = None
    last_val_metrics = None

    for epoch in range(epochs):
        backbone.train()
        head.train()
        optimizer.zero_grad()

        a_ids = train_batch_pos["A"].to(device)
        b_ids = train_batch_pos["B"].to(device)
        c_ids = train_batch_pos["C"].to(device)
        event_ids = train_batch_pos["Event"].to(device)

        text_ab_proj = backbone.project_text(train_text_ab_tensor)
        text_bc_proj = backbone.project_text(train_text_bc_tensor)

        repr_ab_pos = backbone.encode_stage(
            "ab",
            a_ids,
            b_ids,
            c_ids,
            event_ids,
            node_features=node_features,
            text_features=text_ab_proj,
        )
        logits_ab_pos, cond_signal_pos, rel_ab_pred_pos, cond_onehot_pos = head.forward_ab(
            repr_ab_pos, training=True
        )
        cond_rel_emb_pos = backbone.relation_value_from_onehot(cond_onehot_pos.to(device))

        repr_bc_pos = backbone.encode_stage(
            "bc",
            a_ids,
            b_ids,
            c_ids,
            event_ids,
            node_features=node_features,
            cond_rel_emb=cond_rel_emb_pos,
            text_features=text_bc_proj,
        )
        logits_bc_pos = head.forward_bc(repr_bc_pos, cond_signal_pos)

        loss_ab = F.cross_entropy(logits_ab_pos, train_batch_pos["y_ab"].to(device))
        loss_bc = F.cross_entropy(logits_bc_pos, train_batch_pos["y_bc"].to(device))
        loss_pos = loss_ab + loss_bc

        if train_batch_neg is not None and len(train_neg) > 0:
            a_ids_neg = train_batch_neg["A"].to(device)
            b_ids_neg = train_batch_neg["B"].to(device)
            c_ids_neg = train_batch_neg["C"].to(device)
            event_ids_neg = train_batch_neg["Event"].to(device)

            if use_text_features:
                text_ab_neg = torch.zeros(len(train_neg), SBERT_DIM, dtype=torch.float32, device=device)
                text_bc_neg = torch.zeros(len(train_neg), SBERT_DIM, dtype=torch.float32, device=device)
                text_ab_neg_proj = backbone.project_text(text_ab_neg)
                text_bc_neg_proj = backbone.project_text(text_bc_neg)
            else:
                text_ab_neg_proj = None
                text_bc_neg_proj = None

            repr_ab_neg = backbone.encode_stage(
                "ab",
                a_ids_neg,
                b_ids_neg,
                c_ids_neg,
                event_ids_neg,
                node_features=node_features,
                text_features=text_ab_neg_proj,
            )
            logits_ab_neg, cond_signal_neg, rel_ab_pred_neg, cond_onehot_neg = head.forward_ab(
                repr_ab_neg, training=True
            )
            cond_rel_emb_neg = backbone.relation_value_from_onehot(cond_onehot_neg.to(device))
            repr_bc_neg = backbone.encode_stage(
                "bc",
                a_ids_neg,
                b_ids_neg,
                c_ids_neg,
                event_ids_neg,
                node_features=node_features,
                cond_rel_emb=cond_rel_emb_neg,
                text_features=text_bc_neg_proj,
            )
            logits_bc_neg = head.forward_bc(repr_bc_neg, cond_signal_neg)

            loss_ab_neg = F.cross_entropy(logits_ab_neg, train_batch_neg["y_ab"].to(device))
            loss_bc_neg = F.cross_entropy(logits_bc_neg, train_batch_neg["y_bc"].to(device))
            loss = loss_pos + 0.5 * (loss_ab_neg + loss_bc_neg)
        else:
            loss = loss_pos

        loss.backward()
        optimizer.step()

        if (epoch + 1) % val_every == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
            val_metrics = evaluate_path_metrics(
                backbone,
                head,
                val_pos,
                name_to_id,
                id_to_name,
                use_text_features=use_text_features,
                node_features=node_features,
                text_features_ab=text_features_ab_val,
                text_features_bc=text_features_bc_val,
                device=device,
            )
            last_val_metrics = val_metrics
            print(f"  [VAL] Path Acc: {val_metrics['path_acc']:.4f}, Path F1: {val_metrics['path_f1']:.4f}")
            print(f"  [VAL] A-B  Acc: {val_metrics['ab_acc']:.4f}, F1: {val_metrics['ab_f1']:.4f}, "
                  f"AUC: {val_metrics['ab_auc_roc']:.4f}, MCC: {val_metrics['ab_mcc']:.4f}")
            print(f"  [VAL] B-C  Acc: {val_metrics['bc_acc']:.4f}, F1: {val_metrics['bc_f1']:.4f}, "
                  f"AUC: {val_metrics['bc_auc_roc']:.4f}, MCC: {val_metrics['bc_mcc']:.4f}")
            if val_metrics["path_acc"] > best_val_path_acc:
                best_val_path_acc = val_metrics["path_acc"]
                best_epoch = epoch + 1
                best_val_metrics = val_metrics.copy()
                print(f"  New best Path Acc: {best_val_path_acc:.4f}")

    # 8) 测试集评估
    print("\n" + "=" * 80)
    print("[8/8] Final Evaluation on Test Set (with Cascading)")
    print("=" * 80)
    test_metrics = evaluate_path_metrics(
        backbone,
        head,
        test_pos,
        name_to_id,
        id_to_name,
        use_text_features=use_text_features,
        node_features=node_features,
        text_features_ab=text_features_ab_test,
        text_features_bc=text_features_bc_test,
        device=device,
    )
    print(f"\n[TEST] Path Accuracy:  {test_metrics['path_acc']:.4f}")
    print(f"[TEST] Path F1-Score:  {test_metrics['path_f1']:.4f}")
    print(f"      A-B Acc: {test_metrics['ab_acc']:.4f}, F1: {test_metrics['ab_f1']:.4f}, "
          f"AUC: {test_metrics['ab_auc_roc']:.4f}, MCC: {test_metrics['ab_mcc']:.4f}")
    print(f"      B-C Acc: {test_metrics['bc_acc']:.4f}, F1: {test_metrics['bc_f1']:.4f}, "
          f"AUC: {test_metrics['bc_auc_roc']:.4f}, MCC: {test_metrics['bc_mcc']:.4f}")
    print(f"\nBest validation Path Acc: {best_val_path_acc:.4f} (epoch {best_epoch})")

    if best_val_metrics is None:
        if last_val_metrics is not None:
            best_val_metrics = last_val_metrics
            best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
            best_epoch = epochs
        else:
            best_val_metrics = evaluate_path_metrics(
                backbone,
                head,
                val_pos,
                name_to_id,
                id_to_name,
                use_text_features=use_text_features,
                node_features=node_features,
                text_features_ab=text_features_ab_val,
                text_features_bc=text_features_bc_val,
                device=device,
            )
            best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
            best_epoch = epochs

    return {
        "nalp_backbone": backbone,
        "head": head,
        "graph": g,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "test_metrics": test_metrics,
        "val_metrics": best_val_metrics,
        "best_val_metrics": best_val_metrics,
        "best_val_path_acc": best_val_path_acc,
        "best_epoch": best_epoch
    }


