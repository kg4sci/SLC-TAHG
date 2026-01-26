"""
Training pipeline for the NS-HART cascading predictor.

This adapts the native NS-HART architecture so that it follows the same
execution path as the GRAN baseline:
- Enumerate cascaded paths (A, Event, B, C)
- Stage AB: predict rel_AB from the full path context
- Stage BC: condition on the predicted rel_AB and predict rel_BC
- Optional node features and text evidence are fused consistently with NaLP/GRAN
"""

from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .models import HARTCascadingPredictor
from ..graph_build import build_dgl_graph
from ..path_data import enumerate_graph_paths, split_paths, generate_negatives, select_few_shot
from ..db_mongo import get_pair_evidence_embedding
from ..config import NODE_NAME_FIELD, SBERT_MODEL_NAME, SBERT_DIM
from ..device_utils import resolve_device
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)


def load_text_features_for_pairs(
    paths: List[Dict],
    name_to_id: Dict[str, int],
    embed_dim: int = SBERT_DIM,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Fetch SBERT evidence embeddings for A-B and B-C pairs.
    Matches the behaviour used by GRAN / NaLP pipelines.
    """
    text_features_ab: Dict[int, torch.Tensor] = {}
    text_features_bc: Dict[int, torch.Tensor] = {}

    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(SBERT_MODEL_NAME)
    except Exception:
        encoder = None

    for idx, p in enumerate(paths):
        try:
            a_name = p.get("A_name") or p.get(NODE_NAME_FIELD, "")
            b_name = p.get("B_name", "")
            c_name = p.get("C_name", "")

            if a_name and b_name:
                _, emb_ab = get_pair_evidence_embedding(
                    "slc_pathway", a_name, b_name, encoder=encoder, reencode=True
                )
                if emb_ab is None:
                    emb_ab = [0.0] * embed_dim
                text_features_ab[idx] = torch.tensor(emb_ab, dtype=torch.float32)

            if b_name and c_name:
                _, emb_bc = get_pair_evidence_embedding(
                    "pathway_disease", b_name, c_name, encoder=encoder, reencode=True
                )
                if emb_bc is None:
                    emb_bc = [0.0] * embed_dim
                text_features_bc[idx] = torch.tensor(emb_bc, dtype=torch.float32)

        except Exception as exc:
            print(f"  Warning: failed to fetch text evidence for path {idx}: {exc}")
            text_features_ab.setdefault(
                idx, torch.zeros(embed_dim, dtype=torch.float32)
            )
            text_features_bc.setdefault(
                idx, torch.zeros(embed_dim, dtype=torch.float32)
            )

    print(f"  ✓ Loaded text features for {len(text_features_ab)} A-B pairs")
    print(f"  ✓ Loaded text features for {len(text_features_bc)} B-C pairs")
    return text_features_ab, text_features_bc


def build_relation_type_map_from_paths(
    paths: List[Dict],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    relation_types = set()
    for p in paths:
        if "rel_AB" in p:
            relation_types.add(p["rel_AB"])
        if "rel_BC" in p:
            relation_types.add(p["rel_BC"])
    sorted_types = sorted(relation_types)
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
    hart_model: HARTCascadingPredictor,
    g,
    samples: List[Dict],
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
    use_text_features: bool,
    text_features_ab: Dict[int, torch.Tensor] = None,
    text_features_bc: Dict[int, torch.Tensor] = None,
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

    hart_model.eval()

    num_entities = hart_model.num_entities
    num_nodes = g.num_nodes()
    node_features = g.ndata.get("feat", None)

    if node_features is not None:
        node_features = node_features.to(device)
        node_feat_dim = node_features.shape[1]
        if num_entities > num_nodes:
            expanded = torch.zeros(
                num_entities,
                node_feat_dim,
                dtype=node_features.dtype,
                device=device,
            )
            expanded[:num_nodes] = node_features
            node_features = expanded
    elif getattr(hart_model, "node_feat_encoder", None) is not None:
        node_feat_dim = hart_model.node_feat_encoder[0].in_features
        node_features = torch.zeros(
            num_entities, node_feat_dim, dtype=torch.float32, device=device
        )
    else:
        node_features = None

    all_true_ab, all_pred_ab, all_scores_ab = [], [], []
    all_true_bc, all_pred_bc, all_scores_bc = [], [], []
    y_true_path_4class = []
    y_pred_path_4class = []

    with torch.no_grad():
        for idx, p in enumerate(samples):
            A, B, C = p["A"], p["B"], p["C"]
            Event = p.get("Event", p["A"])
            true_ab = name_to_id[p["rel_AB"]]
            true_bc = name_to_id[p["rel_BC"]]

            a_ids = torch.tensor([A], dtype=torch.long, device=device)
            b_ids = torch.tensor([B], dtype=torch.long, device=device)
            c_ids = torch.tensor([C], dtype=torch.long, device=device)
            event_ids = torch.tensor([Event], dtype=torch.long, device=device)

            text_ab = (
                text_features_ab[idx].unsqueeze(0).to(device)
                if use_text_features and text_features_ab and idx in text_features_ab
                else None
            )
            text_bc = (
                text_features_bc[idx].unsqueeze(0).to(device)
                if use_text_features and text_features_bc and idx in text_features_bc
                else None
            )

            logits_ab, logits_bc, _ = hart_model(
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

            y_true_path_4class.append(0)
            is_ab_correct = pred_ab == true_ab
            is_bc_correct = pred_bc == true_bc
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
    encoder_hidden_dim: int,
    encoder_layers: int,
    encoder_heads: int,
    dropout: float,
    val_every: int,
    use_text_features: bool,
    use_node_features: bool,
    *,
    device: Optional[Union[str, torch.device]] = None,
    weight_decay: float = 0.0,
    train_paths: Optional[List[Dict]] = None,
    val_paths: Optional[List[Dict]] = None,
    test_paths: Optional[List[Dict]] = None,
    skip_negatives: bool = False,
    few_shot_k: Optional[int] = None,
    few_shot_balance: Optional[str] = None,
    seed: int = 0,
):
    device = resolve_device(device)
    print(f"[Device] Using {device} for NS-HART pipeline")
    # 图保持在CPU上，避免DGL CUDA依赖问题
    print(f"[Device] Graph will stay on CPU (to avoid DGL CUDA dependency)")

    print("\n[1/8] Building DGL graph from Neo4j...")
    g = build_dgl_graph()[0]  # 图保持在CPU上
    print(f"  ✓ Graph built: {g.num_nodes()} nodes, {g.num_edges()} edges")

    if train_paths is None and val_paths is None and test_paths is None:
        print("\n[2/8] Enumerating N-ary paths...")
        all_paths = enumerate_graph_paths()
        print(f"  ✓ Found {len(all_paths)} paths")
        if len(all_paths) == 0:
            print("  ✗ No paths found!")
            return None

        print("\n[3/8] Building relation type mappings from paths...")
        name_to_id, id_to_name = build_relation_type_map_from_paths(all_paths)
        num_relation_classes = len(name_to_id)
        print(f"  ✓ {num_relation_classes} relation classes")

        print("\n[4/8] Splitting data into train/val/test...")
        train_pos, val_pos, test_pos = split_paths(
            all_paths, train_ratio=0.7, val_ratio=0.15
        )
        if few_shot_k:
            train_pos = select_few_shot(train_pos, few_shot_k, seed=seed, balance_by=few_shot_balance)
        print(
            f"  ✓ Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}"
        )
    else:
        if train_paths is None or val_paths is None:
            raise ValueError("train_paths and val_paths must be provided when using external splits")

        print("\n[2/8] Using externally provided train/val/test splits...")
        train_pos = list(train_paths)
        val_pos = list(val_paths)
        test_pos = list(test_paths) if test_paths is not None else list(val_pos)
        print(
            f"  ✓ Provided splits - Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}"
        )

        combined_paths = train_pos + val_pos + [p for p in test_pos if p not in val_pos]
        if len(combined_paths) == 0:
            print("  ✗ No paths supplied by caller!")
            return None

        print("\n[3/8] Building relation type mappings from provided paths...")
        name_to_id, id_to_name = build_relation_type_map_from_paths(combined_paths)
        num_relation_classes = len(name_to_id)
        print(f"  ✓ {num_relation_classes} relation classes")

        all_paths = combined_paths
        if few_shot_k:
            train_pos = select_few_shot(train_pos, few_shot_k, seed=seed, balance_by=few_shot_balance)

    if skip_negatives:
        print("\n[5/8] Skipping negative sampling (skip_negatives=True)...")
        train_neg = []
    else:
        print("\n[5/8] Generating negative samples...")
        train_neg = generate_negatives(train_pos, all_paths)
        print(f"  ✓ Generated {len(train_neg)} negative samples")

    print("\n[5.5/8] Ensuring text evidence availability...")
    text_features_ab_train = None
    text_features_bc_train = None
    text_features_ab_val = None
    text_features_bc_val = None
    text_features_ab_test = None
    text_features_bc_test = None

    if use_text_features:
        print("  Loading training text features...")
        text_features_ab_train, text_features_bc_train = load_text_features_for_pairs(
            train_pos, name_to_id, embed_dim=SBERT_DIM
        )
        print("  Loading validation text features...")
        text_features_ab_val, text_features_bc_val = load_text_features_for_pairs(
            val_pos, name_to_id, embed_dim=SBERT_DIM
        )
        print("  Loading test text features...")
        text_features_ab_test, text_features_bc_test = load_text_features_for_pairs(
            test_pos, name_to_id, embed_dim=SBERT_DIM
        )

    all_entity_ids = []
    for p in all_paths:
        all_entity_ids.extend([p["A"], p["B"], p["C"]])
        if p.get("Event") is not None:
            all_entity_ids.append(p["Event"])

    if not all_entity_ids:
        print("  ✗ No entity IDs found in paths!")
        return None

    min_entity_id = min(all_entity_ids)
    max_entity_id = max(all_entity_ids)
    if min_entity_id < 0:
        print(
            f"  ⚠ Warning: Found negative entity IDs (min={min_entity_id}). This may cause issues."
        )

    num_nodes = g.num_nodes()
    num_entities = max(num_nodes, max_entity_id + 1)
    print(
        f"  ✓ Graph nodes: {num_nodes}, Entity ID range: [{min_entity_id}, {max_entity_id}], "
        f"Using num_entities: {num_entities}"
    )

    node_feat_dim = 0
    if use_node_features and "feat" in g.ndata:
        node_feat_dim = g.ndata["feat"].shape[1]
        print(f"  ✓ Node features available: dim={node_feat_dim}")
    elif use_node_features:
        print("  ⚠ Requested node features but graph provides none.")

    print("\n[6/8] Initializing HART cascading predictor...")
    hart_model = HARTCascadingPredictor(
        num_entities=num_entities,
        num_relation_classes=num_relation_classes,
        embedding_dim=embedding_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_layers=encoder_layers,
        encoder_heads=encoder_heads,
        dropout=dropout,
        node_feat_dim=node_feat_dim,
        text_dim=SBERT_DIM if use_text_features else 0,
        use_text_features=use_text_features,
    ).to(device)

    optimizer = torch.optim.Adam(
        hart_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_batch_pos = prepare_batches(train_pos, name_to_id)
    train_batch_neg = prepare_batches(train_neg, name_to_id) if train_neg else None

    node_features = g.ndata.get("feat", None)
    if node_features is not None:
        node_features = node_features.to(device)
        if num_entities > num_nodes:
            expanded_node_features = torch.zeros(
                num_entities,
                node_features.shape[1],
                dtype=node_features.dtype,
                device=device,
            )
            expanded_node_features[:num_nodes] = node_features
            node_features = expanded_node_features
            print(
                f"  ✓ Expanded node features to [{num_entities}, {node_features.shape[1]}]"
            )
    elif node_feat_dim > 0:
        node_features = torch.zeros(
            num_entities, node_feat_dim, dtype=torch.float32, device=device
        )
        print(f"  ✓ Created zero node features tensor: [{num_entities}, {node_feat_dim}]")

    print("\n[7/8] Training with cascading supervision...")
    best_val_path_acc = 0.0
    best_epoch = 0
    best_val_metrics = None
    last_val_metrics = None

    for epoch in range(epochs):
        hart_model.train()
        optimizer.zero_grad()

        a_ids = train_batch_pos["A"].to(device)
        b_ids = train_batch_pos["B"].to(device)
        c_ids = train_batch_pos["C"].to(device)
        event_ids = train_batch_pos["Event"].to(device)

        text_ab_batch = None
        text_bc_batch = None
        if use_text_features and text_features_ab_train is not None:
            text_ab_batch = torch.stack(
                [
                    text_features_ab_train.get(
                        idx, torch.zeros(SBERT_DIM, dtype=torch.float32)
                    )
                    for idx in range(len(train_pos))
                ]
            ).to(device)

        if use_text_features and text_features_bc_train is not None:
            text_bc_batch = torch.stack(
                [
                    text_features_bc_train.get(
                        idx, torch.zeros(SBERT_DIM, dtype=torch.float32)
                    )
                    for idx in range(len(train_pos))
                ]
            ).to(device)

        logits_ab_pos, logits_bc_pos, _ = hart_model(
            a_ids,
            b_ids,
            c_ids,
            event_ids,
            training=True,
            node_features=node_features,
            text_features_ab=text_ab_batch,
            text_features_bc=text_bc_batch,
        )
        loss_ab = F.cross_entropy(logits_ab_pos, train_batch_pos["y_ab"].to(device))
        loss_bc = F.cross_entropy(logits_bc_pos, train_batch_pos["y_bc"].to(device))
        loss_pos = loss_ab + loss_bc

        if train_batch_neg is not None and len(train_neg) > 0:
            a_ids_neg = train_batch_neg["A"].to(device)
            b_ids_neg = train_batch_neg["B"].to(device)
            c_ids_neg = train_batch_neg["C"].to(device)
            event_ids_neg = train_batch_neg["Event"].to(device)

            text_ab_neg = (
                torch.zeros(len(train_neg), SBERT_DIM, dtype=torch.float32, device=device)
                if use_text_features
                else None
            )
            text_bc_neg = (
                torch.zeros(len(train_neg), SBERT_DIM, dtype=torch.float32, device=device)
                if use_text_features
                else None
            )

            logits_ab_neg, logits_bc_neg, _ = hart_model(
                a_ids_neg,
                b_ids_neg,
                c_ids_neg,
                event_ids_neg,
                training=True,
                node_features=node_features,
                text_features_ab=text_ab_neg,
                text_features_bc=text_bc_neg,
            )
            loss_ab_neg = F.cross_entropy(
                logits_ab_neg, train_batch_neg["y_ab"].to(device)
            )
            loss_bc_neg = F.cross_entropy(
                logits_bc_neg, train_batch_neg["y_bc"].to(device)
            )
            loss = loss_pos + 0.5 * (loss_ab_neg + loss_bc_neg)
        else:
            loss = loss_pos

        loss.backward()
        optimizer.step()

        if (epoch + 1) % val_every == 0:
            print(f"\nEpoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
            val_metrics = evaluate_path_metrics(
                hart_model,
                g,
                val_pos,
                name_to_id,
                id_to_name,
                use_text_features=use_text_features,
                text_features_ab=text_features_ab_val,
                text_features_bc=text_features_bc_val,
                device=device,
            )
            last_val_metrics = val_metrics
            print(
                f"  [VAL] Path Acc: {val_metrics['path_acc']:.4f}, "
                f"Path F1: {val_metrics['path_f1']:.4f}"
            )
            print(
                f"  [VAL] A-B  Acc: {val_metrics['ab_acc']:.4f}, F1: {val_metrics['ab_f1']:.4f}, "
                f"AUC: {val_metrics['ab_auc_roc']:.4f}, MCC: {val_metrics['ab_mcc']:.4f}"
            )
            print(
                f"  [VAL] B-C  Acc: {val_metrics['bc_acc']:.4f}, F1: {val_metrics['bc_f1']:.4f}, "
                f"AUC: {val_metrics['bc_auc_roc']:.4f}, MCC: {val_metrics['bc_mcc']:.4f}"
            )
            if val_metrics["path_acc"] > best_val_path_acc:
                best_val_path_acc = val_metrics["path_acc"]
                best_epoch = epoch + 1
                best_val_metrics = val_metrics.copy()
                print(f"  ✓ New best Path Acc: {best_val_path_acc:.4f}")

    print("\n" + "=" * 80)
    print("[8/8] Final evaluation on test set")
    print("=" * 80)
    test_metrics = evaluate_path_metrics(
        hart_model,
        g,
        test_pos,
        name_to_id,
        id_to_name,
        use_text_features=use_text_features,
        text_features_ab=text_features_ab_test,
        text_features_bc=text_features_bc_test,
        device=device,
    )
    print(f"\n[TEST] Path Accuracy:  {test_metrics['path_acc']:.4f}")
    print(f"[TEST] Path F1-Score:  {test_metrics['path_f1']:.4f}")
    print(
        f"      A-B Acc: {test_metrics['ab_acc']:.4f}, F1: {test_metrics['ab_f1']:.4f}, "
        f"AUC: {test_metrics['ab_auc_roc']:.4f}, MCC: {test_metrics['ab_mcc']:.4f}"
    )
    print(
        f"      B-C Acc: {test_metrics['bc_acc']:.4f}, F1: {test_metrics['bc_f1']:.4f}, "
        f"AUC: {test_metrics['bc_auc_roc']:.4f}, MCC: {test_metrics['bc_mcc']:.4f}"
    )
    print(f"\nBest validation Path Acc: {best_val_path_acc:.4f} (epoch {best_epoch})")

    if best_val_metrics is None:
        if last_val_metrics is not None:
            best_val_metrics = last_val_metrics
            best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
            best_epoch = epochs
        else:
            best_val_metrics = evaluate_path_metrics(
                hart_model,
                g,
                val_pos,
                name_to_id,
                id_to_name,
                use_text_features=use_text_features,
                text_features_ab=text_features_ab_val,
                text_features_bc=text_features_bc_val,
                device=device,
            )
            best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
            best_epoch = epochs

    return {
        "hart_model": hart_model,
        "graph": g,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "test_metrics": test_metrics,
        "val_metrics": best_val_metrics,
        "best_val_metrics": best_val_metrics,
        "best_val_path_acc": best_val_path_acc,
        "best_epoch": best_epoch,
    }


__all__ = [
    "train_pipeline_from_graph",
    "evaluate_path_metrics",
]

