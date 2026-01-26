"""
Training pipeline for N-ComplEx (ComplEx for N-ary KGs)
流程结构兼容StarE/NaLP/RGCN：直接训练、验证、评测、负采样、支持节点/文本特征
"""
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

from .models import NComplEx, NComplExPredictor
from ..graph_build import build_dgl_graph
from ..path_data import enumerate_graph_paths, split_paths, generate_negatives, select_few_shot
from ..db_mongo import get_pair_evidence_embedding
from ..config import NODE_NAME_FIELD, NEIGHBOR_AGG_METHOD, LABEL_SLC, SBERT_DIM
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

def build_relation_type_map_from_paths(paths: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """提取N-ary路径各段rel_AB/rel_BC类别，建立类别映射"""
    relation_types = set()
    for p in paths:
        if "rel_AB" in p: relation_types.add(p["rel_AB"])
        if "rel_BC" in p: relation_types.add(p["rel_BC"])
    sorted_types = sorted(list(relation_types))
    name_to_id = {name: idx for idx, name in enumerate(sorted_types)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return name_to_id, id_to_name

def prepare_batches(paths: List[Dict], name_to_id: Dict[str, int]):
    if len(paths) == 0:
        return {"A": torch.tensor([]), "B": torch.tensor([]), "C": torch.tensor([]), "Event": torch.tensor([]), "y_ab": torch.tensor([]), "y_bc": torch.tensor([])}
    A = [p["A"] for p in paths]
    B = [p["B"] for p in paths]
    C = [p["C"] for p in paths]
    Event = [p.get("Event", -1) for p in paths]  # 可选中转node
    y_ab = [name_to_id[p["rel_AB"]] for p in paths]
    y_bc = [name_to_id[p["rel_BC"]] for p in paths]
    return {
        "A": torch.tensor(A, dtype=torch.long),
        "B": torch.tensor(B, dtype=torch.long),
        "C": torch.tensor(C, dtype=torch.long),
        "Event": torch.tensor(Event, dtype=torch.long),
        "y_ab": torch.tensor(y_ab, dtype=torch.long),
        "y_bc": torch.tensor(y_bc, dtype=torch.long)
    }

def move_batch_to_device(batch: Dict[str, torch.Tensor], device):
    if batch is None:
        return None
    return {key: tensor.to(device) for key, tensor in batch.items()}

def evaluate_path_metrics(ncomplex_model, predictor, samples, name_to_id, id_to_name, device, num_relation_classes):
    if len(samples) == 0:
        return {"path_acc":0.0, "path_f1":0.0, "ab_acc":0.0, "ab_f1":0.0, "ab_auc_roc":0.0, "ab_mcc":0.0, "bc_acc":0.0, "bc_f1":0.0, "bc_auc_roc":0.0, "bc_mcc":0.0}
    all_true_ab, all_pred_ab, all_scores_ab = [], [], []
    all_true_bc, all_pred_bc, all_scores_bc = [], [], []
    y_true_path_4class, y_pred_path_4class = [], []
    ncomplex_model.eval()
    predictor.eval()
    with torch.no_grad():
        for p in samples:
            A_idx = torch.tensor([p["A"]], dtype=torch.long, device=device)
            B_idx = torch.tensor([p["B"]], dtype=torch.long, device=device)
            C_idx = torch.tensor([p["C"]], dtype=torch.long, device=device)
            true_ab = name_to_id[p["rel_AB"]]
            true_bc = name_to_id[p["rel_BC"]]

            ab_scores = ncomplex_model.score_all_relations(A_idx, B_idx)
            ab_logits = predictor(ab_scores)
            ab_probs = F.softmax(ab_logits, dim=-1).cpu().numpy()[0]
            pred_ab = int(np.argmax(ab_probs))

            ab_weights = F.one_hot(torch.tensor([pred_ab], dtype=torch.long, device=device), num_classes=num_relation_classes).float()
            rel_re, rel_im = ncomplex_model.relation_embedding_from_weights(ab_weights)
            bc_scores = ncomplex_model.score_all_relations_with_context(
                subj=B_idx,
                obj=C_idx,
                context_re=rel_re,
                context_im=rel_im
            )
            bc_logits = predictor(bc_scores)
            bc_probs = F.softmax(bc_logits, dim=-1).cpu().numpy()[0]
            pred_bc = int(np.argmax(bc_probs))

            all_true_ab.append(true_ab)
            all_pred_ab.append(pred_ab)
            all_scores_ab.append(ab_probs[1] if len(ab_probs)>1 else ab_probs[0])
            all_true_bc.append(true_bc)
            all_pred_bc.append(pred_bc)
            all_scores_bc.append(bc_probs[1] if len(bc_probs)>1 else bc_probs[0])

            y_true_path_4class.append(0)
            is_ab_corr = (pred_ab == true_ab)
            pred_bc_corr = (pred_bc == true_bc)
            if is_ab_corr and pred_bc_corr:
                y_pred_path_4class.append(0)
            elif is_ab_corr and not pred_bc_corr:
                y_pred_path_4class.append(1)
            elif (not is_ab_corr) and pred_bc_corr:
                y_pred_path_4class.append(2)
            else:
                y_pred_path_4class.append(3)
    acc_ab = accuracy_score(all_true_ab, all_pred_ab)
    f1_ab = f1_score(all_true_ab, all_pred_ab, average='macro', zero_division=0)
    mcc_ab = matthews_corrcoef(all_true_ab, all_pred_ab)
    auc_roc_ab = 0.0
    try:
        if len(np.unique(all_true_ab))>1:
            auc_roc_ab = roc_auc_score(all_true_ab, all_scores_ab)
        else:
            auc_roc_ab = acc_ab
    except ValueError:
        pass
    acc_bc = accuracy_score(all_true_bc, all_pred_bc)
    f1_bc = f1_score(all_true_bc, all_pred_bc, average='macro', zero_division=0)
    mcc_bc = matthews_corrcoef(all_true_bc, all_pred_bc)
    auc_roc_bc = 0.0
    try:
        if len(np.unique(all_true_bc))>1:
            auc_roc_bc = roc_auc_score(all_true_bc, all_scores_bc)
        else:
            auc_roc_bc = acc_bc
    except ValueError:
        pass
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

    path_acc = accuracy_score(y_true_path_4class, y_pred_path_4class)
    path_f1 = f1_score(y_true_path_4class, y_pred_path_4class, labels=[0], average='macro', zero_division=0)
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
    dropout: float,
    val_every: int,
    use_node_features: bool,
    use_text_features: bool,
    device: str,
    *,
    train_paths: Optional[List[Dict]] = None,
    val_paths: Optional[List[Dict]] = None,
    test_paths: Optional[List[Dict]] = None,
    skip_negatives: bool = False,
    few_shot_k: Optional[int] = None,
    few_shot_balance: Optional[str] = None,
    seed: int = 0,
):
    print("="*60)
    print("N-ComplEx Training Pipeline for N-ary KG")
    print("="*60)
    print("[1/8] Building DGL graph...")
    g = build_dgl_graph()[0]
    print(f"  ✓ Graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
    if train_paths is None and val_paths is None and test_paths is None:
        print("[2/8] Enumerating N-ary paths...")
        all_paths = enumerate_graph_paths()
        print(f"  ✓ {len(all_paths)} paths")
        if len(all_paths) == 0:
            print("  ✗ No paths found!")
            return None
        print("[3/8] Building relation mappings...")
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
        combined_paths = train_pos + val_pos + test_pos
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

    if skip_negatives:
        print("[5/8] Skipping negative sampling (skip_negatives=True)...")
        train_neg = []
    else:
        print("[5/8] Generating negatives...")
        train_neg = generate_negatives(train_pos, all_paths)
        print(f"  ✓ Neg samples: {len(train_neg)}")
    print("[6/8] Initializing model...")
    num_entities = g.num_nodes()
    ncomplex_model = NComplEx(
        num_entities=num_entities,
        num_relations=num_relation_classes,
        embedding_dim=embedding_dim,
        dropout=dropout,
        node_feat_dim=g.ndata["feat"].shape[1] if use_node_features and "feat" in g.ndata else 0,
        use_node_features=use_node_features,
        use_text_features=use_text_features,
        text_dim=SBERT_DIM if use_text_features else 0
    ).to(device)
    predictor = NComplExPredictor(
        embedding_dim=embedding_dim, num_relation_classes=num_relation_classes, dropout=dropout
    ).to(device)
    optim = torch.optim.Adam(list(ncomplex_model.parameters()) + list(predictor.parameters()), lr=lr)
    train_batch_pos = move_batch_to_device(prepare_batches(train_pos, name_to_id), device)
    train_batch_neg = move_batch_to_device(prepare_batches(train_neg, name_to_id), device) if len(train_neg)>0 else None
    print("[7/8] Training...")
    best_val_path_acc = 0.0
    best_epoch = 0
    best_val_metrics = None
    last_val_metrics = None
    for epoch in range(epochs):
        ncomplex_model.train()
        predictor.train()
        optim.zero_grad()
        # 正样本（A-Event-B, B-Event-C）；可进一步扩展支持N-ary历史
        ab_scores_pos = ncomplex_model.score_all_relations(train_batch_pos["A"], train_batch_pos["B"])
        logits_ab_pos = predictor(ab_scores_pos)
        ab_probs_pos = F.softmax(logits_ab_pos, dim=-1)
        ab_embeddings_re, ab_embeddings_im = ncomplex_model.relation_embedding_from_weights(ab_probs_pos)
        bc_scores_pos = ncomplex_model.score_all_relations_with_context(
            subj=train_batch_pos["B"],
            obj=train_batch_pos["C"],
            context_re=ab_embeddings_re,
            context_im=ab_embeddings_im
        )
        logits_bc_pos = predictor(bc_scores_pos)
        loss_ab = F.cross_entropy(logits_ab_pos, train_batch_pos["y_ab"])
        loss_bc = F.cross_entropy(logits_bc_pos, train_batch_pos["y_bc"])
        loss_pos = loss_ab + loss_bc
        if train_batch_neg is not None and train_batch_neg["A"].shape[0]>0:
            ab_scores_neg = ncomplex_model.score_all_relations(train_batch_neg["A"], train_batch_neg["B"])
            logits_ab_neg = predictor(ab_scores_neg)
            ab_probs_neg = F.softmax(logits_ab_neg, dim=-1)
            ab_embeddings_re_neg, ab_embeddings_im_neg = ncomplex_model.relation_embedding_from_weights(ab_probs_neg)
            bc_scores_neg = ncomplex_model.score_all_relations_with_context(
                subj=train_batch_neg["B"],
                obj=train_batch_neg["C"],
                context_re=ab_embeddings_re_neg,
                context_im=ab_embeddings_im_neg
            )
            logits_bc_neg = predictor(bc_scores_neg)
            loss_neg_ab = F.cross_entropy(logits_ab_neg, train_batch_neg["y_ab"])
            loss_neg_bc = F.cross_entropy(logits_bc_neg, train_batch_neg["y_bc"])
            loss = loss_pos + 0.5 * (loss_neg_ab + loss_neg_bc)
        else:
            loss = loss_pos
        loss.backward()
        optim.step()
        # 验证
        if (epoch+1)%val_every==0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
            val_metrics = evaluate_path_metrics(
                ncomplex_model,
                predictor,
                val_pos,
                name_to_id,
                id_to_name,
                device=device,
                num_relation_classes=num_relation_classes
            )
            last_val_metrics = val_metrics
            print(f"  [VAL] Path Acc: {val_metrics['path_acc']:.4f}, Path F1: {val_metrics['path_f1']:.4f}")
            print(f"  [VAL] A-B  Acc: {val_metrics['ab_acc']:.4f}, F1: {val_metrics['ab_f1']:.4f}, "
                  f"AUC: {val_metrics['ab_auc_roc']:.4f}, MCC: {val_metrics['ab_mcc']:.4f}")
            print(f"  [VAL] B-C  Acc: {val_metrics['bc_acc']:.4f}, F1: {val_metrics['bc_f1']:.4f}, "
                  f"AUC: {val_metrics['bc_auc_roc']:.4f}, MCC: {val_metrics['bc_mcc']:.4f}")
            if val_metrics['path_acc'] > best_val_path_acc:
                best_val_path_acc = val_metrics['path_acc']
                best_epoch = epoch + 1
                best_val_metrics = val_metrics.copy()
                print(f"  New best Path Acc: {best_val_path_acc:.4f}")
    print("="*60)
    print("[8/8] Final Evaluation on Test Set")
    test_metrics = evaluate_path_metrics(
        ncomplex_model,
        predictor,
        test_pos,
        name_to_id,
        id_to_name,
        device=device,
        num_relation_classes=num_relation_classes
    )
    print(f"[TEST] Path Acc: {test_metrics['path_acc']:.4f}")
    print(f"[TEST] Path F1 : {test_metrics['path_f1']:.4f}")
    print(f"[TEST] A-B Comp: Acc {test_metrics['ab_acc']:.4f}, F1 {test_metrics['ab_f1']:.4f}, "
          f"AUC {test_metrics['ab_auc_roc']:.4f}, MCC {test_metrics['ab_mcc']:.4f}")
    print(f"[TEST] B-C Comp: Acc {test_metrics['bc_acc']:.4f}, F1 {test_metrics['bc_f1']:.4f}, "
          f"AUC {test_metrics['bc_auc_roc']:.4f}, MCC {test_metrics['bc_mcc']:.4f}")
    print(f"Best val Path Acc: {best_val_path_acc:.4f} (epoch {best_epoch})")

    if best_val_metrics is None:
        if last_val_metrics is not None:
            best_val_metrics = last_val_metrics
            best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
            best_epoch = epochs
        else:
            best_val_metrics = evaluate_path_metrics(
                ncomplex_model,
                predictor,
                val_pos,
                name_to_id,
                id_to_name,
                device=device,
                num_relation_classes=num_relation_classes
            )
            best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
            best_epoch = epochs
    return {
        "model": ncomplex_model,
        "predictor": predictor,
        "graph": g,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "test_metrics": test_metrics,
        "val_metrics": best_val_metrics,
        "best_val_metrics": best_val_metrics,
        "best_val_path_acc": best_val_path_acc,
        "best_epoch": best_epoch
    }
