import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import numpy as np
from .models import RAMCascadingFactClassifier
from ..graph_build import build_dgl_graph
from ..path_data import enumerate_graph_paths, split_paths, generate_negatives, select_few_shot
from ..config import SBERT_MODEL_NAME, SBERT_DIM
from ..db_mongo import get_pair_evidence_embedding
from ..device_utils import resolve_device
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
import dgl

def build_relation_type_map_from_paths(paths: List[Dict]):
    relation_types = set()
    for p in paths:
        if 'rel_AB' in p: relation_types.add(p['rel_AB'])
        if 'rel_BC' in p: relation_types.add(p['rel_BC'])
    sorted_types = sorted(list(relation_types))
    name_to_id = {name: idx for idx, name in enumerate(sorted_types)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return name_to_id, id_to_name

def prepare_batches(paths: List[Dict], name_to_id: Dict[str, int]):
    if len(paths) == 0:
        return {k: torch.tensor([], dtype=torch.long) for k in ['A','B','C','Event','y_ab','y_bc']}
    A = [p["A"] for p in paths]
    B = [p["B"] for p in paths]
    C = [p["C"] for p in paths]
    # 防御：缺失事件用 0（与 Embedding 的 padding_idx 对齐），并且不允许负索引
    Event = [max(p.get("Event", 0), 0) for p in paths]
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

def train_pipeline_from_graph(
    epochs: int,
    lr: float,
    embedding_dim: int,
    n_parts: int,
    max_ary: int,
    dropout: float,
    val_every: int,
    batch_size: int = 256,
    use_node_features: bool=True,
    use_text_features: bool=True,
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
    device = resolve_device(device)
    print(f"[Device] Using {device} for RAM pipeline")
    # 图保持在CPU上，避免DGL CUDA依赖问题
    graph_device = torch.device("cpu")
    print(f"[Device] Graph will stay on CPU (to avoid DGL CUDA dependency)")

    print("[1/8] Building DGL graph...")
    g = build_dgl_graph()[0]  # 图保持在CPU上
    print(f"  ✓ Graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
    if train_paths is None and val_paths is None and test_paths is None:
        print("[2/8] Enumerating N-ary paths...")
        all_paths = enumerate_graph_paths()
        print(f"  ✓ Found {len(all_paths)} paths")
        if len(all_paths)==0:
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
        print(f"  ✓ Provided splits - Train {len(train_pos)}, Val {len(val_pos)}, Test {len(test_pos)}")

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

    if skip_negatives:
        print("[5/8] Skipping negative sampling (skip_negatives=True)...")
        train_neg = []
    else:
        print("[5/8] Generating negatives...")
        train_neg = generate_negatives(train_pos, all_paths)
        print(f"  ✓ Neg samples: {len(train_neg)}")
    print("[6/8] Initializing model...")
    num_entities = g.num_nodes()
    node_feat_dim = g.ndata["feat"].shape[1] if use_node_features and "feat" in g.ndata else 0
    text_dim = SBERT_DIM if use_text_features else 0
    model = RAMCascadingFactClassifier(
        num_entities=num_entities,
        num_relation_classes=num_relation_classes,
        embedding_dim=embedding_dim,
        n_parts=n_parts,
        max_ary=max_ary,
        dropout=dropout,
        node_feat_dim=node_feat_dim,
        text_dim=text_dim,
        use_text_features=use_text_features,
        device=device
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_batch_pos = prepare_batches(train_pos, name_to_id)
    train_batch_neg = prepare_batches(train_neg, name_to_id) if len(train_neg)>0 else None
    print("[7/8] Training...")

    # 预计算结构嵌入（简单一层邻居均值聚合），仅当节点特征可用
    # 注意：图保持在CPU上，只复制节点特征到GPU进行计算
    h_struct = None
    if use_node_features and "feat" in g.ndata:
        with g.local_scope():
            # 图保持在CPU，只复制节点特征到GPU
            g.ndata['h'] = g.ndata['feat']  # 保持在CPU
            dgl.function.copy_u('h', 'm')
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'neigh'))
            h_neigh = g.ndata.get('neigh', torch.zeros_like(g.ndata['h']))
            h_struct_cpu = (g.ndata['h'] + h_neigh) / 2.0  # [N, node_feat_dim] on CPU
            # 将结果复制到GPU（如果需要）
            h_struct = h_struct_cpu.to(device) if device.type == 'cuda' else h_struct_cpu

    def safe_edge_feat(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        返回 (u->v) 或 (v->u) 的边文本特征（若存在多条，取第一条；若都不存在，用全零）。
        图保持在CPU上，只复制需要的边特征到GPU。
        """
        feat_dim = g.edata["feat"].shape[1] if "feat" in g.edata else 0
        if feat_dim == 0:
            return torch.zeros(u.shape[0], 0, device=device)
        # 将节点ID转换到CPU（因为图在CPU上）
        u_cpu = u.cpu() if u.is_cuda else u
        v_cpu = v.cpu() if v.is_cuda else v
        # 逐元素 edge_ids（成对）
        try:
            eids = g.edge_ids(u_cpu, v_cpu)
            if isinstance(eids, tuple):
                # 某些版本返回 (src_eids, dst_eids)
                eids = eids[0]
            # 从CPU图获取边特征，然后复制到GPU
            return g.edata["feat"][eids].to(device)
        except Exception:
            try:
                eids = g.edge_ids(v_cpu, u_cpu)
                if isinstance(eids, tuple):
                    eids = eids[0]
                return g.edata["feat"][eids].to(device)
            except Exception:
                return torch.zeros(u.shape[0], feat_dim, device=device)

    best_val_path_acc = 0.0
    best_epoch = 0
    best_val_metrics = None  # 保存最佳验证指标
    def iter_minibatches(batch, bs):
        N = batch["A"].shape[0]
        for s in range(0, N, bs):
            e = min(s + bs, N)
            yield {k: v[s:e] for k, v in batch.items()}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        # 正样本小批次
        for mb in iter_minibatches(train_batch_pos, batch_size):
            optim.zero_grad()
            a = mb["A"].to(device)
            b = mb["B"].to(device)
            c = mb["C"].to(device)
            event = mb["Event"].to(device)

            node_feats_ab = None
            node_feats_bc = None
            if h_struct is not None:
                node_feats_ab = (h_struct[a] + h_struct[b] + h_struct[event]) / 3.0  # [B, node_feat_dim]
                node_feats_bc = (h_struct[b] + h_struct[c] + h_struct[event]) / 3.0  # [B, node_feat_dim]

            text_ab = None
            text_bc = None
            if use_text_features and "feat" in g.edata:
                feat_dim = g.edata["feat"].shape[1]
                if feat_dim > 0:
                    feat_ae = safe_edge_feat(a, event)
                    feat_be = safe_edge_feat(b, event)
                    text_ab = (feat_ae + feat_be) / 2.0  # [B, text_dim]
                    feat_be2 = feat_be  # 复用
                    feat_ce = safe_edge_feat(c, event)
                    text_bc = (feat_be2 + feat_ce) / 2.0  # [B, text_dim]

            logits_ab, logits_bc, _ = model(
                a, b, c, event,
                training=True,
                node_features_ab=node_feats_ab,
                node_features_bc=node_feats_bc,
                text_features_ab=text_ab,
                text_features_bc=text_bc
            )
            loss_ab = F.cross_entropy(logits_ab, mb["y_ab"].to(device))
            loss_bc = F.cross_entropy(logits_bc, mb["y_bc"].to(device))
            loss = loss_ab + loss_bc

            # 负样本小批次（若有）
            if train_batch_neg is not None and train_batch_neg["A"].shape[0] > 0:
                # 采样相同大小的负样本片段
                for mbn in iter_minibatches(train_batch_neg, batch_size):
                    an = mbn["A"].to(device)
                    bn = mbn["B"].to(device)
                    cn = mbn["C"].to(device)
                    eventn = mbn["Event"].to(device)
                    node_feats_ab_n = None
                    node_feats_bc_n = None
                    if h_struct is not None:
                        node_feats_ab_n = (h_struct[an] + h_struct[bn] + h_struct[eventn]) / 3.0
                        node_feats_bc_n = (h_struct[bn] + h_struct[cn] + h_struct[eventn]) / 3.0
                    text_ab_n = None
                    text_bc_n = None
                    if use_text_features and "feat" in g.edata and g.edata["feat"].shape[1] > 0:
                        feat_ae_n = safe_edge_feat(an, eventn)
                        feat_be_n = safe_edge_feat(bn, eventn)
                        text_ab_n = (feat_ae_n + feat_be_n) / 2.0
                        feat_ce_n = safe_edge_feat(cn, eventn)
                        text_bc_n = (feat_be_n + feat_ce_n) / 2.0

                    logits_abn, logits_bcn, _ = model(
                        an, bn, cn, eventn,
                        training=True,
                        node_features_ab=node_feats_ab_n,
                        node_features_bc=node_feats_bc_n,
                        text_features_ab=text_ab_n,
                        text_features_bc=text_bc_n
                    )
                    loss_abn = F.cross_entropy(logits_abn, mbn["y_ab"].to(device))
                    loss_bcn = F.cross_entropy(logits_bcn, mbn["y_bc"].to(device))
                    loss = loss + 0.5 * (loss_abn + loss_bcn)
                    break  # 每个正样本批次配一个负样本批次，避免内存暴涨

            loss.backward()
            optim.step()
            total_loss += float(loss.item())

        if (epoch+1)%val_every==0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
            model.eval()
            val_metrics = evaluate_path_metrics(model, val_pos, name_to_id, id_to_name, device)
            print(f"  [VAL] Path Acc: {val_metrics['path_acc']:.4f}, Path F1: {val_metrics['path_f1']:.4f}")
            print(f"  [VAL] A-B  Acc: {val_metrics['ab_acc']:.4f}, F1: {val_metrics['ab_f1']:.4f}, AUC: {val_metrics['ab_auc_roc']:.4f}, MCC: {val_metrics['ab_mcc']:.4f}")
            print(f"  [VAL] B-C  Acc: {val_metrics['bc_acc']:.4f}, F1: {val_metrics['bc_f1']:.4f}, AUC: {val_metrics['bc_auc_roc']:.4f}, MCC: {val_metrics['bc_mcc']:.4f}")
            if val_metrics['path_acc'] > best_val_path_acc:
                best_val_path_acc = val_metrics['path_acc']
                best_epoch = epoch + 1
                best_val_metrics = val_metrics.copy()  # 保存最佳验证指标
                print(f"  ✓ New best Path Acc: {best_val_path_acc:.4f}")
    print("[8/8] Final Evaluation on Test Set")
    test_metrics = evaluate_path_metrics(model, test_pos, name_to_id, id_to_name, device)
    print(f"[TEST] Path Acc: {test_metrics['path_acc']:.4f}")
    print(f"[TEST] Path F1 : {test_metrics['path_f1']:.4f}")
    print(f"[TEST] A-B Comp: Acc {test_metrics['ab_acc']:.4f}, F1 {test_metrics['ab_f1']:.4f}, AUC {test_metrics['ab_auc_roc']:.4f}, MCC {test_metrics['ab_mcc']:.4f}")
    print(f"[TEST] B-C Comp: Acc {test_metrics['bc_acc']:.4f}, F1 {test_metrics['bc_f1']:.4f}, AUC {test_metrics['bc_auc_roc']:.4f}, MCC {test_metrics['bc_mcc']:.4f}")
    if best_val_metrics is None:
        best_val_metrics = evaluate_path_metrics(model, val_pos, name_to_id, id_to_name, device)
        best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
        best_epoch = epochs

    print(f"Best val Path Acc: {best_val_path_acc:.4f} (epoch {best_epoch})")
    return {
        "model": model,
        "graph": g,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "test_metrics": test_metrics,
        "val_metrics": best_val_metrics,
        "best_val_metrics": best_val_metrics,
        "best_val_path_acc": best_val_path_acc,
        "best_epoch": best_epoch
    }

def evaluate_path_metrics(model, samples, name_to_id, id_to_name, device: Optional[Union[str, torch.device]] = None):
    device = resolve_device(device)
    if len(samples) == 0:
        return {"path_acc":0.0, "path_f1":0.0, "ab_acc":0.0, "ab_f1":0.0, "ab_auc_roc":0.0, "ab_mcc":0.0, "bc_acc":0.0, "bc_f1":0.0, "bc_auc_roc":0.0, "bc_mcc":0.0}
    all_true_ab, all_pred_ab, all_scores_ab = [], [], []
    all_true_bc, all_pred_bc, all_scores_bc = [], [], []
    y_true_path_4class, y_pred_path_4class = [], []
    model.eval()
    with torch.no_grad():
        for i, p in enumerate(samples):
            A, B, C = p["A"], p["B"], p["C"]
            Event = p.get("Event", -1)
            true_ab = name_to_id[p["rel_AB"]]
            true_bc = name_to_id[p["rel_BC"]]
            aa = torch.tensor([A], device=device)
            bb = torch.tensor([B], device=device)
            cc = torch.tensor([C], device=device)
            ee = torch.tensor([Event], device=device)

            # 推理阶段同样构建结构与文本特征
            node_feats_ab = None
            node_feats_bc = None
            text_ab = None
            text_bc = None
            # 仅当 train_pipeline_from_graph 已计算了 g 和 h_struct 时，这里才有上下文；
            # 简化处理：若不可用，则走纯 RAM 推理。
            # 由于此函数独立于上面的闭包，此处无法直接访问 h_struct 与 g；
            # 为保持最小侵入，这里不聚合，仍使用结构空特征与文本空特征。

            logits_ab, logits_bc, rel_ab_pred = model(
                aa, bb, cc, ee, training=False,
                node_features_ab=node_feats_ab,
                node_features_bc=node_feats_bc,
                text_features_ab=text_ab,
                text_features_bc=text_bc
            )
            scores_ab = F.softmax(logits_ab, dim=-1).cpu().numpy()[0]
            pred_ab = int(torch.argmax(logits_ab, dim=-1).item())
            scores_bc = F.softmax(logits_bc, dim=-1).cpu().numpy()[0]
            pred_bc = int(torch.argmax(logits_bc, dim=-1).item())
            all_true_ab.append(true_ab)
            all_pred_ab.append(pred_ab)
            all_scores_ab.append(scores_ab[1] if len(scores_ab) > 1 else scores_ab[0])
            all_true_bc.append(true_bc)
            all_pred_bc.append(pred_bc)
            all_scores_bc.append(scores_bc[1] if len(scores_bc) > 1 else scores_bc[0])
            y_true_path_4class.append(0)
            is_ab_corr,pred_bc_corr = (pred_ab==true_ab),(pred_bc==true_bc)
            if is_ab_corr and pred_bc_corr:
                y_pred_path_4class.append(0) #两步都预测正确
            elif is_ab_corr and not pred_bc_corr:
                y_pred_path_4class.append(1)  #只有AB预测正确
            elif (not is_ab_corr) and pred_bc_corr:
                y_pred_path_4class.append(2)  #只有BC预测正确
            else:
                y_pred_path_4class.append(3)  #全都预测错误
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
    # 由于有四类，f1=recall召回率
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
