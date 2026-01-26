from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

from .models import RGCN, RelationEmbeddings, PredictorAB, PredictorBC, TextProjector
from ..graph_build import build_dgl_graph
from ..path_data import enumerate_graph_paths, split_paths, generate_negatives, select_few_shot
from ..db_mongo import fetch_pair_evidence_text, get_pair_evidence_embedding
from ..config import NODE_NAME_FIELD, NEIGHBOR_AGG_METHOD, LABEL_SLC, SBERT_MODEL_NAME, SBERT_DIM
from ..device_utils import resolve_device
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    roc_auc_score,        
    matthews_corrcoef,     
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from scipy.stats import rankdata


def build_label_maps(relation_type_map):
    # relation name -> class id
    name_to_id = {k: v for k, v in relation_type_map.items()}
    id_to_name = {v: k for k, v in name_to_id.items()}
    return name_to_id, id_to_name


def build_relation_type_map_from_paths(paths: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build relation type maps from the enumerated paths.
    In the N-ary model, rel_AB and rel_BC are properties of RelaEvent nodes,
    so we extract them from paths rather than graph edge types.
    
    Returns: (name_to_id, id_to_name) where name is the relation string (e.g., "PROMOTION", "SUPPRESSION")
    """
    all_rel_types = set()
    for p in paths:
        all_rel_types.add(p["rel_AB"])
        all_rel_types.add(p["rel_BC"])
    
    rel_types_sorted = sorted(list(all_rel_types))
    name_to_id = {name: i for i, name in enumerate(rel_types_sorted)}
    id_to_name = {i: name for i, name in enumerate(rel_types_sorted)}
    return name_to_id, id_to_name


def prepare_batches(paths: List[Dict], name_to_id) -> Dict[str, torch.Tensor]:
    A = torch.tensor([p["A"] for p in paths], dtype=torch.long)
    B = torch.tensor([p["B"] for p in paths], dtype=torch.long)
    C = torch.tensor([p["C"] for p in paths], dtype=torch.long)
    y_ab = torch.tensor([name_to_id[p["rel_AB"]] for p in paths], dtype=torch.long)
    y_bc = torch.tensor([name_to_id[p["rel_BC"]] for p in paths], dtype=torch.long)
    return {"A": A, "B": B, "C": C, "y_ab": y_ab, "y_bc": y_bc}


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move every tensor in a batch dict to the specified device."""
    return {key: tensor.to(device) for key, tensor in batch.items()} if batch is not None else batch


def aggregate_slc_neighbors(g, node_ids: torch.Tensor, agg_method: str = "mean") -> torch.Tensor:
    """Aggregate embeddings of SLC node neighbors.
    For each SLC node in node_ids, gather all its outgoing neighbors' embeddings and aggregate.
    agg_method: "mean", "sum", or "max"
    Returns: [len(node_ids), feat_dim] tensor of aggregated neighbor embeddings.
    注意：图保持在CPU上，节点特征也在CPU上，但返回的聚合结果会复制到node_ids的设备上。
    """
    node_h = g.ndata["feat"]  # [num_nodes, feat_dim] - 图在CPU上，所以node_h也在CPU上
    target_device = node_ids.device  # 目标设备（可能是GPU）
    feat_dim = node_h.shape[1]
    agg_list = []

    for slc_id in node_ids:
        slc_id_item = int(slc_id.item())  # 转换为Python int
        neighbors = g.successors(slc_id_item)  # 图在CPU上，所以neighbors也在CPU上
        if torch.is_tensor(neighbors):
            # neighbors在CPU上，保持CPU
            neighbors_cpu = neighbors
        else:
            neighbors_cpu = torch.tensor(neighbors, device=torch.device("cpu"), dtype=torch.long)

        if neighbors_cpu.numel() == 0:
            # 创建零向量在CPU上
            agg_list.append(torch.zeros(feat_dim, dtype=node_h.dtype, device=torch.device("cpu")))
            continue

        neighbor_feats = node_h[neighbors_cpu.long()]  # [num_neighbors, feat_dim] - 在CPU上
        if agg_method == "mean":
            agg_feat = neighbor_feats.mean(dim=0)
        elif agg_method == "sum":
            agg_feat = neighbor_feats.sum(dim=0)
        elif agg_method == "max":
            agg_feat = neighbor_feats.max(dim=0)[0]
        else:
            agg_feat = neighbor_feats.mean(dim=0)
        agg_list.append(agg_feat)  # agg_feat在CPU上

    if not agg_list:
        return torch.zeros(0, feat_dim, dtype=node_h.dtype, device=target_device)

    result = torch.stack(agg_list, dim=0)  # 在CPU上
    # 将结果复制到目标设备（可能是GPU）
    return result.to(target_device)


def split_path_dataset(paths: List[Dict], train_ratio=0.7, val_ratio=0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split paths into train/val/test (0.7/0.15/0.15)"""
    idx = np.arange(len(paths))
    np.random.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    train = [paths[i] for i in train_idx]
    val = [paths[i] for i in val_idx]
    test = [paths[i] for i in test_idx]
    print(f"[Data Split] Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def load_text_features_for_pairs(
    paths: List[Dict], 
    name_to_id: Dict[str, int],
    embed_dim: int = SBERT_DIM
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    预先加载并嵌入文本特征（从MongoDB）用于A-B和B-C实体对。
    优化：一次性加载所有文本特征，避免在训练过程中实时查询。
    
    Args:
        paths: 路径列表，每个路径包含 A_name, B_name, C_name 等字段
        name_to_id: 关系类型映射（未使用，保持接口一致性）
        embed_dim: 文本嵌入维度，默认 SBERT_DIM（768）
    
    Returns:
        text_features_ab: Dict[int, torch.Tensor] - 索引到A-B文本特征的映射
        text_features_bc: Dict[int, torch.Tensor] - 索引到B-C文本特征的映射
    """
    text_features_ab: Dict[int, torch.Tensor] = {}
    text_features_bc: Dict[int, torch.Tensor] = {}

    try:
        from sentence_transformers import SentenceTransformer
        print(f"  [Text Features] 加载 SentenceTransformer 模型: {SBERT_MODEL_NAME}")
        encoder = SentenceTransformer(SBERT_MODEL_NAME)
    except Exception as e:
        print(f"  ⚠️  Warning: 无法加载 SentenceTransformer: {e}")
        encoder = None  # 回退：使用缓存或返回零

    print(f"  [Text Features] 开始加载 {len(paths)} 个路径的文本特征...")
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
            if idx % 100 == 0:  # 每100个样本打印一次警告，避免日志过多
                print(f"  Warning: Failed to fetch text for path {idx}: {e}")
            if idx not in text_features_ab:
                text_features_ab[idx] = torch.zeros(embed_dim, dtype=torch.float32)
            if idx not in text_features_bc:
                text_features_bc[idx] = torch.zeros(embed_dim, dtype=torch.float32)
            continue

    print(f"  ✓ Loaded text features for {len(text_features_ab)} A-B pairs")
    print(f"  ✓ Loaded text features for {len(text_features_bc)} B-C pairs")

    return text_features_ab, text_features_bc


def train_pipeline_from_graph(
    sbert_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    dropout_rate: float,
    weight_decay: float,
    use_slc_neighbors: bool = True,
    device=None,
    *,
    train_paths: Optional[List[Dict]] = None,
    val_paths: Optional[List[Dict]] = None,
    test_paths: Optional[List[Dict]] = None,
    neighbor_agg_method: str = NEIGHBOR_AGG_METHOD,
    skip_negatives: bool = False,
    few_shot_k: Optional[int] = None,
    few_shot_balance: Optional[str] = None,
    seed: int = 0,
):
    """
    sbert_dim: 文本嵌入维度
    epochs: 训练轮数
    lr: 学习率
    batch_size: 批次大小（暂未分批，但作为下游接口预留）
    hidden_dim: RGCN 第一层输出和MLP隐藏层
    dropout_rate: Dropout强度
    weight_decay: Adam正则化
    use_slc_neighbors: 是否用SLC邻居
    """
    device = resolve_device(device)
    rgcn_device = torch.device("cpu")

    g, relation_type_map = build_dgl_graph()
    # 图保持在CPU上，避免DGL CUDA依赖问题
    print(f"[Device] Using {device} for RGCN pipeline")
    print(f"[Device] Graph will stay on CPU (to avoid DGL CUDA dependency)")
    if device.type != "cpu":
        print(f"[Device] RGCN layers will run on CPU to match graph device; outputs will be moved to {device}.")
    # 图保持在CPU上，只在需要时复制数据到GPU
    # 不修改图的ndata和edata，保持图在CPU上
    
    if train_paths is None and val_paths is None and test_paths is None:
        # Enumerate positives from graph first
        positives = enumerate_graph_paths()
        print(f"[Graph] Enumerated {len(positives)} total paths")
        if len(positives) == 0:
            print("[Graph] ✗ No paths found!")
            return None

        all_paths = positives

        # Split using 0.7/0.15/0.15
        train_pos, val_pos, test_pos = split_path_dataset(positives, train_ratio=0.7, val_ratio=0.15)
        if few_shot_k:
            train_pos = select_few_shot(train_pos, few_shot_k, seed=seed, balance_by=few_shot_balance)
    else:
        if train_paths is None or val_paths is None:
            raise ValueError("train_paths and val_paths must be provided when using external splits")

        print("[Graph] Using externally provided train/val/test splits...")
        train_pos = list(train_paths)
        val_pos = list(val_paths)
        test_pos = list(test_paths) if test_paths is not None else list(val_pos)
        print(f"  ✓ Provided splits - Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}")

        combined_paths = train_pos + val_pos + [p for p in test_pos if p not in val_pos]
        if len(combined_paths) == 0:
            print("  ✗ No paths supplied by caller!")
            return None
        all_paths = combined_paths
        if few_shot_k:
            train_pos = select_few_shot(train_pos, few_shot_k, seed=seed, balance_by=few_shot_balance)

    # Build relation type map from path properties (rel_AB, rel_BC) instead of graph edges
    name_to_id, id_to_name = build_relation_type_map_from_paths(all_paths)
    num_relations = len(name_to_id)
    print(f"[Relations] Found {num_relations} unique relation types: {sorted(name_to_id.keys())}")
    if num_relations != 2:
        print(f"警告: 关系数量为 {num_relations}。AUC-ROC 和 MCC 最适用于二元分类。")

    # Prepare a set of all positives for filtering during negative sampling
    pos_set = {(p["A"], p["B"], p["C"], p["rel_AB"], p["rel_BC"]) for p in all_paths}

    if skip_negatives:
        print("[Training] Skipping negative sampling (skip_negatives=True)...")
        train_neg = []
    else:
        # Generate negatives for train with ratio 1.0 (1 negative per positive)
        train_neg = generate_negatives(train_pos, pos_set, neg_sample_ratio=1.0)
        print(f"[Training] Generated {len(train_neg)} negative samples from {len(train_pos)} positive samples")

    # 预加载文本特征（优化：避免在训练过程中实时查询）
    print("\n[Text Features] 预加载文本特征...")
    text_features_ab_train, text_features_bc_train = load_text_features_for_pairs(train_pos, name_to_id, embed_dim=sbert_dim)
    text_features_ab_val, text_features_bc_val = load_text_features_for_pairs(val_pos, name_to_id, embed_dim=sbert_dim)
    text_features_ab_test, text_features_bc_test = load_text_features_for_pairs(test_pos, name_to_id, embed_dim=sbert_dim)
    # 负样本通常没有文本特征，使用零向量
    text_features_ab_neg = {idx: torch.zeros(sbert_dim, dtype=torch.float32) for idx in range(len(train_neg))}
    text_features_bc_neg = {idx: torch.zeros(sbert_dim, dtype=torch.float32) for idx in range(len(train_neg))}

    # Models
    # Note: num_relations now represents unique values in rel_AB and rel_BC
    rgcn = RGCN(sbert_dim, hidden_dim, 128, num_rels=len(relation_type_map)).to(rgcn_device)
    rgcn_compute_device = next(rgcn.parameters()).device
    rel_emb = RelationEmbeddings(num_relations=num_relations, dim=128).to(device)
    head_ab = PredictorAB(128, num_relations=num_relations, use_slc_neighbors=use_slc_neighbors).to(device)
    head_bc = PredictorBC(128, num_relations=num_relations).to(device)
    text_proj = TextProjector(input_dim=sbert_dim, out_dim=128).to(device)

    # expose sbert_dim to closures
    global sbert_dim_global
    sbert_dim_global = sbert_dim

    optim = torch.optim.Adam(
        list(rgcn.parameters())
        + list(rel_emb.parameters())
        + list(head_ab.parameters())
        + list(head_bc.parameters())
        + list(text_proj.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    # 优化：使用预加载的文本特征，而不是实时查询

    def get_rel_ab_embedding(logits_ab, rel_emb, training):
        """
        根据训练/评估状态，使用 Gumbel-Softmax (训练) 或 argmax (评估) 从logits_ab取出one-hot, 并乘以relation embedding 矩阵获得h_rel_ab
        """
        if training:
            soft_pred_ab_onehot = F.gumbel_softmax(logits_ab, tau=1.0, hard=True)
            h_rel_ab = soft_pred_ab_onehot @ rel_emb.emb.weight  # (B, num_rel) @ (num_rel, dim)
        else:
            pred_ab = torch.argmax(logits_ab, dim=-1)
            h_rel_ab = rel_emb.emb(pred_ab)
        return h_rel_ab

    def forward_paths(batch: Dict[str, torch.Tensor], raw_batch_list, text_features_ab_dict, text_features_bc_dict, training=True):
        """
        Forward pass for cascaded path prediction in N-ary model.
        优化版本：使用预加载的文本特征，避免实时查询。
        
        Process:
        1. Get node embeddings from RGCN
        2. 使用预加载的文本特征（A-B和B-C）
        3. Predict rel_AB using (A, B) nodes + their SLC-Pathway text evidence
        4. Predict rel_BC using (B, C) nodes + their Pathway-Disease text evidence
           (cascaded: uses predicted rel_AB embedding, not ground truth)
        
        Args:
            batch: 批次数据字典
            raw_batch_list: 原始路径列表（用于索引文本特征）
            text_features_ab_dict: 预加载的A-B文本特征字典 {idx: tensor}
            text_features_bc_dict: 预加载的B-C文本特征字典 {idx: tensor}
            training: 是否训练模式
        """
        # Step 1: Get base node embeddings from RGCN
        feat_cpu = g.ndata["feat"].to(rgcn_compute_device)
        rel_types_cpu = g.edata["rel_type"].to(rgcn_compute_device)
        node_h = rgcn(g, feat_cpu, rel_types_cpu).to(device)  # [N, 128]
        hA = node_h[batch["A"]]    # SLCGene embeddings
        hB = node_h[batch["B"]]    # Pathway embeddings
        hC = node_h[batch["C"]]    # Disease embeddings
        
        # Step 2: 使用预加载的文本特征（优化：避免实时查询）
        batch_size = len(raw_batch_list)
        e_text_ab_list = []
        e_text_bc_list = []
        for idx in range(batch_size):
            # 从预加载的字典中获取文本特征
            e_text_ab_list.append(text_features_ab_dict.get(idx, torch.zeros(sbert_dim_global, dtype=torch.float32)))
            e_text_bc_list.append(text_features_bc_dict.get(idx, torch.zeros(sbert_dim_global, dtype=torch.float32)))
        
        # 堆叠成批次张量并投影
        e_text_ab = text_proj(torch.stack(e_text_ab_list).to(node_h.device))  # [B, 128]
        e_text_bc = text_proj(torch.stack(e_text_bc_list).to(node_h.device))  # [B, 128]
        
        # Step 3: Augment node embeddings with corresponding text evidence
        # For A-B prediction: Use SLC-Pathway evidence
        hA_aug = hA + e_text_ab   # A node + its SLC-Pathway evidence
        hB_aug = hB + e_text_ab   # B node + its SLC-Pathway evidence
        
        # Step 4: Predict rel_AB relationship
        if use_slc_neighbors:
            hA_neighbors_agg = aggregate_slc_neighbors(g, batch["A"], agg_method=NEIGHBOR_AGG_METHOD)
            logits_ab = head_ab(hA_aug, hB_aug, h_a_neighbors_agg=hA_neighbors_agg)
        else:
            logits_ab = head_ab(hA_aug, hB_aug)
        
        # Step 5: Predict rel_BC relationship (cascaded: uses predicted rel_AB)
        # Get the predicted rel_AB as embedding (this makes it cascaded)
        h_rel_ab = get_rel_ab_embedding(logits_ab, rel_emb, training)
        
        # For B-C prediction: Use Pathway-Disease evidence
        hB_aug2 = hB + e_text_bc   # B node + its Pathway-Disease evidence
        hC_aug = hC + e_text_bc    # C node + its Pathway-Disease evidence
        
        logits_bc = head_bc(hB_aug2, hC_aug, h_rel_ab)
        return logits_ab, logits_bc

    # Training
    train_batch_pos = move_batch_to_device(prepare_batches(train_pos, name_to_id), device)
    train_batch_neg = (
        move_batch_to_device(prepare_batches(train_neg, name_to_id), device)
        if len(train_neg) > 0
        else None
    )

    best_val_path_acc = 0.0
    best_epoch = 0
    best_val_metrics = None
    last_val_metrics = None

    for epoch in range(epochs):
        rgcn.train(); rel_emb.train(); head_ab.train(); head_bc.train()
        optim.zero_grad()
        logits_ab_pos, logits_bc_pos = forward_paths(
            train_batch_pos, train_pos, 
            text_features_ab_train, text_features_bc_train, 
            training=True
        )
        loss_pos = F.cross_entropy(logits_ab_pos, train_batch_pos["y_ab"]) + \
                   F.cross_entropy(logits_bc_pos, train_batch_pos["y_bc"])
        if train_batch_neg is not None and len(train_neg) > 0:
            logits_ab_neg, logits_bc_neg = forward_paths(
                train_batch_neg, train_neg,
                text_features_ab_neg, text_features_bc_neg,
                training=True
            )
            # Encourage wrong combinations to have low prob on their chosen labels
            loss_neg = F.cross_entropy(logits_ab_neg, train_batch_neg["y_ab"]) + \
                       F.cross_entropy(logits_bc_neg, train_batch_neg["y_bc"]) 
            # Flip sign by maximizing loss on negatives -> minimize negative log-prob of true wrong label
            loss = loss_pos + 0.5 * (-loss_neg)
        else:
            loss = loss_pos
        loss.backward(); optim.step()

        if epoch % 10 == 0:
            # --- 调用新的评估函数 ---
            print(f"Epoch {epoch}: loss={loss.item():.4f}")
            val_metrics = evaluate_path_metrics(
                g, rgcn, rel_emb, head_ab, head_bc, text_proj,
                val_pos, name_to_id, 
                text_features_ab_val, text_features_bc_val,
                use_slc_neighbors=use_slc_neighbors,
                rgcn_device=rgcn_compute_device,
                target_device=device
            )
            last_val_metrics = val_metrics
            # --- 打印新指标 ---
            print(f"  [Validation Metrics]")
            print(f"  Path (A-B-C): Path_Acc={val_metrics['path_acc']:.4f}, Path_F1={val_metrics['path_f1']:.4f}")
            print(f"  Comp (A-B):   Acc={val_metrics['ab_acc']:.4f}, F1-Macro={val_metrics['ab_f1']:.4f}, AUC-ROC={val_metrics['ab_auc_roc']:.4f}, MCC={val_metrics['ab_mcc']:.4f}")
            print(f"  Comp (B-C):   Acc={val_metrics['bc_acc']:.4f}, F1-Macro={val_metrics['bc_f1']:.4f}, AUC-ROC={val_metrics['bc_auc_roc']:.4f}, MCC={val_metrics['bc_mcc']:.4f}")
            if val_metrics["path_acc"] > best_val_path_acc:
                best_val_path_acc = val_metrics["path_acc"]
                best_val_metrics = val_metrics.copy()
                best_epoch = epoch
                print(f"  ✓ New best Path Acc: {best_val_path_acc:.4f}")

    if best_val_metrics is None:
        if last_val_metrics is None:
            best_val_metrics = evaluate_path_metrics(
                g, rgcn, rel_emb, head_ab, head_bc, text_proj,
                val_pos, name_to_id,
                text_features_ab_val, text_features_bc_val,
                use_slc_neighbors=use_slc_neighbors,
                rgcn_device=rgcn_compute_device,
                target_device=device
            )
        else:
            best_val_metrics = last_val_metrics
        best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
        best_epoch = epochs

    # --- 最终测试评估 ---
    print("\n--- 训练完成，开始最终测试 ---")
    test_metrics = evaluate_path_metrics(
        g, rgcn, rel_emb, head_ab, head_bc, text_proj,
        test_pos, name_to_id,
        text_features_ab_test, text_features_bc_test,
        use_slc_neighbors=use_slc_neighbors,
        rgcn_device=rgcn_compute_device,
        target_device=device
    )
    print(f"--- 最终测试结果 ---")
    print(f"  Path (A-B-C): Path_Acc={test_metrics['path_acc']:.4f}, Path_F1={test_metrics['path_f1']:.4f}")
    print(f"  Comp (A-B):   Acc={test_metrics['ab_acc']:.4f}, F1-Macro={test_metrics['ab_f1']:.4f}, AUC-ROC={test_metrics['ab_auc_roc']:.4f}, MCC={test_metrics['ab_mcc']:.4f}")
    print(f"  Comp (B-C):   Acc={test_metrics['bc_acc']:.4f}, F1-Macro={test_metrics['bc_f1']:.4f}, AUC-ROC={test_metrics['bc_auc_roc']:.4f}, MCC={test_metrics['bc_mcc']:.4f}")
    print(f"----------------------")
    
    # ✅ 返回字典格式，包含模型和指标
    return {
        "rgcn": rgcn,
        "rel_emb": rel_emb,
        "head_ab": head_ab,
        "head_bc": head_bc,
        "text_proj": text_proj,
        "val_metrics": best_val_metrics,
        "best_val_metrics": best_val_metrics,
        "best_val_path_acc": best_val_path_acc,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics
    }

def evaluate_path_metrics(
    g, rgcn, rel_emb, head_ab, head_bc, text_proj,
    samples: List[Dict], 
    name_to_id,
    text_features_ab_dict: Dict[int, torch.Tensor],
    text_features_bc_dict: Dict[int, torch.Tensor],
    use_slc_neighbors: bool,
    rgcn_device: Optional[torch.device] = None,
    target_device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    评估级联路径预测的多个指标：
    - 组件 (A-B): Acc, F1-Macro, AUC-ROC, MCC
    - 组件 (B-C): Acc, F1-Macro, AUC-ROC, MCC
    - 路径 (A-B-C): Path Accuracy, Path F1-Score (for class "Correct-Correct")
    
    使用级联逻辑: B-C 的预测依赖于 A-B 的 *预测* 结果。
    优化版本：使用预加载的文本特征，避免实时查询。
    """
    if len(samples) == 0:
        return {
            "path_acc": 0.0, "path_f1": 0.0,
            "ab_acc": 0.0, "ab_f1": 0.0, "ab_auc_roc": 0.0, "ab_mcc": 0.0,
            "bc_acc": 0.0, "bc_f1": 0.0, "bc_auc_roc": 0.0, "bc_mcc": 0.0
        }
        
    rgcn.eval(); rel_emb.eval(); head_ab.eval(); head_bc.eval()
    
    if rgcn_device is None:
        rgcn_device = next(rgcn.parameters()).device

    if target_device is None:
        target_device = next(head_ab.parameters()).device

    feat_cpu = g.ndata["feat"].to(rgcn_device)
    rel_types_cpu = g.edata["rel_type"].to(rgcn_device)
    node_h = rgcn(g, feat_cpu, rel_types_cpu).to(target_device)
    device = node_h.device
    sbert_dim = text_proj.input_dim if hasattr(text_proj, 'input_dim') else SBERT_DIM
    
    # 收集列表用于计算指标
    all_true_ab, all_pred_ab, all_scores_ab_auc = [], [], []
    all_true_bc, all_pred_bc, all_scores_bc_auc = [], [], []
    
    # 用于 Path F1-Score (4分类)
    # 0=(T,T), 1=(T,F), 2=(F,T), 3=(F,F)
    y_true_path_4class = []
    y_pred_path_4class = []
    
    with torch.no_grad():
        for idx, p in enumerate(samples):
            A, B, C = p["A"], p["B"], p["C"]
            true_ab = name_to_id[p["rel_AB"]]
            true_bc = name_to_id[p["rel_BC"]]
            hA, hB, hC = node_h[A].unsqueeze(0), node_h[B].unsqueeze(0), node_h[C].unsqueeze(0)
            
            # 使用预加载的文本特征
            e_text_ab = text_features_ab_dict.get(idx, torch.zeros(sbert_dim, dtype=torch.float32)).to(device)
            e_text_bc = text_features_bc_dict.get(idx, torch.zeros(sbert_dim, dtype=torch.float32)).to(device)
            e_text_ab = text_proj(e_text_ab.unsqueeze(0))  # [1, 128]
            e_text_bc = text_proj(e_text_bc.unsqueeze(0))  # [1, 128]
            
            # 增强节点嵌入
            hA_aug = hA + e_text_ab
            hB_aug = hB + e_text_ab
            
            # --- STEP 1: 评估 A-B 预测 ---
            if use_slc_neighbors:
                hA_neighbors_agg = aggregate_slc_neighbors(
                    g,
                    torch.tensor([A], dtype=torch.long, device=device),
                    agg_method=NEIGHBOR_AGG_METHOD,
                )
                logits_ab = head_ab(hA_aug, hB_aug, h_a_neighbors_agg=hA_neighbors_agg)
            else:
                logits_ab = head_ab(hA_aug, hB_aug)
            
            # 假设 num_relations=2 (标签为 0 和 1)
            # F.softmax 返回 [prob_class_0, prob_class_1]
            scores_ab_softmax = F.softmax(logits_ab, dim=-1).cpu().numpy()[0]
            pred_ab = int(torch.argmax(logits_ab, dim=-1).item())
            
            all_true_ab.append(true_ab)
            all_pred_ab.append(pred_ab)
            # 为 AUC-ROC 收集 "positive" class (class 1) 的概率
            all_scores_ab_auc.append(scores_ab_softmax[1])
            
            # --- STEP 2: 评估 B-C 预测 (使用 *预测* 的 e_ab) ---
            e_pred_ab = rel_emb(torch.tensor([pred_ab]).to(device)) # 关键：使用 pred_ab
            hB_aug2 = hB + e_text_bc
            hC_aug = hC + e_text_bc
            logits_bc = head_bc(hB_aug2, hC_aug, e_pred_ab)
            
            scores_bc_softmax = F.softmax(logits_bc, dim=-1).cpu().numpy()[0]
            pred_bc = int(torch.argmax(logits_bc, dim=-1).item())

            all_true_bc.append(true_bc)
            all_pred_bc.append(pred_bc)
            all_scores_bc_auc.append(scores_bc_softmax[1])

            # --- STEP 3: 评估 Path 4分类指标 ---
            # 评估集只包含正样本, 所以 y_true 总是 Class 0
            y_true_path_4class.append(0) 
            
            is_ab_correct = (pred_ab == true_ab)
            is_bc_correct = (pred_bc == true_bc)
            
            if is_ab_correct and is_bc_correct:
                y_pred_path_4class.append(0) # (T, T)
            elif is_ab_correct and not is_bc_correct:
                y_pred_path_4class.append(1) # (T, F)
            elif not is_ab_correct and is_bc_correct:
                y_pred_path_4class.append(2) # (F, T)
            else:
                y_pred_path_4class.append(3) # (F, F)

    # --- 计算所有聚合指标 ---
    
    # (A-B) 组件指标
    acc_ab = accuracy_score(all_true_ab, all_pred_ab)
    f1_ab = f1_score(all_true_ab, all_pred_ab, average='macro', zero_division=0)
    mcc_ab = matthews_corrcoef(all_true_ab, all_pred_ab)
    auc_roc_ab = 0.0
    try:
        # AUC-ROC 需要至少两个类别
        if len(np.unique(all_true_ab)) > 1:
            auc_roc_ab = roc_auc_score(all_true_ab, all_scores_ab_auc)
        else:
            auc_roc_ab = acc_ab # 如果只有一个类别，AUC未定义，用Acc代替
    except ValueError:
        pass # 保持 0.0
        
    # (B-C) 组件指标
    acc_bc = accuracy_score(all_true_bc, all_pred_bc)
    f1_bc = f1_score(all_true_bc, all_pred_bc, average='macro', zero_division=0)
    mcc_bc = matthews_corrcoef(all_true_bc, all_pred_bc)
    auc_roc_bc = 0.0
    try:
        if len(np.unique(all_true_bc)) > 1:
            auc_roc_bc = roc_auc_score(all_true_bc, all_scores_bc_auc)
        else:
            auc_roc_bc = acc_bc
    except ValueError:
        pass

    # (A-B-C) 路径指标
    # Path Accuracy (4分类的整体准确率)
    path_acc = accuracy_score(y_true_path_4class, y_pred_path_4class)
    
    # Path F1-Score (只看 Class 0, 即 "Correct-Correct")
    # labels=[0] 指定我们只关心 true-true
    # average='macro' 在这里只计算 Class 0 的F1
    path_f1 = f1_score(y_true_path_4class, y_pred_path_4class, labels=[0], average='macro', zero_division=0)

    id_to_name_local = {v: k for k, v in name_to_id.items()} if isinstance(name_to_id, dict) else {}

    labels_ab = sorted(set(all_true_ab) | set(all_pred_ab))
    ab_cm = confusion_matrix(all_true_ab, all_pred_ab, labels=labels_ab).tolist()
    ab_prec, ab_rec, ab_f1_cls, ab_sup = precision_recall_fscore_support(
        all_true_ab,
        all_pred_ab,
        labels=labels_ab,
        zero_division=0,
    )
    ab_bal_acc = balanced_accuracy_score(all_true_ab, all_pred_ab) if len(labels_ab) > 1 else acc_ab
    ab_label_names = [str(id_to_name_local.get(int(lbl), str(lbl))) for lbl in labels_ab]

    labels_bc = sorted(set(all_true_bc) | set(all_pred_bc))
    bc_cm = confusion_matrix(all_true_bc, all_pred_bc, labels=labels_bc).tolist()
    bc_prec, bc_rec, bc_f1_cls, bc_sup = precision_recall_fscore_support(
        all_true_bc,
        all_pred_bc,
        labels=labels_bc,
        zero_division=0,
    )
    bc_bal_acc = balanced_accuracy_score(all_true_bc, all_pred_bc) if len(labels_bc) > 1 else acc_bc
    bc_label_names = [str(id_to_name_local.get(int(lbl), str(lbl))) for lbl in labels_bc]

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

