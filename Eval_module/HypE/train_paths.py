"""
Training pipeline for HypE model on N-ary Knowledge Graphs.

This module adapts the HypE architecture for the specific task:
- Predicting rel_AB (SLCGene -> Pathway relationships)
- Predicting rel_BC (Pathway -> Disease relationships)
- Using RelaEvent nodes as central N-ary statement nodes
- CASCADING: First predict rel_AB, then predict rel_BC based on predicted rel_AB
- NODE FEATURES: Support encoding of node attributes (SLC neighbors, etc.)
- TEXT FEATURES: Support MongoDB text evidence integration
"""

from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np

from .models import HypEBackbone, CascadingHypEPredictor, HypETextProjector
from ..graph_build import build_dgl_graph
from ..path_data import enumerate_graph_paths, split_paths, generate_negatives, select_few_shot
from ..db_mongo import fetch_pair_evidence_text, get_pair_evidence_embedding
from ..config import NODE_NAME_FIELD, NEIGHBOR_AGG_METHOD, LABEL_SLC, SBERT_DIM
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


def load_text_features_for_pairs(
    paths: List[Dict], 
    name_to_id: Dict[str, int],
    embed_dim: int = SBERT_DIM
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load and embed text features from MongoDB for A-B and B-C entity pairs.
    Use the same convention as RGCN: call get_pair_evidence_embedding(pair_type, key1_name, key2_name, ...)
    and return numeric tensors (no raw strings to tensor!).
    """
    text_features_ab: Dict[int, torch.Tensor] = {}
    text_features_bc: Dict[int, torch.Tensor] = {}

    try:
        from sentence_transformers import SentenceTransformer
        from ..config import SBERT_MODEL_NAME
        encoder = SentenceTransformer(SBERT_MODEL_NAME)
    except Exception:
        encoder = None  # 回退：下面调用将使用缓存或返回零

    for idx, p in enumerate(paths):
        try:
            # A-B: slc_pathway using names
            a_name = p.get("A_name", "")
            b_name = p.get("B_name", "")
            if a_name and b_name:
                pair_type = "slc_pathway"
                _, emb_ab = get_pair_evidence_embedding(pair_type, a_name, b_name, encoder=encoder, reencode=False)
                if emb_ab is None:
                    emb_ab = [0.0] * embed_dim
                text_features_ab[idx] = torch.tensor(emb_ab, dtype=torch.float32)

            # B-C: pathway_disease using names
            b_name2 = p.get("B_name", "")
            c_name = p.get("C_name", "")
            if b_name2 and c_name:
                pair_type = "pathway_disease"
                _, emb_bc = get_pair_evidence_embedding(pair_type, b_name2, c_name, encoder=encoder, reencode=False)
                if emb_bc is None:
                    emb_bc = [0.0] * embed_dim
                text_features_bc[idx] = torch.tensor(emb_bc, dtype=torch.float32)

        except Exception as e:
            print(f"  Warning: Failed to fetch text for path {idx}: {e}")
            # 回退为零向量，避免后续报错
            if idx not in text_features_ab:
                text_features_ab[idx] = torch.zeros(embed_dim, dtype=torch.float32)
            if idx not in text_features_bc:
                text_features_bc[idx] = torch.zeros(embed_dim, dtype=torch.float32)
            continue

    print(f"  ✓ Loaded text features for {len(text_features_ab)} A-B pairs")
    print(f"  ✓ Loaded text features for {len(text_features_bc)} B-C pairs")

    return text_features_ab, text_features_bc


def get_text_embeddings_for_batch(
    batch_indices: torch.Tensor,
    text_features_dict: Dict[int, torch.Tensor],
    embed_dim: int = SBERT_DIM,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Get text embeddings for a batch of indices.
    
    Args:
        batch_indices: Tensor of indices into the batch
        text_features_dict: Dictionary mapping index to embeddings
        embed_dim: Embedding dimension
        device: Device to place tensor on
    
    Returns:
        Batch tensor of text embeddings [batch_size, embed_dim]
        Returns zeros if text features not available
    """
    device = resolve_device(device)

    batch_size = len(batch_indices)
    batch_embeddings = torch.zeros(batch_size, embed_dim, dtype=torch.float32, device=device)
    
    for i, idx in enumerate(batch_indices.cpu().numpy()):
        if int(idx) in text_features_dict:
            batch_embeddings[i] = text_features_dict[int(idx)].to(device)
    
    return batch_embeddings


def build_label_maps(relation_type_map):
    """Build bidirectional mapping between relation names and IDs."""
    name_to_id = {k: v for k, v in relation_type_map.items()}
    id_to_name = {v: k for k, v in name_to_id.items()}
    return name_to_id, id_to_name


def build_relation_type_map_from_paths(paths: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Extract relation types from N-ary paths and build mapping.
    
    In N-ary model:
    - rel_AB and rel_BC are stored as attributes on RelaEvent nodes
    - We extract these values from path data
    
    Args:
        paths: List of path dictionaries with rel_AB and rel_BC fields
        
    Returns:
        Tuple of (name_to_id, id_to_name) mappings
    """
    relation_types = set()
    for p in paths:
        if "rel_AB" in p:
            relation_types.add(p["rel_AB"])
        if "rel_BC" in p:
            relation_types.add(p["rel_BC"])
    
    # Sort for consistent ordering
    sorted_types = sorted(list(relation_types))
    name_to_id = {name: idx for idx, name in enumerate(sorted_types)}
    id_to_name = {idx: name for idx, name in enumerate(sorted_types)}
    return name_to_id, id_to_name


def prepare_batches(paths: List[Dict], name_to_id: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """Prepare batches from paths."""
    A = torch.tensor([p["A"] for p in paths], dtype=torch.long)
    B = torch.tensor([p["B"] for p in paths], dtype=torch.long)
    C = torch.tensor([p["C"] for p in paths], dtype=torch.long)
    Event = torch.tensor([p.get("Event", p["A"]) for p in paths], dtype=torch.long)  # Use Event if available, else A
    y_ab = torch.tensor([name_to_id[p["rel_AB"]] for p in paths], dtype=torch.long)
    y_bc = torch.tensor([name_to_id[p["rel_BC"]] for p in paths], dtype=torch.long)
    return {"A": A, "B": B, "C": C, "Event": Event, "y_ab": y_ab, "y_bc": y_bc}


def evaluate_path_metrics(
    hype_backbone: HypEBackbone,
    predictor: CascadingHypEPredictor,
    samples: List[Dict],
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
    text_features_ab: Dict[int, torch.Tensor],
    text_features_bc: Dict[int, torch.Tensor],
    text_projector: HypETextProjector | None,
    use_text_features: bool,
    node_features: torch.Tensor | None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """
    Evaluate cascading metrics using the backbone to encode entire batches of paths.
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

    hype_backbone.eval()
    predictor.eval()

    batch = prepare_batches(samples, name_to_id)
    A = batch["A"].to(device)
    B = batch["B"].to(device)
    C = batch["C"].to(device)
    Event = batch["Event"].to(device)

    text_ab_proj = None
    text_bc_proj = None
    if use_text_features and text_projector is not None:
        idx_tensor = torch.arange(len(samples), dtype=torch.long, device=device)
        text_ab_raw = get_text_embeddings_for_batch(idx_tensor.cpu(), text_features_ab, embed_dim=SBERT_DIM, device=device)
        text_bc_raw = get_text_embeddings_for_batch(idx_tensor.cpu(), text_features_bc, embed_dim=SBERT_DIM, device=device)
        text_ab_proj = text_projector(text_ab_raw)
        text_bc_proj = text_projector(text_bc_raw)

    with torch.no_grad():
        repr_ab = hype_backbone.encode_stage(
            "ab",
            A,
            B,
            C,
            Event,
            node_features=node_features,
            text_features=text_ab_proj,
        )
        repr_bc = hype_backbone.encode_stage(
            "bc",
            A,
            B,
            C,
            Event,
            node_features=node_features,
            text_features=text_bc_proj,
        )
        logits_ab, logits_bc, rel_ab_pred = predictor(repr_ab, repr_bc, training=False)

    all_true_ab = batch["y_ab"].cpu().numpy()
    all_true_bc = batch["y_bc"].cpu().numpy()
    probs_ab = F.softmax(logits_ab, dim=-1).detach().cpu().numpy()
    probs_bc = F.softmax(logits_bc, dim=-1).detach().cpu().numpy()
    all_pred_ab = probs_ab.argmax(axis=-1)
    all_pred_bc = probs_bc.argmax(axis=-1)

    y_true_path_4class = np.zeros(len(samples), dtype=np.int64)
    y_pred_path_4class = []
    for pa, pb in zip(all_pred_ab == all_true_ab, all_pred_bc == all_true_bc):
        if pa and pb:
            y_pred_path_4class.append(0)
        elif pa and not pb:
            y_pred_path_4class.append(1)
        elif (not pa) and pb:
            y_pred_path_4class.append(2)
        else:
            y_pred_path_4class.append(3)

    ab_acc = accuracy_score(all_true_ab, all_pred_ab)
    ab_f1 = f1_score(all_true_ab, all_pred_ab, average="macro", zero_division=0)
    ab_mcc = matthews_corrcoef(all_true_ab, all_pred_ab)

    bc_acc = accuracy_score(all_true_bc, all_pred_bc)
    bc_f1 = f1_score(all_true_bc, all_pred_bc, average="macro", zero_division=0)
    bc_mcc = matthews_corrcoef(all_true_bc, all_pred_bc)

    path_acc = accuracy_score(y_true_path_4class, y_pred_path_4class)
    path_f1 = f1_score(
        y_true_path_4class,
        y_pred_path_4class,
        average="macro",
        labels=[0],
        zero_division=0,
    )

    try:
        ab_auc_roc = roc_auc_score(all_true_ab, probs_ab[:, 1]) if len(np.unique(all_true_ab)) > 1 else 0.0
    except ValueError:
        ab_auc_roc = 0.0

    try:
        bc_auc_roc = roc_auc_score(all_true_bc, probs_bc[:, 1]) if len(np.unique(all_true_bc)) > 1 else 0.0
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
    ab_bal_acc = balanced_accuracy_score(all_true_ab, all_pred_ab) if len(labels_ab) > 1 else ab_acc
    ab_label_names = [str(id_to_name.get(int(lbl), str(lbl))) for lbl in labels_ab]

    labels_bc = sorted(set(all_true_bc) | set(all_pred_bc))
    bc_cm = confusion_matrix(all_true_bc, all_pred_bc, labels=labels_bc).tolist()
    bc_prec, bc_rec, bc_f1_cls, bc_sup = precision_recall_fscore_support(
        all_true_bc,
        all_pred_bc,
        labels=labels_bc,
        zero_division=0,
    )
    bc_bal_acc = balanced_accuracy_score(all_true_bc, all_pred_bc) if len(labels_bc) > 1 else bc_acc
    bc_label_names = [str(id_to_name.get(int(lbl), str(lbl))) for lbl in labels_bc]

    return {
        "path_acc": float(path_acc),
        "path_f1": float(path_f1),
        "ab_acc": float(ab_acc),
        "ab_f1": float(ab_f1),
        "ab_auc_roc": float(ab_auc_roc),
        "ab_mcc": float(ab_mcc),
        "bc_acc": float(bc_acc),
        "bc_f1": float(bc_f1),
        "bc_auc_roc": float(bc_auc_roc),
        "bc_mcc": float(bc_mcc),
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
    Training pipeline for HypE model on N-ary Knowledge Graph.
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        embedding_dim: Embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of HypE layers
        dropout: Dropout rate
        val_every: Validation frequency
        use_text_features: Whether to use text features from MongoDB
        use_node_features: Whether to use node features
        device: Device to use
    
    Returns:
        Dictionary containing:
        - hype_model: Trained HypE model
        - predictor: Trained cascading predictor
        - graph: DGL graph
        - name_to_id: Relation name to ID mapping
        - id_to_name: Relation ID to name mapping
        - test_metrics: Test set metrics
        - val_metrics: Validation set metrics
        - best_val_path_acc: Best validation path accuracy
        - best_epoch: Best epoch
    """
    print("=" * 80)
    print("HypE Training Pipeline for N-ary Knowledge Graph (with Cascading)")
    print("=" * 80)
    
    device = resolve_device(device)
    print(f"[Device] Using {device} for HypE pipeline")
    # 图保持在CPU上，避免DGL CUDA依赖问题
    print(f"[Device] Graph will stay on CPU (to avoid DGL CUDA dependency)")

    # Step 1: Build graph
    print("\n[1/8] Building DGL graph from Neo4j...")
    g = build_dgl_graph()[0]  # 图保持在CPU上
    print(f"  ✓ Graph built: {g.num_nodes()} nodes, {g.num_edges()} edges")
    
    # Step 2: Handle path splits (internal vs external)
    if train_paths is None and val_paths is None and test_paths is None:
        print("\n[2/8] Enumerating N-ary paths...")
        all_paths = enumerate_graph_paths()
        print(f"  ✓ Found {len(all_paths)} paths")

        if len(all_paths) == 0:
            print("  ✗ No paths found! Check Neo4j query and graph structure.")
            return None

        # Step 3: Build relation mappings
        print("\n[3/8] Building relation type mappings from paths...")
        name_to_id, id_to_name = build_relation_type_map_from_paths(all_paths)
        num_relation_classes = len(name_to_id)
        print(f"  ✓ Found {num_relation_classes} unique relation types:")
        for name, idx in name_to_id.items():
            print(f"      {idx}: {name}")

        # Step 4: Split data
        print("\n[4/8] Splitting data into train/val/test...")
        train_pos, val_pos, test_pos = split_paths(all_paths, train_ratio=0.7, val_ratio=0.15)
        if few_shot_k:
            train_pos = select_few_shot(train_pos, few_shot_k, seed=seed, balance_by=few_shot_balance)
        print(f"  ✓ Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}")
    else:
        if train_paths is None or val_paths is None:
            raise ValueError("train_paths and val_paths must be provided when using external splits")

        print("\n[2/8] Using externally provided train/val/test splits...")
        train_pos = list(train_paths)
        val_pos = list(val_paths)
        test_pos = list(test_paths) if test_paths is not None else list(val_pos)
        print(f"  ✓ Provided splits - Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}")

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

    # Step 5: Generate negative samples (optional)
    if skip_negatives:
        print("\n[5/8] Skipping negative sampling (skip_negatives=True)...")
        train_neg = []
    else:
        print("\n[5/8] Generating negative samples...")
        train_neg = generate_negatives(train_pos, all_paths)
        print(f"  ✓ Generated {len(train_neg)} negative samples for training")
    
    # Step 6: Initialize models
    print("\n[6/8] Initializing HypE backbone with cascading head...")
    num_nodes = g.num_nodes()
    node_feat_dim = g.ndata["feat"].shape[1] if (use_node_features and "feat" in g.ndata) else 0
    if node_feat_dim > 0:
        print(f"  ✓ Node features detected: dimension = {node_feat_dim}")
    else:
        print("  ⊗ Node attribute support: DISABLED")

    hype_model = HypEBackbone(
        num_entities=num_nodes,
        embedding_dim=embedding_dim,
        kernel_size=3,
        num_filters=embedding_dim,
        dropout=dropout,
        node_feat_dim=node_feat_dim,
    ).to(device)

    predictor = CascadingHypEPredictor(
        embedding_dim=embedding_dim,
        num_relation_classes=num_relation_classes,
        dropout=dropout,
    ).to(device)

    text_projector = None
    if use_text_features:
        text_projector = HypETextProjector(
            text_dim=SBERT_DIM,
            embedding_dim=embedding_dim,
            dropout=dropout,
        ).to(device)
        print("  ✓ Text feature support: ENABLED (MongoDB integration)")
    else:
        print("  ⊗ Text feature support: DISABLED")

    params = list(hype_model.parameters()) + list(predictor.parameters())
    if text_projector is not None:
        params += list(text_projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # Prepare batches
    train_batch_pos = prepare_batches(train_pos, name_to_id)
    train_batch_neg = prepare_batches(train_neg, name_to_id) if len(train_neg) > 0 else None
    
    # Load text features from MongoDB if enabled
    text_features_ab_train = {}
    text_features_bc_train = {}
    text_features_ab_val = {}
    text_features_bc_val = {}
    text_features_ab_test = {}
    text_features_bc_test = {}
    
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
    
    node_features = g.ndata.get("feat", None)
    if node_features is not None and use_node_features:
        node_features = node_features.to(device)
    else:
        node_features = None

    if use_text_features:
        train_idx_tensor = torch.arange(len(train_pos), dtype=torch.long)
        train_text_ab_tensor = get_text_embeddings_for_batch(
            train_idx_tensor, text_features_ab_train, embed_dim=SBERT_DIM, device=device
        )
        train_text_bc_tensor = get_text_embeddings_for_batch(
            train_idx_tensor, text_features_bc_train, embed_dim=SBERT_DIM, device=device
        )
    else:
        train_text_ab_tensor = None
        train_text_bc_tensor = None
    
    # Step 7: Training loop
    print("\n[7/8] Training HypE model with Cascading...")
    print(f"  Epochs: {epochs}, LR: {lr}, Device: {device}")
    print("-" * 80)
    
    best_val_path_acc = 0.0
    best_epoch = 0
    best_val_metrics = None
    last_val_metrics = None
    
    for epoch in range(epochs):
        hype_model.train()
        predictor.train()
        
        # Forward pass
        optimizer.zero_grad()
        
        if use_text_features and text_projector is not None:
            text_proj_ab = text_projector(train_text_ab_tensor)
            text_proj_bc = text_projector(train_text_bc_tensor)
        else:
            text_proj_ab = None
            text_proj_bc = None

        repr_ab_pos = hype_model.encode_stage(
            "ab",
            train_batch_pos["A"],
            train_batch_pos["B"],
            train_batch_pos["C"],
            train_batch_pos["Event"],
            node_features=node_features,
            text_features=text_proj_ab,
        )
        repr_bc_pos = hype_model.encode_stage(
            "bc",
            train_batch_pos["A"],
            train_batch_pos["B"],
            train_batch_pos["C"],
            train_batch_pos["Event"],
            node_features=node_features,
            text_features=text_proj_bc,
        )

        logits_ab_pos, logits_bc_pos, _ = predictor(repr_ab_pos, repr_bc_pos, training=True)
        
        loss_ab = F.cross_entropy(logits_ab_pos, train_batch_pos["y_ab"].to(device))
        loss_bc = F.cross_entropy(logits_bc_pos, train_batch_pos["y_bc"].to(device))
        loss_pos = loss_ab + loss_bc
        
        # Negative samples (if any)
        if train_batch_neg is not None and len(train_neg) > 0:
            repr_ab_neg = hype_model.encode_stage(
                "ab",
                train_batch_neg["A"],
                train_batch_neg["B"],
                train_batch_neg["C"],
                train_batch_neg["Event"],
                node_features=node_features,
                text_features=None,
            )
            repr_bc_neg = hype_model.encode_stage(
                "bc",
                train_batch_neg["A"],
                train_batch_neg["B"],
                train_batch_neg["C"],
                train_batch_neg["Event"],
                node_features=node_features,
                text_features=None,
            )
            
            logits_ab_neg, logits_bc_neg, _ = predictor(repr_ab_neg, repr_bc_neg, training=True)
            
            loss_ab_neg = F.cross_entropy(logits_ab_neg, train_batch_neg["y_ab"].to(device))
            loss_bc_neg = F.cross_entropy(logits_bc_neg, train_batch_neg["y_bc"].to(device))
            loss_neg = loss_ab_neg + loss_bc_neg
            
            total_loss = loss_pos + 0.5 * loss_neg
        else:
            total_loss = loss_pos
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Validation
        if (epoch + 1) % val_every == 0:
            print(f"\nEpoch {epoch+1}/{epochs} - Loss: {total_loss.item():.4f}")
            
            # Evaluate on validation set (backbone already encodes entire batch)
            val_metrics = evaluate_path_metrics(
                hype_backbone=hype_model,
                predictor=predictor,
                samples=val_pos,
                name_to_id=name_to_id,
                id_to_name=id_to_name,
                text_features_ab=text_features_ab_val,
                text_features_bc=text_features_bc_val,
                text_projector=text_projector,
                use_text_features=use_text_features,
                node_features=node_features,
                device=device,
            )
            last_val_metrics = val_metrics
            
            print(f"  [VAL] Path Acc: {val_metrics['path_acc']:.4f}, Path F1: {val_metrics['path_f1']:.4f}")
            print(f"  [VAL] A-B  Acc: {val_metrics['ab_acc']:.4f}, F1: {val_metrics['ab_f1']:.4f}, "
                  f"AUC: {val_metrics['ab_auc_roc']:.4f}, MCC: {val_metrics['ab_mcc']:.4f}")
            print(f"  [VAL] B-C  Acc: {val_metrics['bc_acc']:.4f}, F1: {val_metrics['bc_f1']:.4f}, "
                  f"AUC: {val_metrics['bc_auc_roc']:.4f}, MCC: {val_metrics['bc_mcc']:.4f}")
            
            # Track best model
            if val_metrics['path_acc'] > best_val_path_acc:
                best_val_path_acc = val_metrics['path_acc']
                best_epoch = epoch + 1
                best_val_metrics = val_metrics.copy()
                print(f"  ✓ New best Path Acc: {best_val_path_acc:.4f}")
    
    # Step 8: Final evaluation on test set
    print("\n" + "=" * 80)
    print("[8/8] Final Evaluation on Test Set (with Cascading)")
    print("=" * 80)
    
    test_metrics = evaluate_path_metrics(
        hype_backbone=hype_model,
        predictor=predictor,
        samples=test_pos,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        text_features_ab=text_features_ab_test,
        text_features_bc=text_features_bc_test,
        text_projector=text_projector,
        use_text_features=use_text_features,
        node_features=node_features,
        device=device,
    )
    
    print(f"\n[TEST] Path Accuracy:  {test_metrics['path_acc']:.4f} (Cascading)")
    print(f"[TEST] Path F1-Score:  {test_metrics['path_f1']:.4f} (Cascading)")
    print(f"\n[TEST] A-B Component (First Prediction):")
    print(f"  Accuracy:   {test_metrics['ab_acc']:.4f}")
    print(f"  F1-Macro:   {test_metrics['ab_f1']:.4f}")
    print(f"  AUC-ROC:    {test_metrics['ab_auc_roc']:.4f}")
    print(f"  MCC:        {test_metrics['ab_mcc']:.4f}")
    print(f"\n[TEST] B-C Component (Cascaded Prediction based on predicted rel_AB):")
    print(f"  Accuracy:   {test_metrics['bc_acc']:.4f}")
    print(f"  F1-Macro:   {test_metrics['bc_f1']:.4f}")
    print(f"  AUC-ROC:    {test_metrics['bc_auc_roc']:.4f}")
    print(f"  MCC:        {test_metrics['bc_mcc']:.4f}")
    
    print(f"\nBest validation Path Acc: {best_val_path_acc:.4f} (epoch {best_epoch})")
    print("=" * 80)

    if best_val_metrics is None:
        if last_val_metrics is not None:
            best_val_metrics = last_val_metrics
            best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
            best_epoch = epochs
        else:
            best_val_metrics = evaluate_path_metrics(
                hype_backbone=hype_model,
                predictor=predictor,
                samples=val_pos,
                name_to_id=name_to_id,
                id_to_name=id_to_name,
                text_features_ab=text_features_ab_val,
                text_features_bc=text_features_bc_val,
                text_projector=text_projector,
                use_text_features=use_text_features,
                node_features=node_features,
                device=device,
            )
            best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
            best_epoch = epochs
    
    return {
        "hype_model": hype_model,
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

