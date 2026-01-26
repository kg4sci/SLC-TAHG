"""
Training pipeline for StarE model on N-ary Knowledge Graphs.

This module adapts the StarE architecture for the specific task:
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

from .models import StarE, CascadingStarEPredictor, StarEWithTextProjector
from ..graph_build import build_dgl_graph
from ..path_data import enumerate_graph_paths, split_paths, generate_negatives, select_few_shot
from ..db_mongo import fetch_pair_evidence_text, get_pair_evidence_embedding
from ..config import NODE_NAME_FIELD, NEIGHBOR_AGG_METHOD, LABEL_SLC, SBERT_DIM
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
from ..device_utils import resolve_device


def build_stare_graph_repr_from_dgl(g):
    """
    Convert the constructed DGL graph into the numpy-based representation that
    StarEEncoder expects (edge_index with both directions + edge_type ids).
    """
    src, dst = g.edges()
    rel_types = g.edata["rel_type"]

    src = src.long().cpu().numpy()
    dst = dst.long().cpu().numpy()
    rel_types = rel_types.long().cpu().numpy()

    num_edges = src.shape[0]
    if num_edges == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)
        num_rel = 1
    else:
        num_rel = int(rel_types.max()) + 1
        edge_index = np.zeros((2, num_edges * 2), dtype=np.int64)
        edge_index[:, :num_edges] = np.stack([src, dst], axis=0)
        edge_index[:, num_edges:] = np.stack([dst, src], axis=0)

        edge_type = np.zeros(num_edges * 2, dtype=np.int64)
        edge_type[:num_edges] = rel_types
        edge_type[num_edges:] = rel_types + num_rel

    return {"edge_index": edge_index, "edge_type": edge_type}, num_rel


def build_stare_config(
    num_entities: int,
    num_relations: int,
    embedding_dim: int,
    gcn_dim: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
):
    """
    Minimal StarE configuration dictionary for the encoder backbone.
    """
    device = torch.device(device)
    stare_args = {
        "N_BASES": max(1, min(num_relations, 64)),
        "LAYERS": num_layers,
        "GCN_DIM": gcn_dim,
        "HID_DROP": dropout,
        "FEAT_DROP": dropout,
        "QUAL_REPR": "none",
        "QUAL_AGGREGATE": "sum",
        "QUAL_OPN": "corr",
        "QUAL_N": "sum",
        "OPN": "corr",
        "GCN_DROP": dropout,
        "ATTENTION": False,
        "ATTENTION_HEADS": 1,
        "ATTENTION_SLOPE": 0.2,
        "ATTENTION_DROP": 0.0,
        "BIAS": True,
        "TRIPLE_QUAL_WEIGHT": 0.5,
        "STATEMENT_LEN": 3,
    }

    return {
        "MODEL_NAME": "stare",
        "EMBEDDING_DIM": embedding_dim,
        "NUM_RELATIONS": num_relations,
        "NUM_ENTITIES": num_entities,
        "DEVICE": device,
        "STATEMENT_LEN": 3,
        "STAREARGS": stare_args,
    }


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
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    
    return name_to_id, id_to_name


def prepare_batches(paths: List[Dict], name_to_id: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    Prepare batched tensors from path data for StarE model.
    
    Args:
        paths: List of path dictionaries
        name_to_id: Relation name to ID mapping
        
    Returns:
        Dictionary of tensors: A, B, C, Event, y_ab, y_bc
    """
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
    Event_ids = [p.get("Event", -1) for p in paths]  # RelaEvent node ID
    
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
    stare_model,
    predictor,
    samples: List[Dict],
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
    text_features_ab: Dict[int, torch.Tensor],
    text_features_bc: Dict[int, torch.Tensor],
    use_text_features: bool,
    device: Optional[Union[str, torch.device]] = None
) -> Dict[str, float]:
    """
    Evaluate StarE model on N-ary path prediction with CASCADING.
    
    ⭐ CASCADING EVALUATION:
    1. First predict rel_AB using node embeddings
    2. Then predict rel_BC using the PREDICTED rel_AB (not ground truth!)
    3. This is realistic because in real scenarios we don't know ground truth rel_AB
    
    Metrics:
    1. A-B prediction: ACC, F1, AUC-ROC, MCC
    2. B-C prediction: ACC, F1, AUC-ROC, MCC
    3. Path prediction: Path ACC, Path F1
    
    Args:
        stare_model: Trained StarE encoder
        predictor: Trained CascadingStarEPredictor
        g: DGL graph
        samples: List of path dictionaries
        name_to_id: Relation name to ID mapping
        id_to_name: Relation ID to name mapping
        device: Computing device
        
    Returns:
        Dictionary of evaluation metrics
    """
    device = resolve_device(device)

    if len(samples) == 0:
        return {
            "path_acc": 0.0, "path_f1": 0.0,
            "ab_acc": 0.0, "ab_f1": 0.0, "ab_auc_roc": 0.0, "ab_mcc": 0.0,
            "bc_acc": 0.0, "bc_f1": 0.0, "bc_auc_roc": 0.0, "bc_mcc": 0.0
        }
    
    stare_model.eval()
    predictor.eval()
    
    with torch.no_grad():
        model_output = stare_model()
        if isinstance(model_output, tuple):
            node_h = model_output[0]
        else:
            node_h = model_output
    
    # Collect predictions
    all_true_ab, all_pred_ab, all_scores_ab = [], [], []
    all_true_bc, all_pred_bc, all_scores_bc = [], [], []
    y_true_path_4class = []  # 0=(T,T), 1=(T,F), 2=(F,T), 3=(F,F)
    y_pred_path_4class = []
    
    with torch.no_grad():
        for p in samples:
            A, B, C = p["A"], p["B"], p["C"]
            Event = p.get("Event", 0)  # RelaEvent node
            
            true_ab = name_to_id[p["rel_AB"]]
            true_bc = name_to_id[p["rel_BC"]]
            
            # Get embeddings
            h_a = node_h[A].unsqueeze(0)
            h_b = node_h[B].unsqueeze(0)
            h_c = node_h[C].unsqueeze(0)
            h_event = node_h[Event].unsqueeze(0)
            
            # Predict with CASCADING
            logits_ab, logits_bc, rel_ab_pred = predictor(h_a, h_b, h_c, h_event, training=False)
            
            # A-B predictions
            scores_ab = F.softmax(logits_ab, dim=-1).cpu().numpy()[0]
            pred_ab = int(torch.argmax(logits_ab, dim=-1).item())
            all_true_ab.append(true_ab)
            all_pred_ab.append(pred_ab)
            all_scores_ab.append(scores_ab[1] if len(scores_ab) > 1 else scores_ab[0])
            
            # B-C predictions (already cascaded from predicted rel_AB)
            scores_bc = F.softmax(logits_bc, dim=-1).cpu().numpy()[0]
            pred_bc = int(torch.argmax(logits_bc, dim=-1).item())
            all_true_bc.append(true_bc)
            all_pred_bc.append(pred_bc)
            all_scores_bc.append(scores_bc[1] if len(scores_bc) > 1 else scores_bc[0])
            
            # Path 4-class evaluation
            y_true_path_4class.append(0)  # Ground truth is always (T, T)
            
            is_ab_correct = (pred_ab == true_ab)
            is_bc_correct = (pred_bc == true_bc)
            
            if is_ab_correct and is_bc_correct:
                y_pred_path_4class.append(0)  # (T, T)
            elif is_ab_correct and not is_bc_correct:
                y_pred_path_4class.append(1)  # (T, F)
            elif not is_ab_correct and is_bc_correct:
                y_pred_path_4class.append(2)  # (F, T)
            else:
                y_pred_path_4class.append(3)  # (F, F)
    
    # Calculate metrics
    # A-B metrics
    acc_ab = accuracy_score(all_true_ab, all_pred_ab)
    f1_ab = f1_score(all_true_ab, all_pred_ab, average='macro', zero_division=0)
    mcc_ab = matthews_corrcoef(all_true_ab, all_pred_ab)
    
    auc_roc_ab = 0.0
    try:
        if len(np.unique(all_true_ab)) > 1:
            auc_roc_ab = roc_auc_score(all_true_ab, all_scores_ab)
        else:
            auc_roc_ab = acc_ab
    except ValueError:
        pass
    
    # B-C metrics
    acc_bc = accuracy_score(all_true_bc, all_pred_bc)
    f1_bc = f1_score(all_true_bc, all_pred_bc, average='macro', zero_division=0)
    mcc_bc = matthews_corrcoef(all_true_bc, all_pred_bc)
    
    auc_roc_bc = 0.0
    try:
        if len(np.unique(all_true_bc)) > 1:
            auc_roc_bc = roc_auc_score(all_true_bc, all_scores_bc)
        else:
            auc_roc_bc = acc_bc
    except ValueError:
        pass
    
    # Path metrics
    path_acc = accuracy_score(y_true_path_4class, y_pred_path_4class)
    path_f1 = f1_score(y_true_path_4class, y_pred_path_4class, labels=[0], average='macro', zero_division=0)
    
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
    Complete training pipeline for StarE model on N-ary KG with:
    ✅ Cascading prediction (rel_AB → rel_BC)
    ✅ Node attribute encoding (SLC neighbors, etc.)
    ✅ Text feature integration (MongoDB)
    
    Pipeline:
    1. Build DGL graph from Neo4j and convert it to StarE's edge representation
    2. Enumerate N-ary paths (A-Event-B, Event-C)
    3. Build relation type mappings from paths
    4. Split into train/val/test
    5. Generate negative samples
    6. Train StarE model with cascading predictor
    7. Evaluate on val/test sets
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        embedding_dim: Dimension of entity embeddings
        hidden_dim: Hidden dimension for StarE layers
        num_layers: Number of StarE layers
        dropout: Dropout rate
        val_every: Validation frequency
        use_text_features: Whether to use text evidence from MongoDB
        use_node_features: Whether to use node attribute features (ignored for original backbone)
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing:
        - trained models
        - evaluation results
        - relation mappings
    """
    device = resolve_device(device)
    print("=" * 80)
    print("StarE Training Pipeline for N-ary Knowledge Graph (with Cascading)")
    print("=" * 80)
    print(f"[Device] Using {device} for StarE pipeline")
    # 图保持在CPU上，避免DGL CUDA依赖问题
    print(f"[Device] Graph will stay on CPU (to avoid DGL CUDA dependency)")
    
    # Step 1: Build graph
    print("\n[1/8] Building DGL graph from Neo4j...")
    g = build_dgl_graph()[0]  # 图保持在CPU上
    print(f"  ✓ Graph built: {g.num_nodes()} nodes, {g.num_edges()} edges")
    
    # Step 2: Enumerate paths
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
        train_pos, val_pos, test_pos = split_paths(all_paths, train_ratio=0.7, val_ratio=0.15, seed=seed)
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
        print(f"  ✓ Found {num_relation_classes} unique relation types:")
        for name, idx in name_to_id.items():
            print(f"      {idx}: {name}")

        all_paths = combined_paths
        if few_shot_k:
            train_pos = select_few_shot(train_pos, few_shot_k, seed=seed, balance_by=few_shot_balance)
    
    # Step 5: Generate negative samples
    if skip_negatives:
        print("\n[5/8] Skipping negative sampling (skip_negatives=True)...")
        train_neg = []
    else:
        print("\n[5/8] Generating negative samples...")
        train_neg = generate_negatives(train_pos, all_paths)
        print(f"  ✓ Generated {len(train_neg)} negative samples for training")
    
    # Step 6: Initialize models
    print("\n[6/8] Initializing StarE model with Cascading Predictor...")
    num_nodes = g.num_nodes()
    graph_repr, base_num_relations = build_stare_graph_repr_from_dgl(g)

    # 节点特征处理（可选）
    node_features = None
    node_feat_dim = 0
    # 计算实体ID上界（路径中可能超过图节点数，需要扩展特征表）
    max_entity_id = max(max(p["A"], p["B"], p["C"], p.get("Event", 0)) for p in all_paths)
    num_entities = max(num_nodes, max_entity_id + 1)

    if use_node_features and "feat" in g.ndata:
        node_features_cpu = g.ndata["feat"]
        node_feat_dim = node_features_cpu.shape[1]
        if num_entities > num_nodes:
            expanded = torch.zeros(
                num_entities, node_feat_dim,
                dtype=node_features_cpu.dtype,
                device=torch.device("cpu"),
            )
            expanded[:num_nodes] = node_features_cpu
            node_features_cpu = expanded
            print(f"  ✓ Expanded node features from [{num_nodes}, {node_feat_dim}] to [{num_entities}, {node_feat_dim}]")
        node_features = node_features_cpu.to(device)
        print(f"  ✓ Node feature support: ENABLED (dim={node_feat_dim})")
    elif use_node_features:
        print("  ⚠ use_node_features=True 但图中没有节点特征，继续使用随机嵌入")

    stare_config = build_stare_config(
        num_entities=num_entities,
        num_relations=base_num_relations,
        embedding_dim=embedding_dim,
        gcn_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )
    
    if use_text_features:
        from .models import StarEWithTextProjector
        stare_model = StarEWithTextProjector(
            graph_repr=graph_repr,
            config=stare_config,
            num_relation_classes=num_relation_classes,
            text_dim=SBERT_DIM,
            dropout=dropout,
            node_features=node_features,
            node_feat_dim=node_feat_dim,
        ).to(device)
        predictor = stare_model.predictor
        print(f"  ✓ StarEWithTextProjector (original backbone): {sum(p.numel() for p in stare_model.parameters())} parameters")
        print(f"  ✓ Text feature support: ENABLED (MongoDB integration)")
    else:
        stare_model = StarE(
            graph_repr=graph_repr,
            config=stare_config,
            node_features=node_features,
            node_feat_dim=node_feat_dim,
        ).to(device)
        predictor = CascadingStarEPredictor(
            embedding_dim=embedding_dim,
            num_relation_classes=num_relation_classes,
            dropout=dropout
        ).to(device)
        print(f"  ✓ StarE backbone parameters: {sum(p.numel() for p in stare_model.parameters())}")
        print(f"  ✓ CascadingStarEPredictor parameters: {sum(p.numel() for p in predictor.parameters())}")
        print(f"  ⊗ Text feature support: DISABLED")
    
    
    # Optimizer
    if use_text_features:
        optimizer_params = stare_model.parameters()
    else:
        optimizer_params = list(stare_model.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    
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
    
    # Step 7: Training loop
    print("\n[7/8] Training StarE model with Cascading...")
    print(f"  Epochs: {epochs}, LR: {lr}, Device: {device}")
    print("-" * 80)
    
    best_val_path_acc = 0.0
    best_epoch = 0
    best_val_metrics = None
    last_val_metrics = None
    
    for epoch in range(epochs):
        stare_model.train()
        predictor.train()
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get node embeddings
        if use_text_features:
            # Get text embeddings for positive samples
            text_ab_pos = get_text_embeddings_for_batch(
                train_batch_pos["A"], text_features_ab_train, embed_dim=SBERT_DIM, device=device
            )
            text_bc_pos = get_text_embeddings_for_batch(
                train_batch_pos["B"], text_features_bc_train, embed_dim=SBERT_DIM, device=device
            )
            node_h, (text_proj_ab, text_proj_bc) = stare_model(
                text_features_ab=text_ab_pos,
                text_features_bc=text_bc_pos,
            )
        else:
            node_h = stare_model()
            text_proj_ab = None
            text_proj_bc = None
        
        # Positive samples: slice node embeddings for A, B, C, Event
        h_a_pos = node_h[train_batch_pos["A"].to(device)]
        h_b_pos = node_h[train_batch_pos["B"].to(device)]
        h_c_pos = node_h[train_batch_pos["C"].to(device)]
        h_event_pos = node_h[train_batch_pos["Event"].to(device)]
        
        # Text fusion (add) on batch-aligned tensors (if provided)
        if use_text_features and text_proj_ab is not None:
            h_a_pos = h_a_pos + text_proj_ab
            h_b_pos = h_b_pos + text_proj_ab
        if use_text_features and text_proj_bc is not None:
            h_c_pos = h_c_pos + text_proj_bc
        
        # Cascading prediction: rel_BC uses predicted rel_AB
        logits_ab_pos, logits_bc_pos, _ = predictor(h_a_pos, h_b_pos, h_c_pos, h_event_pos, training=True)
        
        loss_ab = F.cross_entropy(logits_ab_pos, train_batch_pos["y_ab"].to(device))
        loss_bc = F.cross_entropy(logits_bc_pos, train_batch_pos["y_bc"].to(device))
        loss_pos = loss_ab + loss_bc
        
        # Negative samples (if any)
        if train_batch_neg is not None and len(train_neg) > 0:
            h_a_neg = node_h[train_batch_neg["A"].to(device)]
            h_b_neg = node_h[train_batch_neg["B"].to(device)]
            h_c_neg = node_h[train_batch_neg["C"].to(device)]
            h_event_neg = node_h[train_batch_neg["Event"].to(device)]
            
            logits_ab_neg, logits_bc_neg, _ = predictor(h_a_neg, h_b_neg, h_c_neg, h_event_neg, training=True)
            
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
            
            # Evaluate on validation set
            val_metrics = evaluate_path_metrics(
                stare_model, predictor, val_pos, name_to_id, id_to_name,
                text_features_ab=text_features_ab_val,
                text_features_bc=text_features_bc_val,
                use_text_features=use_text_features,
                device=device
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
                print(f" New best Path Acc: {best_val_path_acc:.4f}")
    
    # Step 8: Final evaluation on test set
    print("\n" + "=" * 80)
    print("[8/8] Final Evaluation on Test Set (with Cascading)")
    print("=" * 80)
    
    test_metrics = evaluate_path_metrics(
        stare_model, predictor, test_pos, name_to_id, id_to_name,
        text_features_ab=text_features_ab_test,
        text_features_bc=text_features_bc_test,
        use_text_features=use_text_features,
        device=device
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
                stare_model, predictor, val_pos, name_to_id, id_to_name,
                text_features_ab=text_features_ab_val,
                text_features_bc=text_features_bc_val,
                use_text_features=use_text_features,
                device=device
            )
            best_val_path_acc = best_val_metrics.get("path_acc", 0.0)
            best_epoch = epochs
    
    return {
        "stare_model": stare_model,
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

