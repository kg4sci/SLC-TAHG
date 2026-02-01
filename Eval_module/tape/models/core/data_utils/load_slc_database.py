"""
Loader for SLC_database:
- Neo4j: 节点/边及其属性（实时拉取，用于结构与标签）
- 预生成摘要：`data/precomputed_summaries/*.jsonl` 提供 pair-level 证据

返回 PyG Data + 文本列表：
- data.x: None（由 LM 生成后再作为特征使用）
- data.edge_index: [2, E]
- data.y: 节点标签（用节点的首个 label 编码成 id）
- train/val/test_mask: 按 0.7 / 0.15 / 0.15 随机划分
- raw_texts: 节点文本列表（长度 = 节点数），供 LMTrainer 使用
"""

import importlib
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional

class SimpleData:
    """
    仅用于本项目的轻量级 Data 容器，避免依赖 torch_geometric.Data。
    只要属性名兼容 (x, edge_index, y, train_mask, val_mask, test_mask, num_nodes)，
    其余代码（如 CustomDGLDataset、LMTrainer、GNNTrainer）即可正常工作。

    同时提供一个简化版 `.to(device)`，以便与 PyG 的 Data 接口保持兼容：
    - 对所有 tensor 属性调用 `.to(device)`
    - 其余属性保持不变
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        """Return a new SimpleData whose tensor attributes are moved to `device`."""
        moved = SimpleData()
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(moved, k, v.to(device))
            else:
                setattr(moved, k, v)
        return moved

try:
    from db_neo4j import load_neo4j_graph  # type: ignore[import]
except ImportError:
    _neo4j_module = None
    _neo4j_module_candidates = []

    if __package__:
        _parent = __package__
        while _parent:
            _neo4j_module_candidates.append(f"{_parent}.db_neo4j")
            if "." not in _parent:
                break
            _parent = _parent.rsplit(".", 1)[0]

    _neo4j_module_candidates.extend([
        "Eval_module.db_neo4j",
        "benchmark_paper.Eval_module.db_neo4j",
    ])

    for _module_name in _neo4j_module_candidates:
        try:
            _neo4j_module = importlib.import_module(_module_name)
            break
        except ModuleNotFoundError:
            continue

    if _neo4j_module is None:
        raise

    load_neo4j_graph = getattr(_neo4j_module, "load_neo4j_graph")

try:
    from local_summary_loader import get_pair_evidence_text
except ImportError:
    _summary_module_candidates = [
        "Eval_module.local_summary_loader",
        "benchmark_paper.Eval_module.local_summary_loader",
    ]
    if __package__:
        _summary_module_candidates.insert(0, f"{__package__}.local_summary_loader")

    _summary_module = None
    for _module_name in _summary_module_candidates:
        try:
            _summary_module = __import__(_module_name, fromlist=["get_pair_evidence_text"])
            break
        except ModuleNotFoundError:
            continue

    if _summary_module is None:
        raise

    get_pair_evidence_text = getattr(_summary_module, "get_pair_evidence_text")


def _split_masks(n: int, train_ratio=0.7, val_ratio=0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _apply_few_shot_mask(
    train_mask: torch.Tensor,
    y: torch.Tensor,
    k: int,
    seed: int = 0,
    balance_by: Optional[str] = None,
) -> torch.Tensor:
    """
    在已有的 train_mask 上执行 K-shot 截断（仅影响训练节点，保持全图）。
    balance_by 可选 "label"：按节点标签分层。
    """
    idx = torch.nonzero(train_mask, as_tuple=False).view(-1).cpu().numpy()
    if k is None or k <= 0 or k >= len(idx):
        return train_mask

    rng = np.random.default_rng(seed)

    if balance_by == "label":
        buckets = {}
        for i in idx:
            label = int(y[i].item()) if y.numel() > i else -1
            buckets.setdefault(label, []).append(i)
        sampled = []
        per_bucket = max(1, k // max(1, len(buckets)))
        for _, arr in buckets.items():
            take = min(per_bucket, len(arr))
            sampled.extend(rng.choice(arr, size=take, replace=False))
        if len(sampled) < k:
            remaining = [i for i in idx if i not in sampled]
            extra = rng.choice(remaining, size=min(k - len(sampled), len(remaining)), replace=False)
            sampled.extend(extra)
        selected = np.array(sampled[:k], dtype=np.int64)
    else:
        selected = rng.choice(idx, size=k, replace=False)

    new_mask = torch.zeros_like(train_mask)
    new_mask[selected] = True
    return new_mask


def _collect_pair_texts_per_event(
    nodes: List[Dict], rels: List[Dict], node_id_map: Dict[int, int]
) -> Tuple[List[str], List[str]]:
    """
    按照 rel_AB / rel_BC 分别聚合证据文本，并仅分配给事件节点 (RelaEvent)：
    - rel_AB：SLC-Pathway 证据，聚合到事件节点的 ab 文本槽位
    - rel_BC：Pathway-Disease 证据，聚合到事件节点的 bc 文本槽位
    实体节点对应位置为空串。返回 (ab_texts, bc_texts)，长度均为 num_nodes。
    """
    id_to_label = {n["id"]: (n.get("labels", []) or [""])[0] for n in nodes}
    id_to_name = {n["id"]: n.get("name", "") for n in nodes}

    ab_acc: List[List[str]] = [[] for _ in range(len(nodes))]
    bc_acc: List[List[str]] = [[] for _ in range(len(nodes))]

    for r in rels:
        h_id, t_id = r["head_id"], r["tail_id"]
        h_label = id_to_label.get(h_id, "")
        t_label = id_to_label.get(t_id, "")
        rel_type = r.get("relation_type", "")

        if h_label != "RelaEvent" and t_label != "RelaEvent":
            continue

        event_id = h_id if h_label == "RelaEvent" else t_id
        other_id = t_id if h_label == "RelaEvent" else h_id
        other_label = id_to_label.get(other_id, "")
        other_name = id_to_name.get(other_id, "")

        connected = []
        for rr in rels:
            if rr["head_id"] == event_id or rr["tail_id"] == event_id:
                ent_id = rr["tail_id"] if rr["head_id"] == event_id else rr["head_id"]
                connected.append(ent_id)

        edge_ab_text = ""
        edge_bc_text = ""

        if rel_type in ["IS_SOURCE", "IS_MEDIATOR"]:
            slc_name, pw_name = None, None
            if other_label == "SLCGene":
                slc_name = other_name
            elif other_label == "Pathway":
                pw_name = other_name
            for ent_id in connected:
                ent_label = id_to_label.get(ent_id, "")
                ent_name = id_to_name.get(ent_id, "")
                if ent_label == "SLCGene" and slc_name is None:
                    slc_name = ent_name
                if ent_label == "Pathway" and pw_name is None:
                    pw_name = ent_name
            if slc_name and pw_name:
                edge_ab_text = get_pair_evidence_text(
                    "slc_pathway", slc_name, pw_name, default=""
                )

        elif rel_type == "IS_TARGET":
            pw_name, dz_name = None, None
            if other_label == "Pathway":
                pw_name = other_name
            elif other_label == "Disease":
                dz_name = other_name
            for ent_id in connected:
                ent_label = id_to_label.get(ent_id, "")
                ent_name = id_to_name.get(ent_id, "")
                if ent_label == "Pathway" and pw_name is None:
                    pw_name = ent_name
                if ent_label == "Disease" and dz_name is None:
                    dz_name = ent_name
            if pw_name and dz_name:
                edge_bc_text = get_pair_evidence_text(
                    "pathway_disease", pw_name, dz_name, default=""
                )

        if edge_ab_text and event_id in node_id_map:
            ab_acc[node_id_map[event_id]].append(edge_ab_text)
        if edge_bc_text and event_id in node_id_map:
            bc_acc[node_id_map[event_id]].append(edge_bc_text)

    ab_texts = [" ".join(txts) for txts in ab_acc]
    bc_texts = [" ".join(txts) for txts in bc_acc]
    return ab_texts, bc_texts


def get_raw_text_slc_database(
    use_text: bool = True,
    seed: int = 0,
    few_shot_k: Optional[int] = None,
    few_shot_balance: Optional[str] = None,
) -> Tuple[SimpleData, List[str]]:
    """
    use_text=False: 返回 (data, None)，仅含图结构与节点标签
    use_text=True:  返回 (data, text_list)，其中 text_list 既包含节点自身属性合并的文本，
                   也包含与其相关的预生成 pair-level 证据文本（按节点聚合）
    """
    nodes, rels = load_neo4j_graph()  # live query

    # map node id -> contiguous idx
    node_id_map = {n["id"]: i for i, n in enumerate(nodes)}

    # labels to id
    label_set = []
    for n in nodes:
        labels = n.get("labels", [])
        if labels:
            label_set.append(labels[0])
    label_vocab = {name: i for i, name in enumerate(sorted(set(label_set)))}

    y = torch.tensor([label_vocab.get(n.get("labels", [None])[0], 0) for n in nodes], dtype=torch.long)

    # edge_index
    src = [node_id_map[r["head_id"]] for r in rels]
    dst = [node_id_map[r["tail_id"]] for r in rels]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    n_nodes = len(nodes)
    train_mask, val_mask, test_mask = _split_masks(n_nodes, seed=seed)
    if few_shot_k:
        train_mask = _apply_few_shot_mask(train_mask, y, few_shot_k, seed=seed, balance_by=few_shot_balance)

    data = SimpleData(
        x=None,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=n_nodes,
    )

    text_list = None
    if use_text:
        # 基础节点属性文本（已在 db_neo4j 合并，避免 rel_AB/rel_BC 泄露）
        base_text = [n.get("text", "") or "" for n in nodes]

        # 节点 id -> 名称 / 标签
        id_to_label = {n["id"]: (n.get("labels", []) or [""])[0] for n in nodes}
        id_to_name = {n["id"]: n.get("name", "") for n in nodes}

        # 构造以 RelaEvent 为中心的三元文本：搜集其直连的 SLCGene / Pathway / Disease 名称
        triplet_texts = []
        rel_ab_texts, rel_bc_texts = ["" for _ in range(n_nodes)], ["" for _ in range(n_nodes)]
        for n in nodes:
            nid = n["id"]
            label = id_to_label.get(nid, "")
            name = id_to_name.get(nid, "")

            if label != "RelaEvent":
                # 非事件节点：仍写入基础信息，方便索引对齐
                triplet_texts.append(
                    f"NodeLabel:{label}; Name:{name}; SLCs:[]; Pathways:[]; Diseases:[]; Attrs:{base_text[node_id_map[nid]]}"
                )
                continue

            slc_set, pw_set, dz_set = set(), set(), set()
            for r in rels:
                h, t = r["head_id"], r["tail_id"]
                hl, tl = id_to_label.get(h, ""), id_to_label.get(t, "")
                hn, tn = id_to_name.get(h, ""), id_to_name.get(t, "")

                if h == nid:
                    if tl == "SLCGene":
                        slc_set.add(tn)
                    elif tl == "Pathway":
                        pw_set.add(tn)
                    elif tl == "Disease":
                        dz_set.add(tn)
                if t == nid:
                    if hl == "SLCGene":
                        slc_set.add(hn)
                    elif hl == "Pathway":
                        pw_set.add(hn)
                    elif hl == "Disease":
                        dz_set.add(hn)

            triplet_texts.append(
                f"NodeLabel:{label}; Name:{name}; SLCs:{sorted(slc_set)}; Pathways:{sorted(pw_set)}; Diseases:{sorted(dz_set)}; Attrs:{base_text[node_id_map[nid]]}"
            )

        # 预生成证据文本：按 rel_AB / rel_BC 分开，只给事件节点
        ab_texts, bc_texts = _collect_pair_texts_per_event(nodes, rels, node_id_map)

        # 对 LM 训练的主文本，改为以 RelaEvent 为中心的三元描述
        text_list = triplet_texts
        # 同时把 AB/BC 证据挂到 data 上，供后续级联任务或自定义处理使用
        data.rel_ab_texts = ab_texts
        data.rel_bc_texts = bc_texts

    return data, text_list

