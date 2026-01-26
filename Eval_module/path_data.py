import os
from importlib import import_module
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
from neo4j import GraphDatabase

try:
    from config import (  # type: ignore[attr-defined]
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        LABEL_SLC,
        LABEL_PATHWAY,
        LABEL_DISEASE,
        LABEL_RELA_EVENT,
    )
except ImportError:
    module_candidates = []
    if __package__:
        module_candidates.append(f"{__package__}.config")
    module_candidates.extend([
        "Eval_module.config",
        "benchmark_paper.Eval_module.config",
    ])

    config_module = None
    for module_name in module_candidates:
        try:
            config_module = import_module(module_name)
            break
        except ModuleNotFoundError:
            continue

    if config_module is None:
        raise

    NEO4J_URI = getattr(config_module, "NEO4J_URI", os.environ.get("NEO4J_URI", "bolt://localhost:7690"))
    NEO4J_USER = getattr(config_module, "NEO4J_USER", os.environ.get("NEO4J_USER", "neo4j"))
    NEO4J_PASSWORD = getattr(config_module, "NEO4J_PASSWORD", os.environ.get("NEO4J_PASSWORD", "neo4j"))
    LABEL_SLC = getattr(config_module, "LABEL_SLC", "SLCGene")
    LABEL_PATHWAY = getattr(config_module, "LABEL_PATHWAY", "Pathway")
    LABEL_DISEASE = getattr(config_module, "LABEL_DISEASE", "Disease")
    LABEL_RELA_EVENT = getattr(config_module, "LABEL_RELA_EVENT", "RelaEvent")


def split_paths(
    paths: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: Optional[int] = None,
):
    """
    将路径集合划分为train/val/test，支持可选随机种子以便结果可复现。
    """
    idx = np.arange(len(paths))
    if seed is None:
        np.random.shuffle(idx)
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    train = [paths[i] for i in train_idx]
    val = [paths[i] for i in val_idx]
    test = [paths[i] for i in test_idx]
    return train, val, test


def select_few_shot(
    paths: List[Dict],
    k: int,
    seed: int = 0,
    balance_by: Optional[str] = None,
) -> List[Dict]:
    """
    从完整路径列表中抽取 K 条完整路径（A, Event, B, C 及 rel_AB/rel_BC），保证按“整条路径”为单位。

    Args:
        paths: 全部正样本路径列表。
        k: 需要抽取的路径数；k <= 0 或 k >= len(paths) 时返回原列表。
        seed: 随机种子，便于可复现。
        balance_by: 可选分层键，None/\"relation\"/\"disease\"/\"pathway\"。
            - relation: 按 (rel_AB, rel_BC) 分类，尽量每类都有样本。
            - disease: 按 C（或 C_name）分类。
            - pathway: 按 B（或 B_name）分类。

    Returns:
        选出的路径子集（长度不超过 k）。
    """
    if k <= 0 or k >= len(paths):
        return list(paths)

    rng = np.random.default_rng(seed)

    if balance_by is None:
        idx = rng.choice(len(paths), size=k, replace=False)
        return [paths[i] for i in idx]

    # 分层抽样
    buckets: Dict[object, List[int]] = {}
    for i, p in enumerate(paths):
        if balance_by == "relation":
            key = (p.get("rel_AB"), p.get("rel_BC"))
        elif balance_by == "disease":
            key = p.get("C") if "C" in p else p.get("C_name")
        elif balance_by == "pathway":
            key = p.get("B") if "B" in p else p.get("B_name")
        else:
            key = None
        buckets.setdefault(key, []).append(i)

    sampled: List[int] = []
    per_bucket = max(1, k // max(1, len(buckets)))
    for _, idxs in buckets.items():
        take = min(per_bucket, len(idxs))
        sampled.extend(rng.choice(idxs, size=take, replace=False))

    if len(sampled) < k:
        remaining = [i for i in range(len(paths)) if i not in sampled]
        extra = rng.choice(remaining, size=min(k - len(sampled), len(remaining)), replace=False)
        sampled.extend(extra)

    return [paths[i] for i in sampled[:k]]


def generate_negatives(train_pos: List[Dict], all_pos_set: Set[Tuple[int,int,int,str,str]], neg_sample_ratio=1.0) -> List[Dict]:
    """
    Generate negative samples for the N-ary model.
    
    Since rel_AB and rel_BC are properties of RelaEvent nodes (not edge types),
    we generate negatives by:
    1. Swapping rel_AB with other observed rel_AB values
    2. Swapping rel_BC with other observed rel_BC values
    3. Randomly corrupting both if insufficient
    
    neg_sample_ratio: multiplier for number of negatives per positive (default 1.0 = 1:1 ratio)
    """
    negatives: List[Dict] = []
    
    # Collect all unique relation types observed in training data
    all_rel_ab_types = list(set(p["rel_AB"] for p in train_pos))
    all_rel_bc_types = list(set(p["rel_BC"] for p in train_pos))
    
    try:
        for p in train_pos:
            A, B, C = p["A"], p["B"], p["C"]
            true_ab, true_bc = p["rel_AB"], p["rel_BC"]

            # Strategy 1: Replace rel_AB with alternative types
            for alt_ab in all_rel_ab_types:
                if alt_ab == true_ab:
                    continue
                cand = (A, B, C, alt_ab, true_bc)
                if cand not in all_pos_set:
                    negatives.append({"A": A, "B": B, "C": C, "rel_AB": alt_ab, "rel_BC": true_bc})

            # Strategy 2: Replace rel_BC with alternative types
            for alt_bc in all_rel_bc_types:
                if alt_bc == true_bc:
                    continue
                cand = (A, B, C, true_ab, alt_bc)
                if cand not in all_pos_set:
                    negatives.append({"A": A, "B": B, "C": C, "rel_AB": true_ab, "rel_BC": alt_bc})
    finally:
        pass
    
    # Strategy 3: Random negative sampling if not enough
    neg_count_target = max(1, int(len(train_pos) * neg_sample_ratio))
    while len(negatives) < neg_count_target:
        idx = np.random.randint(0, len(train_pos))
        p = train_pos[idx]
        A, B, C = p["A"], p["B"], p["C"]
        true_ab, true_bc = p["rel_AB"], p["rel_BC"]
        
        # Randomly replace rel_AB
        if np.random.rand() > 0.5:
            rand_ab = np.random.choice(all_rel_ab_types)
            if rand_ab != true_ab:
                cand = (A, B, C, rand_ab, true_bc)
                if cand not in all_pos_set:
                    negatives.append({"A": A, "B": B, "C": C, "rel_AB": rand_ab, "rel_BC": true_bc})
                    continue
        
        # Randomly replace rel_BC
        rand_bc = np.random.choice(all_rel_bc_types)
        if rand_bc != true_bc:
            cand = (A, B, C, true_ab, rand_bc)
            if cand not in all_pos_set:
                negatives.append({"A": A, "B": B, "C": C, "rel_AB": true_ab, "rel_BC": rand_bc})
    
    print(f"[Negative Sampling] Generated {len(negatives)} negatives (target: {neg_count_target})")
    return negatives[:neg_count_target]  # Trim to target ratio


def enumerate_graph_paths() -> List[Dict]:
    """
    Enumerate paths in the N-ary relation model:
    SLCGene (A) --[IS_SOURCE]--> RelaEvent --[IS_TARGET]--> Disease (C)
                                     ^
                                     |[IS_MEDIATOR]
                                  Pathway (B)
    
    For each valid path, extract rel_AB and rel_BC from the RelaEvent node's properties.
    Returns: List[Dict] with A, A_name, B, B_name, C, C_name, rel_AB, rel_BC
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results: List[Dict] = []
    try:
        with driver.session() as session:
            # NEW QUERY: Find all paths through RelaEvent nodes
            # SLCGene --[IS_SOURCE]--> RelaEvent <--[IS_MEDIATOR]-- Pathway
            #                             ↑
            #                        [IS_TARGET]
            #                             |
            #                          Disease
            q_nary = (
                f"MATCH (a:`{LABEL_SLC}`)-[:IS_SOURCE]->(e:`{LABEL_RELA_EVENT}`), "
                f"(b:`{LABEL_PATHWAY}`)-[:IS_MEDIATOR]->(e:`{LABEL_RELA_EVENT}`), "
                f"(e:`{LABEL_RELA_EVENT}`)<-[:IS_TARGET]-(c:`{LABEL_DISEASE}`) "
                "RETURN id(a) as A, a.name as A_name, "
                "id(b) as B, b.name as B_name, "
                "id(c) as C, c.name as C_name, "
                "id(e) as Event, "
                "e.rel_AB as rel_AB, e.rel_BC as rel_BC"
            )
            recs = session.run(q_nary)
            for rec in recs:
                results.append({
                    "A": rec["A"], "A_name": rec.get("A_name"),
                    "B": rec["B"], "B_name": rec.get("B_name"),
                    "C": rec["C"], "C_name": rec.get("C_name"),
                    "Event": rec["Event"],
                    "rel_AB": rec["rel_AB"], "rel_BC": rec["rel_BC"],
                })
    finally:
        driver.close()
    return results


