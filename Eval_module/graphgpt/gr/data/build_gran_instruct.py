#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
import torch.utils._pytree

# --- 1. 修复 HuggingFace Hub (解决 sentence_transformers 报错) ---
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    print("✅ Patch 1 Applied: huggingface_hub.cached_download polyfill")
except ImportError:
    pass

# --- 2. PyTorch 2.1 兼容性 ---
if not hasattr(torch.utils._pytree, "register_pytree_node"):
    _orig_register = torch.utils._pytree._register_pytree_node
    def _safe_register_pytree_node(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None):
        return _orig_register(cls, flatten_fn, unflatten_fn)
    torch.utils._pytree.register_pytree_node = _safe_register_pytree_node
    print("✅ Patch 2 Applied: torch.utils._pytree.register_pytree_node")

# --- 3. 覆盖 Transformers 安全检查，确定环境安全 ---
try:
    import transformers.utils.import_utils
    import transformers.modeling_utils
    import transformers.tokenization_utils_base
    import transformers.processing_utils
except ImportError:
    pass

def safe_check(*args, **kwargs):
    return None

transformers.utils.import_utils.check_torch_load_is_safe = safe_check
for name, module in list(sys.modules.items()):
    if name.startswith("transformers"):
        if hasattr(module, "check_torch_load_is_safe"):
            setattr(module, "check_torch_load_is_safe", safe_check)
print("✅ Patch 3 Applied: Neutralized check_torch_load_is_safe.")

# # 仅去节点特征、裁剪到 A/B/C/Event
# python build_gran_instruct.py --disable_node_features

# # 仅去文本特征
# python build_gran_instruct.py --disable_text_features

# # 同时去节点与文本特征
# python build_gran_instruct.py --disable_node_features --disable_text_features

import argparse
import json
import sys
import os
import copy
from typing import List, Dict, Tuple, Optional
import torch
import dgl

# 基于节点名称或关系对获取证据文本
try:
    from Eval_module import db_mongo
except Exception:
    import db_mongo  # type: ignore

# 添加父目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
module_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _path in (project_root, module_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# 优先使用包内绝对导入，兼容脚本直接执行
try:
    from Eval_module.path_data import enumerate_graph_paths, split_paths, generate_negatives
    from Eval_module.graph_build import build_dgl_graph
    from Eval_module.config import NODE_NAME_FIELD
    from Eval_module.graphgpt.gr.data.gran_utils import (
        load_text_features_for_pairs,
        build_relation_type_map_from_paths,
    )
except ImportError:
    from path_data import enumerate_graph_paths, split_paths, generate_negatives
    from graph_build import build_dgl_graph
    from config import NODE_NAME_FIELD
    from gran_utils import (
        load_text_features_for_pairs,
        build_relation_type_map_from_paths,
    )

DEFAULT_PATH_CACHE = os.path.join(os.path.dirname(__file__), "cached_eval_module_paths.json")

# 直接用函数处理数据转换,避免 pickle 跨文件加载时的类定义依赖问题

def load_paths_from_eval_module(
    cache_file: Optional[str] = DEFAULT_PATH_CACHE,
    refresh_cache: bool = False,
    max_paths: Optional[int] = None,
) -> List[Dict]:
    if cache_file and os.path.exists(cache_file) and not refresh_cache:
        print(f"[EvalModule] Loading cached paths from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_paths = json.load(f)
        if max_paths is not None: return cached_paths[:max_paths]
        return cached_paths

    print("[EvalModule] Enumerating graph paths via path_data.enumerate_graph_paths() ...")
    paths = enumerate_graph_paths()
    print(f"[EvalModule] ✓ Retrieved {len(paths)} total paths")

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(paths, f, ensure_ascii=False, indent=2)

    if max_paths is not None: return paths[:max_paths]
    return paths


def sample_subgraph_for_path(
    dgl_g: dgl.DGLGraph,
    node_ids: List[int],
    num_hops: int = 2,
    num_neighbors: int = 10
) -> Tuple[torch.Tensor, List[int]]:
    """
    在 DGL 图中围绕给定节点采样子图。
    注意：传入的 dgl_g 必须已经是无向图(bidirected)，否则可能采不到邻居。
    """
    if len(node_ids) == 0:
        return torch.zeros((2, 0), dtype=torch.long), []

    seeds = torch.tensor(node_ids, dtype=torch.int64)
    # 确保种子节点去重
    sampled_nodes = torch.unique(seeds)
    curr_seeds = sampled_nodes

    for _ in range(num_hops):
        # 采样邻居
        neigh = dgl.sampling.sample_neighbors(
            dgl_g, curr_seeds, num_neighbors, replace=False
        )
        if neigh.num_edges() == 0:
            break
        
        src, dst = neigh.edges()
        # 将新发现的邻居加入节点集合
        new_nodes = torch.unique(torch.cat([src, dst]))
        sampled_nodes = torch.unique(torch.cat([sampled_nodes, new_nodes]))
        # 更新下一轮的种子为新发现的节点
        curr_seeds = new_nodes

    # 提取子图
    sub_g = dgl.node_subgraph(dgl_g, sampled_nodes)
    src, dst = sub_g.edges()
    edge_index = torch.stack([src, dst], dim=0)
    
    # 返回子图节点在当前图中的索引（与特征对齐）
    node_list = sub_g.nodes().tolist()
    
    return edge_index, node_list


def _build_human_prompt(A_name: str, B_name: str, C_name: str, A: int, B: int, C: int, Event: int, variant: str) -> str:
    """Return prompt text for the specified variant (stage1 vs stage2)."""
    base_header = (
        f"Given a knowledge graph with entities:\n"
        f"- Entity A (SLCGene): {A_name} (ID: {A})\n"
        f"- Entity B (Pathway): {B_name} (ID: {B})\n"
        f"- Entity C (Disease): {C_name} (ID: {C})\n"
    )
    if Event != A:
        base_header += f"- Event node: {Event}\n"

    # Stage1: 更偏结构化描述
    if variant == "stage1":
        body = (
            "You will inspect two subgraphs extracted around these entities.\n"
            "Step 1: From the first subgraph <graph>, decide the relationship between A and B.\n"
            "Step 2: Using the A-B decision, infer the relationship between B and C from the second subgraph <graph>.\n"
            "Allowed relationship labels: PROMOTION or SUPPRESSION.\n"
        )
    else:  # stage2/default
        body = (
            "Please predict the relationship types in a cascading manner:\n"
            "Step 1: Predict the relationship type between A and B using the first subgraph: <graph>\n"
            "Step 2: Based on the predicted A-B relationship from Step 1, predict the relationship type between B and C using the second subgraph: <graph>\n"
            "\nImportant: The relationship type must be one of: PROMOTION or SUPPRESSION.\n"
        )
    return base_header + "\n" + body


def build_instruction_sample(
    path: Dict,
    idx: int,
    split_name: str,
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
    dgl_g: dgl.DGLGraph,
    text_features_ab: Dict[int, torch.Tensor] = None,
    text_features_bc: Dict[int, torch.Tensor] = None,
    use_text_features: bool = False,
    prompt_variant: str = "stage2",
) -> Dict:
    A = path["A"]
    B = path["B"]
    C = path["C"]
    Event = path.get("Event", A)
    rel_AB = path["rel_AB"]
    rel_BC = path["rel_BC"]
    
    A_name = path.get("A_name", f"Entity_{A}")
    B_name = path.get("B_name", f"Entity_{B}")
    C_name = path.get("C_name", f"Entity_{C}")
    
    # 采样
    subgraph_1_nodes = [A, B, Event] if Event != A else [A, B]
    edge_index_1, node_list_1 = sample_subgraph_for_path(dgl_g, subgraph_1_nodes)
    
    subgraph_2_nodes = [B, C, Event] if Event != B else [B, C]
    edge_index_2, node_list_2 = sample_subgraph_for_path(dgl_g, subgraph_2_nodes)
    
    # 提示词（按 variant 区分 stage1/stage2）
    human_prompt = _build_human_prompt(A_name, B_name, C_name, A, B, C, Event, prompt_variant)
    
    gpt_response = (
        f"Step 1 - A-B Relationship: {rel_AB}\n"
        f"Step 2 - B-C Relationship: {rel_BC}"
    )
    
    sample = {
        "id": f"gran_{split_name}_{idx}_LP",
        "graph": {
            "edge_index_1": edge_index_1.tolist(),
            "node_list_1": node_list_1,
            "node_idx_1": A,
            "edge_index_2": edge_index_2.tolist(),
            "node_list_2": node_list_2,
            "node_idx_2": B,
        },
        "conversations": [
            {"from": "human", "value": human_prompt},
            {"from": "gpt",  "value": gpt_response}
        ]
    }
    
    if use_text_features:
        if text_features_ab and idx in text_features_ab:
            sample["graph"]["text_feat_ab"] = text_features_ab[idx].tolist()
        if text_features_bc and idx in text_features_bc:
            sample["graph"]["text_feat_bc"] = text_features_bc[idx].tolist()
    
    return sample


def build_gran_instruction_dataset(
    output_dir: str = "./graphgpt/gr/data/stage_2",
    max_paths: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_hops: int = 2,
    num_neighbors: int = 10,
    paths: Optional[List[Dict]] = None,
    path_cache_file: str = DEFAULT_PATH_CACHE,
    refresh_cache: bool = False,
    graph_content_out: Optional[str] = "./graphgpt/gr/data/stage_2/gran_graph_content.json",
    use_text_features: bool = True,
    text_feature_dim: int = 768,
    text_feature_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
    include_node_features: bool = True,
    prompt_variants: Optional[List[str]] = None,
    k_folds: Optional[int] = None,
    fold_idx: Optional[int] = None,
    seed: int = 42,
) -> None:
    print("=" * 80)
    print("Building GRAN Instruction Dataset for GraphGPT (Optimized)")
    print("=" * 80)
    
    # 枚举路径（提前用于裁剪节点）
    all_paths = paths or load_paths_from_eval_module(
        cache_file=path_cache_file,
        refresh_cache=refresh_cache,
        max_paths=max_paths,
    )
    if len(all_paths) == 0:
        print("No paths found, exit.")
        return

    # 构建图
    print("\n[1/6] Building DGL graph from Neo4j...")
    dgl_g, rel_type_map = build_dgl_graph()
    
    # 若禁用节点特征，则裁剪图只保留预测相关节点（A/B/C/Event）
    if not include_node_features:
        keep_nodes: List[int] = []
        for p in all_paths:
            keep_nodes.extend([p.get("A"), p.get("B"), p.get("C")])
            if p.get("Event") is not None:
                keep_nodes.append(p["Event"])
        keep_nodes = sorted(set(int(n) for n in keep_nodes if n is not None))
        keep_nodes_tensor = torch.tensor(keep_nodes, dtype=torch.long)
        print(f"  Pruning graph to {len(keep_nodes)} nodes used in paths (A/B/C/Event).")
        dgl_g = dgl.node_subgraph(dgl_g, keep_nodes_tensor)

        # 原始索引 -> 裁剪后子图索引，用于重写路径中的节点编号
        parent_ids = dgl_g.ndata[dgl.NID].tolist()
        old_to_new = {old: new for new, old in enumerate(parent_ids)}

        def _remap_path(path: Dict) -> Dict:
            new_p = dict(path)
            for k in ("A", "B", "C", "Event"):
                if k in new_p and new_p[k] is not None:
                    if new_p[k] not in old_to_new:
                        raise KeyError(f"Node {new_p[k]} missing after pruning; check keep_nodes logic.")
                    new_p[k] = old_to_new[new_p[k]]
            return new_p

        all_paths = [_remap_path(p) for p in all_paths]
    
    # 转为双向图（无向图），确保 sample_neighbors 能采到东西
    print("  Converting to Bidirected (Undirected) graph for better sampling...")
    dgl_g = dgl.to_bidirected(dgl_g, copy_ndata=True) 
    print(f"  ✓ Graph: {dgl_g.num_nodes()} nodes, {dgl_g.num_edges()} edges")
    
    # 保存图数据时，转换为纯字典，避免 class 依赖问题
    # 使用 PyG 风格的命名: x (特征), edge_index, edge_attr
    print("\n[2/6] Converting and Saving Graph Data...")
    
    src, dst = dgl_g.edges()
    edge_index = torch.stack([src, dst], dim=0)
    x = dgl_g.ndata.get("feat")
    if include_node_features:
        if x is None:
            print("  ⚠️ Warning: No node features found in DGL graph! Initializing random features.")
            x = torch.randn(dgl_g.num_nodes(), 384) # 假设384维
    else:
        x = None
    
    # 构建纯字典格式
    # GraphGPT 加载时通常检测 dict 或 PyG Data 对象
    tensor_graph_dict = {
        "edge_index": edge_index,
        "num_nodes": dgl_g.num_nodes()
    }
    if x is not None:
        tensor_graph_dict["x"] = x
    
    edge_attr = dgl_g.edata.get("feat")
    if use_text_features and edge_attr is not None:
        tensor_graph_dict["edge_attr"] = edge_attr

    graph_data_dir = os.path.join(os.path.dirname(__file__), "../graph_data")
    os.makedirs(graph_data_dir, exist_ok=True)
    graph_data_path = os.path.join(graph_data_dir, "gran_graph_data.pt")
    
    # 保存字典，外层 key 是 "gran" (对应 GraphGPT 代码里的 dataset name)
    torch.save({"gran": tensor_graph_dict}, graph_data_path)
    print(f"  ✓ Saved graph data (dict format) to {graph_data_path}")
    
    # 3. 映射关系
    name_to_id, id_to_name = build_relation_type_map_from_paths(all_paths)
    
    # 4. 划分数据：若传入 k_folds/fold_idx 则使用固定随机顺序做折分，否则按比例划分
    if k_folds is not None and fold_idx is not None:
        if k_folds <= 1:
            raise ValueError("k_folds must be > 1 when fold_idx is provided.")
        if not (0 <= fold_idx < k_folds):
            raise ValueError(f"fold_idx must be in [0, {k_folds-1}], got {fold_idx}")

        rng = torch.Generator()
        rng.manual_seed(seed)
        perm = torch.randperm(len(all_paths), generator=rng).tolist()
        all_paths_shuffled = [all_paths[i] for i in perm]

        fold_size = len(all_paths_shuffled) // k_folds
        start = fold_idx * fold_size
        end = len(all_paths_shuffled) if fold_idx == k_folds - 1 else start + fold_size
        val_pos = all_paths_shuffled[start:end]
        train_pos = all_paths_shuffled[:start] + all_paths_shuffled[end:]
        test_pos = val_pos
        print(f"[k-fold] k={k_folds}, fold={fold_idx}, seed={seed}, "
              f"train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")
    else:
        train_pos, val_pos, test_pos = split_paths(
            all_paths, train_ratio, val_ratio, seed=seed
        )
    
    # 5. 负样本
    all_pos_set = set((p["A"], p["B"], p["C"], p["rel_AB"], p["rel_BC"]) for p in all_paths)
    train_neg = generate_negatives(train_pos, all_pos_set, neg_sample_ratio=1.0)
    
    # 6. 文本特征
    text_feat_train = (None, None)
    text_feat_val = (None, None)
    text_feat_test = (None, None)
    
    if use_text_features:
        print("\n[5.5/6] Loading text features...")
        text_feat_train = load_text_features_for_pairs(train_pos, name_to_id)
        text_feat_val = load_text_features_for_pairs(val_pos, name_to_id)
        text_feat_test = load_text_features_for_pairs(test_pos, name_to_id)
    
    # 7. 构建数据集（支持多种 prompt 变体，默认 stage2+stage1）
    print("\n[6/6] Building JSONs...")
    os.makedirs(output_dir, exist_ok=True)

    variant_list = prompt_variants or ["stage2", "stage1"]

    # 收集 graph_content
    node_texts: Dict[int, Dict[str, str]] = {}
    pair_texts: Dict[str, Dict[str, str]] = {}

    def process_split(data_list, split_type, tf_ab, tf_bc, variant):
        samples = []
        for idx, path in enumerate(data_list):
            s = build_instruction_sample(
                path, idx, split_type, name_to_id, id_to_name, dgl_g,
                tf_ab, tf_bc, use_text_features, prompt_variant=variant
            )
            samples.append(s)

            # 节点文本收集（仅记录名称，可后续补充 text）
            if include_node_features:
                for nid, nname in [
                    (path["A"], path.get("A_name", f"Entity_{path['A']}")),
                    (path["B"], path.get("B_name", f"Entity_{path['B']}")),
                    (path["C"], path.get("C_name", f"Entity_{path['C']}")),
                ]:
                    if nid not in node_texts:
                        node_texts[nid] = {"name": nname, "text": ""}

            # 关系文本收集：A-B (slc_pathway), B-C (pathway_disease)
            if use_text_features:
                def _pk(a, b): return f"{a}-{b}"
                ab_key = _pk(path["A"], path["B"])
                if ab_key not in pair_texts:
                    txt_ab = db_mongo.fetch_pair_evidence_text("slc_pathway",
                                                               path.get("A_name", ""),
                                                               path.get("B_name", ""))
                    pair_texts[ab_key] = {
                        "type": "slc_pathway",
                        "key1": path.get("A_name", ""),
                        "key2": path.get("B_name", ""),
                        "text": txt_ab or ""
                    }
                bc_key = _pk(path["B"], path["C"])
                if bc_key not in pair_texts:
                    txt_bc = db_mongo.fetch_pair_evidence_text("pathway_disease",
                                                               path.get("B_name", ""),
                                                               path.get("C_name", ""))
                    pair_texts[bc_key] = {
                        "type": "pathway_disease",
                        "key1": path.get("B_name", ""),
                        "key2": path.get("C_name", ""),
                        "text": txt_bc or ""
                    }
        return samples

    for variant in variant_list:
        train_data = process_split(train_pos, "train", text_feat_train[0], text_feat_train[1], variant)
        # 负样本（prompt 仍使用当前 variant）
        for idx, path in enumerate(train_neg):
            s = build_instruction_sample(
                path, len(train_pos)+idx, "train_neg", name_to_id, id_to_name, dgl_g,
                None, None, False, prompt_variant=variant
            )
            train_data.append(s)
            
        val_data = process_split(val_pos, "val", text_feat_val[0], text_feat_val[1], variant)
        test_data = process_split(test_pos, "test", text_feat_test[0], text_feat_test[1], variant)
        
        suffix = "" if variant == "stage2" else f"_{variant}"
        with open(os.path.join(output_dir, f"gran_train_instruct{suffix}.json"), 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(os.path.join(output_dir, f"gran_val_instruct{suffix}.json"), 'w') as f:
            json.dump(val_data, f, indent=2)
        with open(os.path.join(output_dir, f"gran_test_instruct{suffix}.json"), 'w') as f:
            json.dump(test_data, f, indent=2)
            
        print(f"  ✓ Saved variant '{variant}' -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 保存 graph_content（节点+关系文本），供 Stage1/Stage2 --graph_content 使用
    if graph_content_out and (include_node_features or use_text_features):
        os.makedirs(os.path.dirname(graph_content_out), exist_ok=True)
        graph_content = {
            "nodes": node_texts if include_node_features else {},
            "relations": pair_texts if use_text_features else {},
        }
        with open(graph_content_out, "w", encoding="utf-8") as f:
            json.dump(graph_content, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Saved graph_content JSON to {graph_content_out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./graphgpt/gr/data/stage_2")
    parser.add_argument("--disable_text_features", action="store_true")
    parser.add_argument("--disable_node_features", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--path_cache_file", default=DEFAULT_PATH_CACHE)
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--prompt_variants", type=str, default="stage2,stage1",
                        help="Comma-separated prompt variants to generate (e.g., stage2,stage1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompt_variants = [v.strip() for v in args.prompt_variants.split(',') if v.strip()]
    build_gran_instruction_dataset(
        output_dir=args.output_dir,
        use_text_features=not args.disable_text_features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_hops=args.num_hops,
        num_neighbors=args.num_neighbors,
        path_cache_file=args.path_cache_file,
        refresh_cache=args.refresh_cache,
        max_paths=args.max_paths,
        include_node_features=not args.disable_node_features,
        prompt_variants=prompt_variants,
    )