import os
import torch
from typing import List, Dict, Tuple, Optional

# 仅依赖 Eval_module 根目录下的通用数据代码（例如 path_data.py），
# 不导入 HypE 子包中的任何模型或训练逻辑。


def _enumerate_paths_from_graph() -> List[Dict]:
    """
    使用 Eval_module 根目录下的 path_data.enumerate_graph_paths
    从 Neo4j 图中枚举 N 元关系路径。

    返回的每个元素是一个 dict，通常包含字段：
    - A, B, C, Event : int 节点 id
    - rel_AB, rel_BC : 级联预测的两个关系标签
    - 可选 A_name, B_name, C_name 等
    """
    from path_data import enumerate_graph_paths  # type: ignore
    return enumerate_graph_paths()


def _split_paths(
    all_paths: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: Optional[int] = None,
):
    """复用 Eval_module 根目录下 path_data.split_paths 做划分。"""
    from path_data import split_paths  # type: ignore
    return split_paths(all_paths, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


def _build_relation_maps(paths: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """根据 paths 中出现的 rel_AB / rel_BC 构建 label←→id 双向映射。"""
    rel_types = set()
    for p in paths:
        if "rel_AB" in p and p["rel_AB"] is not None:
            rel_types.add(p["rel_AB"])
        if "rel_BC" in p and p["rel_BC"] is not None:
            rel_types.add(p["rel_BC"])
    sorted_types = sorted(list(rel_types))
    name_to_id = {n: i for i, n in enumerate(sorted_types)}
    id_to_name = {i: n for i, n in enumerate(sorted_types)}
    return name_to_id, id_to_name


def load_hype_cascading_paths(
    cache_file: str = "hype_paths.pt",
    use_cache: bool = True,
    force_refresh: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict:
    """
    Load cascading (AB->BC) path data prepared for HypE and expose to TAPE.

    Args:
        cache_file: location of serialized list of paths. When absent and use_cache is True,
                    we will enumerate via HypE and then save.
        use_cache: whether to try loading the cached list first.
        force_refresh: ignore cache and re-enumerate.
        train_ratio / val_ratio: split ratios; test is the remainder.

    Returns:
        {
            "train": List[Dict],
            "val": List[Dict],
            "test": List[Dict],
            "name_to_id": Dict[str, int],
            "id_to_name": Dict[int, str],
        }
    """
    paths: Optional[List[Dict]] = None

    if use_cache and not force_refresh and os.path.exists(cache_file):
        try:
            paths = torch.load(cache_file)
            print(f"[HypE->TAPE] Loaded cached paths from {cache_file} ({len(paths)} samples)")
        except Exception as e:
            print(f"[HypE->TAPE] Failed to load cache {cache_file}: {e}")

    if paths is None:
        print("[HypE->TAPE] Enumerating paths from Neo4j graph (Eval_module.path_data) ...")
        paths = _enumerate_paths_from_graph()
        print(f"[HypE->TAPE] Enumerated {len(paths)} paths")
        if use_cache or force_refresh:
            try:
                torch.save(paths, cache_file)
                print(f"[HypE->TAPE] Cached paths to {cache_file}")
            except Exception as e:
                print(f"[HypE->TAPE] Failed to save cache {cache_file}: {e}")

    if not paths:
        raise ValueError("[HypE->TAPE] No paths found; please check HypE preprocessing.")

    name_to_id, id_to_name = _build_relation_maps(paths)
    train_pos, val_pos, test_pos = _split_paths(
        paths, train_ratio=train_ratio, val_ratio=val_ratio, seed=None
    )

    return {
        "train": train_pos,
        "val": val_pos,
        "test": test_pos,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
    }


