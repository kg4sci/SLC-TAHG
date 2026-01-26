import os
from importlib import import_module
from typing import Dict, List, Tuple

from neo4j import GraphDatabase

try:
    from .config import (  # type: ignore[attr-defined]
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        REL_DOI_FIELD,
        NODE_NAME_FIELD,
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
    REL_DOI_FIELD = getattr(config_module, "REL_DOI_FIELD", "paper_dois")
    NODE_NAME_FIELD = getattr(config_module, "NODE_NAME_FIELD", "name")
    LABEL_RELA_EVENT = getattr(config_module, "LABEL_RELA_EVENT", "RelaEvent")


def _merge_text_properties(props: Dict, exclude_keys: List[str] = None) -> str:
    """
    Merge all string-like properties for a node into a single text blob.
    
    Args:
        props: Node properties dictionary
        exclude_keys: List of property keys to exclude (e.g., ['rel_AB', 'rel_BC'] to avoid data leakage)
    """
    if not props:
        return ""
    if exclude_keys is None:
        exclude_keys = []
    texts: List[str] = []
    for key, value in props.items():
        # 排除指定的键（避免数据泄露）
        if key in exclude_keys:
            continue
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend([str(v) for v in value if isinstance(v, (str, int, float))])
        elif isinstance(value, (int, float)):
            continue
    return " ".join(texts)


def load_neo4j_graph() -> Tuple[List[Dict], List[Dict]]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            nodes_query = (
                "MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as props"
            )
            nodes_result = session.run(nodes_query).data()

            rels_query = (
                "MATCH (h)-[r]->(t) RETURN id(h) as head_id, type(r) as relation_type, "
                "id(t) as tail_id, id(r) as rel_id, r.{} as paper_dois".format(REL_DOI_FIELD)
            )
            relations_result = session.run(rels_query).data()
    finally:
        driver.close()

    # Preprocess: normalize node text and relation DOI field; include node names for MongoDB lookup
    for node in nodes_result:
        props = node.get("props", {})
        labels = node.get("labels", [])
        
        # 对于RelaEvent节点，排除rel_AB和rel_BC属性（避免数据泄露）
        exclude_keys = []
        if LABEL_RELA_EVENT in labels:
            exclude_keys = ["rel_AB", "rel_BC"]
        
        node["text"] = _merge_text_properties(props, exclude_keys=exclude_keys)
        # Extract node name from properties for MongoDB queries
        node["name"] = props.get(NODE_NAME_FIELD, "")

    for rel in relations_result:
        dois_raw = rel.get("paper_dois")
        if isinstance(dois_raw, str):
            parts = [p.strip() for p in dois_raw.replace("|", ";").split(";")]
            rel["doi_list"] = [p for p in parts if p]
        elif isinstance(dois_raw, list):
            rel["doi_list"] = [str(p).strip() for p in dois_raw if str(p).strip()]
        else:
            rel["doi_list"] = []

    return nodes_result, relations_result


