from typing import Dict, List, Tuple, Set
import torch
import os
os.environ["DGL_USE_GRAPHBOLT"] = "0"
import dgl
from sentence_transformers import SentenceTransformer

from .db_neo4j import load_neo4j_graph
from .db_mongo import fetch_pair_evidence_text, fetch_papers_by_dois
from .config import SBERT_MODEL_NAME, SBERT_DIM, LABEL_RELA_EVENT

os.environ['TRANSFORMERS_NO_TF'] = '1'

def build_dgl_graph() -> Tuple[dgl.DGLGraph, Dict[str, int]]:
    nodes, rels = load_neo4j_graph()

    # Map original Neo4j node id -> contiguous id
    node_id_map: Dict[int, int] = {n["id"]: idx for idx, n in enumerate(nodes)}
    # Map node id -> node name (for MongoDB queries)
    node_id_to_name: Dict[int, str] = {n["id"]: n.get("name", "") for n in nodes}
    # Map node id -> node labels
    node_id_to_labels: Dict[int, List[str]] = {n["id"]: n.get("labels", []) for n in nodes}

    # Create a mapping of node id -> node label for quick lookup
    node_id_to_label_single = {}
    for node in nodes:
        node_id = node["id"]
        labels = node.get("labels", [])
        # Pick the first meaningful label
        if labels:
            node_id_to_label_single[node_id] = labels[0]

    # Create a mapping of RelaEvent id -> list of connected entity nodes
    # This helps us find the SLC and Pathway nodes for rel_AB, Pathway and Disease for rel_BC
    event_id_to_entities = {}
    for r in rels:
        head_id = r["head_id"]
        tail_id = r["tail_id"]
        rel_type = r.get("relation_type", "")
        
        head_label = node_id_to_label_single.get(head_id, "")
        tail_label = node_id_to_label_single.get(tail_id, "")
        
        # Track connections to RelaEvent nodes
        if tail_label == "RelaEvent":
            event_id = tail_id
            entity_id = head_id
            if event_id not in event_id_to_entities:
                event_id_to_entities[event_id] = []
            event_id_to_entities[event_id].append({
                "entity_id": entity_id,
                "entity_label": head_label,
                "entity_name": node_id_to_name.get(head_id, ""),
                "rel_type": rel_type
            })
        elif head_label == "RelaEvent":
            event_id = head_id
            entity_id = tail_id
            if event_id not in event_id_to_entities:
                event_id_to_entities[event_id] = []
            event_id_to_entities[event_id].append({
                "entity_id": entity_id,
                "entity_label": tail_label,
                "entity_name": node_id_to_name.get(tail_id, ""),
                "rel_type": rel_type
            })

    # Collect node and edge texts for encoding
    node_texts: List[str] = [n.get("text", "") for n in nodes]

    edge_texts: List[str] = []
    for r in rels:
        head_id = r["head_id"]
        tail_id = r["tail_id"]
        head_name = node_id_to_name.get(head_id, "")
        tail_name = node_id_to_name.get(tail_id, "")
        rel_type = r.get("relation_type", "")
        
        head_label = node_id_to_label_single.get(head_id, "")
        tail_label = node_id_to_label_single.get(tail_id, "")
        
        edge_text = ""
        
        # Handle edges connected to RelaEvent nodes
        if head_label == "RelaEvent" or tail_label == "RelaEvent":
            event_id = head_id if head_label == "RelaEvent" else tail_id
            entities_info = event_id_to_entities.get(event_id, [])
            
            # Determine which pair type based on the edge relation type
            if rel_type in ["IS_SOURCE", "IS_MEDIATOR"]:
                # This is for rel_AB (SLC-Pathway pair)
                # Find SLC and Pathway entities connected to this event
                slc_name = None
                pathway_name = None
                for entity_info in entities_info:
                    if entity_info["entity_label"] == "SLCGene":
                        slc_name = entity_info["entity_name"]
                    elif entity_info["entity_label"] == "Pathway":
                        pathway_name = entity_info["entity_name"]
                
                if slc_name and pathway_name:
                    edge_text = fetch_pair_evidence_text("slc_pathway", slc_name, pathway_name)
            
            elif rel_type == "IS_TARGET":
                # This is for rel_BC (Pathway-Disease pair)
                # Find Pathway and Disease entities connected to this event
                pathway_name = None
                disease_name = None
                for entity_info in entities_info:
                    if entity_info["entity_label"] == "Pathway":
                        pathway_name = entity_info["entity_name"]
                    elif entity_info["entity_label"] == "Disease":
                        disease_name = entity_info["entity_name"]
                
                if pathway_name and disease_name:
                    edge_text = fetch_pair_evidence_text("pathway_disease", pathway_name, disease_name)
        
        elif head_label in ["SLCGene", "Pathway", "Disease"] and tail_label in ["SLCGene", "Pathway", "Disease"]:
            # Direct entity-to-entity edge (fallback, shouldn't happen in N-ary model)
            pair_type = None
            if head_label == "SLCGene" and tail_label == "Pathway":
                pair_type = "slc_pathway"
            elif head_label == "Pathway" and tail_label == "Disease":
                pair_type = "pathway_disease"
            
            if pair_type:
                edge_text = fetch_pair_evidence_text(pair_type, head_name, tail_name)
        
        # Fallback: use DOI list if pair lookup failed or labels don't match
        if not edge_text and r.get("doi_list"):
            edge_text = fetch_papers_by_dois(r.get("doi_list", []))
        
        edge_texts.append(edge_text)

    try:
        encoder = SentenceTransformer(SBERT_MODEL_NAME)
    except ValueError as err:
        print(f"[graph_build] Failed to load SentenceTransformer due to ValueError: {err}. "
              "Falling back to zero embeddings. Upgrade torch>=2.6 or install safetensors checkpoints for full features.")
        encoder = None
    except Exception as err:
        print(f"[graph_build] Warning: SentenceTransformer initialization failed ({type(err).__name__}: {err}). "
              "Falling back to zero embeddings.")
        encoder = None

    all_texts = node_texts + edge_texts
    if len(all_texts) == 0:
        raise ValueError("No texts found to encode for nodes/edges.")

    if encoder is not None:
        features = encoder.encode(all_texts, show_progress_bar=True)
        features_tensor = torch.tensor(features, dtype=torch.float)
    else:
        features_tensor = torch.zeros((len(all_texts), SBERT_DIM), dtype=torch.float)

    node_features = features_tensor[: len(nodes)]
    edge_features = features_tensor[len(nodes) :]

    # Relation type mapping
    rel_types_unique = sorted(list({r["relation_type"] for r in rels}))
    rel_type_map = {name: i for i, name in enumerate(rel_types_unique)}

    # Build edges
    src: List[int] = []
    dst: List[int] = []
    rel_type_ids: List[int] = []
    for r in rels:
        src.append(node_id_map[r["head_id"]])
        dst.append(node_id_map[r["tail_id"]])
        rel_type_ids.append(rel_type_map[r["relation_type"]])

    g = dgl.graph(
        (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
        ),
        num_nodes=len(nodes),
    )
    g.ndata["feat"] = node_features
    g.edata["feat"] = edge_features
    g.edata["rel_type"] = torch.tensor(rel_type_ids, dtype=torch.long)
    return g, rel_type_map


