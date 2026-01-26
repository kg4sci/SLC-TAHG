import os
from importlib import import_module
from typing import List, Optional, Tuple

from pymongo import MongoClient

try:
    from .config import (  # type: ignore[attr-defined]
        MONGO_URI,
        MONGO_DB_NAME,
        MONGO_COLLECTION_PAPERS,
        MONGO_COLLECTION_PW_DZ,
        MONGO_COLLECTION_SLC_PW,
        MONGO_FIELD_SLC,
        MONGO_FIELD_PATHWAY,
        MONGO_FIELD_DISEASE,
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

    MONGO_URI = getattr(config_module, "MONGO_URI", os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
    MONGO_DB_NAME = getattr(config_module, "MONGO_DB_NAME", os.environ.get("MONGO_DB_NAME", "SLCdb"))
    MONGO_COLLECTION_PAPERS = getattr(config_module, "MONGO_COLLECTION_PAPERS", "papers")
    MONGO_COLLECTION_PW_DZ = getattr(config_module, "MONGO_COLLECTION_PW_DZ", "pathway_disease")
    MONGO_COLLECTION_SLC_PW = getattr(config_module, "MONGO_COLLECTION_SLC_PW", "slc_pathway")
    MONGO_FIELD_SLC = getattr(config_module, "MONGO_FIELD_SLC", "slc")
    MONGO_FIELD_PATHWAY = getattr(config_module, "MONGO_FIELD_PATHWAY", "pathway")
    MONGO_FIELD_DISEASE = getattr(config_module, "MONGO_FIELD_DISEASE", "disease")

try:
    from .local_summary_loader import get_pair_evidence_text  # type: ignore[attr-defined]
except ImportError:
    summary_candidates = []
    if __package__:
        summary_candidates.append(f"{__package__}.local_summary_loader")
    summary_candidates.extend([
        "Eval_module.local_summary_loader",
        "benchmark_paper.Eval_module.local_summary_loader",
    ])

    summary_module = None
    for module_name in summary_candidates:
        try:
            summary_module = import_module(module_name)
            break
        except ModuleNotFoundError:
            continue

    if summary_module is None:
        raise

    get_pair_evidence_text = getattr(summary_module, "get_pair_evidence_text")
#如果想使用原版本实时链接，则运行设置环境export USE_MONGO_LIVE=1

def _norm_doi(doi: str) -> str:
    if not doi:
        return ""
    d = doi.strip().lower()
    if d.startswith("https://") or d.startswith("http://"):
        parts = d.split("doi.org/")
        d = parts[-1] if len(parts) > 1 else d.split("/", 3)[-1]
    d = d.replace(" ", "")
    return d


def fetch_papers_by_dois(dois: List[str]) -> str:
    """Query papers collection by DOI list, return concatenated title + abstract."""
    if not dois:
        return ""
    normed = list({_norm_doi(x) for x in dois if _norm_doi(x)})
    if not normed:
        return ""

    client = MongoClient(MONGO_URI)
    try:
        db = client[MONGO_DB_NAME]
        col = db[MONGO_COLLECTION_PAPERS]
        cursor = col.find(
            {"$or": [
                {"doi": {"$in": normed}},
                {"normalized_doi": {"$in": normed}},
            ]},
            {"_id": 0, "title": 1, "abstract": 1}
        )
        texts: List[str] = []
        doc_count = 0
        for doc in cursor:
            doc_count += 1
            if isinstance(doc.get("title"), str) and doc["title"].strip():
                texts.append(doc["title"])
            if isinstance(doc.get("abstract"), str) and doc["abstract"].strip():
                texts.append(doc["abstract"])
        result = " ".join(texts)
        #是否获取到mongoDB数据
        # if doc_count % 100 == 0:
        #     print(f"[MongoDB] √fetch_papers_by_dois: found {doc_count} papers, returned {len(texts)} text segments ({len(result)} chars)")
        # else:
        #     print(f"[MongoDB] fetch_papers_by_dois: no papers found for DOIs: {normed}")
        return result
    except Exception as e:
        print(f"[MongoDB]  fetch_papers_by_dois ERROR: {e}")
        return ""
    finally:
        client.close()


def fetch_pair_evidence_text(pair_type: str, key1: str, key2: str) -> str:
    """Fetch evidence texts for a node pair.
    
    pair_type: "slc_pathway" or "pathway_disease"
    key1, key2: node names (e.g., slc_name and pathway_name for slc_pathway)
    
    Flow:
    1. Query the pair collection (slc_pathway or pathway_disease) using both keys.
    2. Extract paper_dois from the query result.
    3. Query papers collection using the DOI list.
    4. Return concatenated title + abstract.
    """
    #使用提取的精简版text
    local_text = get_pair_evidence_text(pair_type, key1, key2, default="")
    if local_text:
        return local_text

    if os.environ.get("USE_MONGO_LIVE", "0") != "1":
        return ""

    client = MongoClient(MONGO_URI)
    try:
        db = client[MONGO_DB_NAME]
        
        # Determine collection and field names based on pair_type
        if pair_type == "slc_pathway":
            pair_col = db[MONGO_COLLECTION_SLC_PW]
            query = {MONGO_FIELD_SLC: key1, MONGO_FIELD_PATHWAY: key2}
            # print(f"[MongoDB] query: {query}")
        elif pair_type == "pathway_disease":
            pair_col = db[MONGO_COLLECTION_PW_DZ]
            query = {MONGO_FIELD_PATHWAY: key1, MONGO_FIELD_DISEASE: key2}
        else:
            # print(f"[MongoDB] ✗ fetch_pair_evidence_text: unknown pair_type '{pair_type}'")
            return ""
        
        # Query pair collection for paper_dois
        rec = pair_col.find_one(query, {"paper_dois": 1, "_id": 0})
        if not rec:
            # print(f"[MongoDB] ✗ fetch_pair_evidence_text ({pair_type}): no record for {{{MONGO_FIELD_SLC if pair_type=='slc_pathway' else MONGO_FIELD_PATHWAY}: '{key1}', {MONGO_FIELD_PATHWAY if pair_type=='slc_pathway' else MONGO_FIELD_DISEASE}: '{key2}'}}")
            return ""
        
        if not rec.get("paper_dois"):
            # print(f"[MongoDB] ⚠ fetch_pair_evidence_text ({pair_type}): record found but no 'paper_dois' for {{{MONGO_FIELD_SLC if pair_type=='slc_pathway' else MONGO_FIELD_PATHWAY}: '{key1}', {MONGO_FIELD_PATHWAY if pair_type=='slc_pathway' else MONGO_FIELD_DISEASE}: '{key2}'}}")
            return ""
        
        # Extract DOI list (handle both string and list formats)
        dois_raw = rec.get("paper_dois", [])
        if isinstance(dois_raw, str):
            dois = [d.strip() for d in dois_raw.replace("|", ";").split(";") if d.strip()]
        elif isinstance(dois_raw, list):
            dois = [str(d).strip() for d in dois_raw if str(d).strip()]
        else:
            dois = []
        
        if not dois:
            # print(f"[MongoDB] ⚠ fetch_pair_evidence_text ({pair_type}): no valid DOIs extracted for {{{MONGO_FIELD_SLC if pair_type=='slc_pathway' else MONGO_FIELD_PATHWAY}: '{key1}', {MONGO_FIELD_PATHWAY if pair_type=='slc_pathway' else MONGO_FIELD_DISEASE}: '{key2}'}}")
            return ""
        
        # print(f"[MongoDB] ✓ fetch_pair_evidence_text ({pair_type}): {len(dois)} DOIs for {{{MONGO_FIELD_SLC if pair_type=='slc_pathway' else MONGO_FIELD_PATHWAY}: '{key1}', {MONGO_FIELD_PATHWAY if pair_type=='slc_pathway' else MONGO_FIELD_DISEASE}: '{key2}'}} -> {dois[:5]}{'...' if len(dois) > 5 else ''}")
        
        # Query papers collection using DOI list
        text = fetch_papers_by_dois(dois)
        # print(f"[MongoDB] ✓ fetch_pair_evidence_text ({pair_type}): retrieved {len(text)} chars of text")
        return text
    except Exception as e:
        print(f"[MongoDB] ✗ fetch_pair_evidence_text ({pair_type}) ERROR: {e}")
        return ""
    finally:
        client.close()


# Optional caching utilities
_embedding_cache = {}


def get_pair_evidence_embedding(pair_type: str, key1: str, key2: str, *,
                                encoder=None, reencode: bool = True) -> Tuple[Optional[str], Optional[List[float]]]:
    """Return (raw_text, embedding) for a pair.
    
    - If reencode=False and cached, returns cached embedding.
    - If reencode=True or not cached, fetch text and encode with provided encoder.
    - encoder: a SentenceTransformer instance; if None, a new one will be created.
    """
    cache_key = (pair_type, key1, key2)
    if (not reencode) and cache_key in _embedding_cache:
        cached_text, cached_emb = _embedding_cache[cache_key]
        # if cached_emb is not None:
        #     print(f"[Cache] ✓ HIT for ({pair_type}, {key1[:20]}, {key2[:20]})")
        return cached_text, cached_emb

    text = fetch_pair_evidence_text(pair_type, key1, key2)
    if text is None:
        text = ""
    try:
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            from .config import SBERT_MODEL_NAME
            encoder = SentenceTransformer(SBERT_MODEL_NAME)
        emb = encoder.encode([text], show_progress_bar=False)[0].tolist()
    except Exception as e:
        print(f"[Encoding] ✗ ERROR encoding text: {e}")
        emb = None
    _embedding_cache[cache_key] = (text, emb)
    if emb is not None:
        print(f"[Encoding] ✓ encoded ({pair_type}, {key1[:20]}, {key2[:20]}) -> {len(emb)} dims")
    else:
        print(f"[Encoding] ✗ FAILED for ({pair_type}, {key1[:20]}, {key2[:20]})")
    return text, emb


