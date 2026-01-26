"""Offline script to precompute LLM summaries for pathway/disease/SLC pairs.

Usage example::

    python llm_generate_summaries.py \
        --pair-types slc_pathway pathway_disease \
        --overwrite

The script connects to MongoDB once, reads all pair documents, gathers the
associated papers' title + abstract text, invokes an LLM to summarise, and
stores per-pair summaries in local JSONL files. Downstream models can consume
these JSONL files without touching MongoDB again.

Environment variables of interest:
    OPENAI_API_KEY         API key for OpenAI-compatible endpoint.
    OPENAI_BASE_URL        Optional base URL for self-hosted deployments.
    LLM_MODEL_NAME         Override default model (default: config.LLM_MODEL_NAME).
    SUMMARY_OUTPUT_DIR     Output directory (default: config.SUMMARY_OUTPUT_DIR).
    LLM_MAX_OUTPUT_TOKENS  Max tokens for summaries (default: 512).

The produced JSON lines follow the schema::

    {
        "pair_type": "slc_pathway",
        "key1": "SLC1A5",
        "key2": "oxidative phosphorylation",
        "summary": "... <= 512 tokens ...",
        "dois": ["10.1000/j.jmb.2010.01.001", ...]
    }

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import re

from pymongo import MongoClient

try:  # Optional dependency for accurate token counting/truncation
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional
    tiktoken = None  # type: ignore

try:  # OpenAI Python SDK >=1.0
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - ensures actionable error
    raise SystemExit(
        "Please install the OpenAI Python SDK (pip install openai) before running this script."
    ) from exc

client = OpenAI(
# chatgpt
# base_url="https://35api.huinong.co/v1",#中转api，如果直接用openai的key则不需要加这一行
# api_key="sk-Iyo0TKs6ttlhpFOm73Ef3aB2E98840A0927341D15a5c1704"
# bltcy.ai API (兼容OpenAI格式)
base_url="https://api.bltcy.ai/v1",  #https://one-api.bltcy.top/v1/chat/completions(gemini-1.5-pro)
api_key="sk-nNaTuesu68YEGEK060BTydE5fihxblAyJnPRrROuiEEUxmrx"
)

from config import (
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MODEL_NAME,
    MONGO_COLLECTION_PAPERS,
    MONGO_COLLECTION_PW_DZ,
    MONGO_COLLECTION_SLC_PW,
    MONGO_FIELD_DISEASE,
    MONGO_FIELD_PATHWAY,
    MONGO_FIELD_SLC,
    MONGO_DB_NAME,
    MONGO_URI,
    SUMMARY_OUTPUT_DIR,
)


@dataclass(frozen=True)
class PairCollectionConfig:
    pair_type: str
    collection: str
    key_fields: Tuple[str, str]


PAIR_CONFIGS: Dict[str, PairCollectionConfig] = {
    "slc_pathway": PairCollectionConfig(
        pair_type="slc_pathway",
        collection=MONGO_COLLECTION_SLC_PW,
        key_fields=(MONGO_FIELD_SLC, MONGO_FIELD_PATHWAY),
    ),
    "pathway_disease": PairCollectionConfig(
        pair_type="pathway_disease",
        collection=MONGO_COLLECTION_PW_DZ,
        key_fields=(MONGO_FIELD_PATHWAY, MONGO_FIELD_DISEASE),
    ),
}


def _normalize_doi(doi: str) -> str:
    if not doi:
        return ""
    d = doi.strip().lower()
    if d.startswith("https://") or d.startswith("http://"):
        parts = d.split("doi.org/")
        d = parts[-1] if len(parts) > 1 else d.split("/", 3)[-1]
    d = d.replace(" ", "")
    return d


class TokenLimiter:
    def __init__(self, model_name: str, limit: int) -> None:
        self.limit = limit
        self.model_name = model_name
        self.encoding = None
        if tiktoken is not None:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except Exception:
                # Fall back to cl100k_base which supports GPT-style models
                try:
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self.encoding = None

    def truncate(self, text: str) -> str:
        if not text or self.limit <= 0:
            return text
        if self.encoding is None:
            tokens = text.split()
            if len(tokens) <= self.limit:
                return text
            return " ".join(tokens[: self.limit])
        token_ids = self.encoding.encode(text)
        if len(token_ids) <= self.limit:
            return text
        truncated = token_ids[: self.limit]
        return self.encoding.decode(truncated)

class LLMClient:
    def __init__(self, model_name: str, max_output_tokens: int) -> None:
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self._token_limiter = TokenLimiter(model_name, max_output_tokens)

    def summarize_pair(self, pair_type: str, key1: str, key2: str, merged_text: str) -> str:
        if not merged_text.strip():
            return ""

        system_content = (
            "You are an expert SLC-tumor research assistant. "
            "Analyse up to a few hundred paper titles and abstracts to extract verbatim phrases or "
            "sentences that describe promotion or suppression relationships between the entities. "
            "Your answer must not exceed 512 tokens."
        )
        user_content = (
            f"Pair Type: {pair_type}\n"
            f"Entity 1: {key1}\n"
            f"Entity 2: {key2}\n"
            "\n"
            "Use the following paper snippets (titles and abstracts). There may be between 1 and 256 "
            "abstracts. Select ONLY exact phrases or sentences from the original text that explicitly "
            "indicate that one entity promotes, activates, enhances, induces, facilitates, or otherwise "
            "supports the other entity, OR that one suppresses, inhibits, represses, reduces, or "
            "otherwise blocks the other entity. Do not paraphrase, and do not invent new wording. "
            "Output format requirements (STRICT):\n"
            "1) Output ONLY the copied original excerpts, nothing else.\n"
            "2) No preface, no headings, no labels, no explanations, no markdown, no numbering.\n"
            "3) One excerpt per line (plain text).\n"
            "4) Each line must be a verbatim substring from Source Text.\n"
            "5) Each excerpt must be a PHRASE or FULL SENTENCE (not a single word).\n"
            "   - Prefer complete sentences ending with punctuation when possible.\n"
            "   - Do NOT output isolated keywords like 'promotes' or 'inhibits' alone.\n"
            "6) Try to fill up to the 512-token limit with the most informative excerpts.\n"
            "If you cannot find explicit promotion/suppression lines, include the closest mechanistic "
            "relationship lines as backup, still verbatim."
            "\n\n"
            f"Source Text:\n{merged_text}"
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=self.max_output_tokens,
        )

        summary = ""
        if response and getattr(response, "choices", None):
            first_choice = response.choices[0]
            message = getattr(first_choice, "message", {})
            if isinstance(message, dict):
                summary = message.get("content", "") or ""
            else:
                summary = getattr(message, "content", "") or ""
        summary = _extractive_postprocess(summary)
        return self._token_limiter.truncate(summary)


def _extractive_postprocess(text: str) -> str:
    """Remove any non-excerpt scaffolding (headers, explanations).

    Keep only lines that look like direct excerpts.
    """

    if not text:
        return ""
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    min_chars = int(os.environ.get("MIN_EXCERPT_CHARS", "40"))
    min_words = int(os.environ.get("MIN_EXCERPT_WORDS", "6"))
    kept: List[str] = []
    for ln in lines:
        if not ln:
            continue
        lower = ln.lower()

        # Drop typical preface/headers
        if lower.startswith("based on"):
            continue
        if lower.startswith("note:") or lower.startswith("note "):
            continue
        if lower.startswith("#") or lower.startswith("##"):
            continue
        if "verbatim" in lower and ("here are" in lower or "below" in lower):
            continue
        if "relationship" in lower and ("describe" in lower or "between" in lower) and ("here" in lower):
            continue

        # Remove common bullet markers but keep the excerpt content
        ln2 = ln
        for prefix in ("- ", "• ", "* "):
            if ln2.startswith(prefix):
                ln2 = ln2[len(prefix):].strip()
                break

        # If the model wrapped excerpts in quotes, strip only the outermost quotes
        if (ln2.startswith('"') and ln2.endswith('"')) or (ln2.startswith("'") and ln2.endswith("'")):
            ln2 = ln2[1:-1].strip()

        # Drop lines that still look like labels
        low2 = ln2.lower()
        if low2.endswith(":"):
            continue
        if low2 in {"promotion", "suppression", "related context", "promotion/association"}:
            continue

        # Drop single-word / too-short excerpts (common failure mode)
        if len(ln2) < min_chars:
            continue
        if len(ln2.split()) < min_words:
            continue

        if ln2:
            kept.append(ln2)

    return "\n".join(kept).strip()


def load_existing_pairs(path: Path) -> Set[Tuple[str, str]]:
    if not path.exists():
        return set()
    existing: Set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                existing.add((record["key1"], record["key2"]))
            except Exception:
                continue
    return existing


def fetch_paper_texts(papers_collection, dois: Sequence[str]) -> List[str]:
    normed = sorted({_normalize_doi(doi) for doi in dois if _normalize_doi(doi)})
    if not normed:
        return []
    query = {
        "$or": [
            {"doi": {"$in": normed}},
            {"normalized_doi": {"$in": normed}},
        ]
    }
    cursor = papers_collection.find(query, {"_id": 0, "title": 1, "abstract": 1})
    texts: List[str] = []
    for doc in cursor:
        title = doc.get("title")
        abstract = doc.get("abstract")
        parts: List[str] = []
        if isinstance(title, str) and title.strip():
            parts.append(f"Title: {title.strip()}")
        if isinstance(abstract, str) and abstract.strip():
            parts.append(f"Abstract: {abstract.strip()}")
        if parts:
            texts.append("\n".join(parts))
    return texts


def aggregate_texts(texts: Iterable[str]) -> str:
    return "\n\n".join(t for t in texts if t.strip())


PROMOTE_KEYWORDS = [
    "promote",
    "promotes",
    "promoting",
    "activate",
    "activates",
    "activation",
    "enhance",
    "enhances",
    "enhanced",
    "increase",
    "increases",
    "increased",
    "induce",
    "induces",
    "induced",
    "facilitate",
    "facilitates",
    "upregulate",
    "upregulates",
    "upregulated",
]

SUPPRESS_KEYWORDS = [
    "suppress",
    "suppresses",
    "suppression",
    "inhibit",
    "inhibits",
    "inhibited",
    "repress",
    "represses",
    "reduce",
    "reduces",
    "reduced",
    "decrease",
    "decreases",
    "decreased",
    "downregulate",
    "downregulates",
    "downregulated",
    "block",
    "blocks",
]

MECHANISM_HINTS = [
    "via",
    "through",
    "mediated",
    "pathway",
    "signaling",
    "mechanism",
    "regulat",
]


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.!?;。！？；])\s+", t)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences


def _contains(s: str, sub: str) -> bool:
    return sub.lower() in s.lower()


def score_sentence(sentence: str, key1: str, key2: str) -> float:
    s = sentence.strip()
    if not s:
        return -1.0
    score = 0.0
    has_k1 = _contains(s, key1)
    has_k2 = _contains(s, key2)
    if has_k1:
        score += 2.0
    if has_k2:
        score += 2.0
    if has_k1 and has_k2:
        score += 4.0

    sl = s.lower()
    if any(k in sl for k in PROMOTE_KEYWORDS):
        score += 3.0
    if any(k in sl for k in SUPPRESS_KEYWORDS):
        score += 3.0
    if any(k in sl for k in MECHANISM_HINTS):
        score += 1.0

    if len(s) < 20:
        score -= 0.5
    return score


def count_tokens(text: str, model_name: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return len(text.split())
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def build_candidate_pool(
    paper_texts: Sequence[str],
    pair_type: str,
    key1: str,
    key2: str,
    model_name: str,
    min_pool_tokens: int,
    max_pool_tokens: int,
) -> str:
    """High-recall candidate pool.

    Strategy:
    - Split each paper into sentences.
    - Rank sentences by a simple heuristic score.
    - Start with top-k per paper and expand k / add neighbor sentences until pool >= min_pool_tokens.
    - Finally pack into <= max_pool_tokens for LLM input.
    """

    per_paper_sentences: List[List[str]] = []
    per_paper_scored: List[List[Tuple[float, int]]] = []
    for text in paper_texts:
        sents = split_sentences(text)
        per_paper_sentences.append(sents)
        scores = [(score_sentence(s, key1, key2), idx) for idx, s in enumerate(sents)]
        scores.sort(key=lambda x: x[0], reverse=True)
        per_paper_scored.append(scores)

    def _collect(k: int, add_neighbors: bool) -> List[Tuple[float, int, int, str]]:
        collected: List[Tuple[float, int, int, str]] = []
        for p_idx, scores in enumerate(per_paper_scored):
            sents = per_paper_sentences[p_idx]
            picked: Set[int] = set()
            for score, s_idx in scores[: max(0, k)]:
                if score < 0:
                    continue
                picked.add(s_idx)
                if add_neighbors:
                    if s_idx - 1 >= 0:
                        picked.add(s_idx - 1)
                    if s_idx + 1 < len(sents):
                        picked.add(s_idx + 1)
            for s_idx in sorted(picked):
                s = sents[s_idx]
                sc = score_sentence(s, key1, key2)
                collected.append((sc, p_idx, s_idx, s))
        collected.sort(key=lambda x: (x[0], -len(x[3])), reverse=True)
        return collected

    k = int(os.environ.get("CAND_K_PER_PAPER", "3"))
    add_neighbors = os.environ.get("CAND_ADD_NEIGHBORS", "1") != "0"
    expansions = [k, max(k, 5), max(k, 8), max(k, 12), max(k, 20)]

    collected: List[Tuple[float, int, int, str]] = []
    for k_try in expansions:
        collected = _collect(k_try, add_neighbors)
        joined = "\n".join([c[3] for c in collected])
        if count_tokens(joined, model_name) >= min_pool_tokens:
            break

    header = (
        "[Candidate Excerpts Pool]\n"
        "The following lines are extracted from titles/abstracts. Only copy verbatim phrases/sentences from these lines.\n"
        f"Pair Type: {pair_type} | Entity1: {key1} | Entity2: {key2}\n"
        "---\n"
    )

    packed_lines: List[str] = []
    used_keys: Set[Tuple[int, int]] = set()
    cur_tokens = count_tokens(header, model_name)

    for sc, p_idx, s_idx, s in collected:
        key = (p_idx, s_idx)
        if key in used_keys:
            continue
        line = s.strip()
        if not line:
            continue
        line_tokens = count_tokens(line + "\n", model_name)
        if cur_tokens + line_tokens > max_pool_tokens:
            continue
        packed_lines.append(line)
        used_keys.add(key)
        cur_tokens += line_tokens
        if cur_tokens >= max_pool_tokens:
            break

    return header + "\n".join(packed_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LLM summarisation pipeline")
    parser.add_argument(
        "--pair-types",
        nargs="*",
        default=list(PAIR_CONFIGS.keys()),
        help="Pair types to process (default: all configured types)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSONL files instead of appending/skipping",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing JSONL (skip existing pairs). Ignored when --overwrite is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of pairs per type (default: 0 = process all; set >0 for debugging)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run Mongo queries but skip LLM calls and file writes",
    )
    parser.add_argument(
        "--min-pool-tokens",
        type=int,
        default=int(os.environ.get("MIN_POOL_TOKENS", "3000")),
        help="Minimum candidate pool tokens before calling LLM (default: 3000)",
    )
    parser.add_argument(
        "--max-pool-tokens",
        type=int,
        default=int(os.environ.get("MAX_POOL_TOKENS", "6000")),
        help="Maximum candidate pool tokens passed into LLM (default: 6000)",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    output_dir = Path(SUMMARY_OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_output_dir(output_dir)

    llm_model = os.environ.get("LLM_MODEL_NAME", LLM_MODEL_NAME)
    max_tokens = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", LLM_MAX_OUTPUT_TOKENS))
    llm_client = LLMClient(model_name=llm_model, max_output_tokens=max_tokens)

    client = MongoClient(MONGO_URI)
    try:
        db = client[MONGO_DB_NAME]
        papers_collection = db[MONGO_COLLECTION_PAPERS]

        for pair_type in args.pair_types:
            if pair_type not in PAIR_CONFIGS:
                print(f"[WARN] Unknown pair type '{pair_type}', skipping.")
                continue

            cfg = PAIR_CONFIGS[pair_type]
            pair_collection = db[cfg.collection]
            output_path = output_dir / f"{cfg.pair_type}.jsonl"

            existing_pairs = set()
            if (args.resume or not args.overwrite) and output_path.exists() and not args.overwrite:
                existing_pairs = load_existing_pairs(output_path)

            mode = "w" if args.overwrite else "a"
            processed = 0
            written = 0
            limit_per_type: Optional[int] = args.limit if args.limit and args.limit > 0 else None

            with output_path.open(mode, encoding="utf-8") as writer:
                if args.overwrite:
                    existing_pairs = set()

                cursor = pair_collection.find({}, {cfg.key_fields[0]: 1, cfg.key_fields[1]: 1, "paper_dois": 1})
                for doc in cursor:
                    key1 = str(doc.get(cfg.key_fields[0], "")).strip()
                    key2 = str(doc.get(cfg.key_fields[1], "")).strip()
                    if not key1 or not key2:
                        continue

                    pair_key = (key1, key2)
                    if pair_key in existing_pairs:
                        continue

                    dois_field = doc.get("paper_dois")
                    if isinstance(dois_field, str):
                        dois = [d.strip() for d in dois_field.replace("|", ";").split(";") if d.strip()]
                    elif isinstance(dois_field, Sequence):
                        dois = [str(d).strip() for d in dois_field if str(d).strip()]
                    else:
                        dois = []

                    if not dois:
                        continue

                    try:
                        paper_texts = fetch_paper_texts(papers_collection, dois)
                        if not paper_texts:
                            continue

                        candidate_pool = build_candidate_pool(
                            paper_texts=paper_texts,
                            pair_type=pair_type,
                            key1=key1,
                            key2=key2,
                            model_name=llm_model,
                            min_pool_tokens=args.min_pool_tokens,
                            max_pool_tokens=args.max_pool_tokens,
                        )
                        if not candidate_pool.strip():
                            continue

                        if args.dry_run:
                            summary = ""
                        else:
                            summary = llm_client.summarize_pair(pair_type, key1, key2, candidate_pool)
                    except Exception as e:
                        print(f"[WARN] pair_type={pair_type} key1={key1} key2={key2} failed: {e}")
                        continue

                    record = {
                        "pair_type": pair_type,
                        "key1": key1,
                        "key2": key2,
                        "summary": summary,
                        "dois": dois,
                    }

                    if not args.dry_run:
                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                        writer.flush()
                        written += 1
                    processed += 1

                    if limit_per_type is not None and processed >= limit_per_type:
                        break

            print(
                f"[INFO] pair_type={pair_type} processed={processed} new_records={written} "
                f"output={output_path}"
            )

    finally:
        client.close()


if __name__ == "__main__":
    main()
