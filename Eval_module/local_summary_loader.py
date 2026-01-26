from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

SUMMARY_OUTPUT_DIR = os.environ.get("SUMMARY_OUTPUT_DIR", "data/precomputed_summaries")

def _norm_key(s: str) -> str:
    return (s or "").strip().lower()


class LocalSummaryStore:
    def __init__(self, base_dir: Optional[str] = None) -> None:
        p = Path(base_dir) if base_dir else Path(SUMMARY_OUTPUT_DIR)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        self.base_dir = p
        self._loaded: Dict[str, bool] = {}
        self._cache: Dict[Tuple[str, str, str], str] = {}

    def _load_pair_type(self, pair_type: str) -> None:
        pt = pair_type.strip()
        if self._loaded.get(pt):
            return
        path = self.base_dir / f"{pt}.jsonl"
        if not path.exists():
            self._loaded[pt] = True
            return

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                k1 = _norm_key(rec.get("key1", ""))
                k2 = _norm_key(rec.get("key2", ""))
                summ = rec.get("summary", "")
                if not isinstance(summ, str):
                    continue
                if k1 and k2:
                    self._cache[(pt, k1, k2)] = summ
        self._loaded[pt] = True

    def get(self, pair_type: str, key1: str, key2: str, default: str = "") -> str:
        pt = pair_type.strip()
        self._load_pair_type(pt)
        return self._cache.get((pt, _norm_key(key1), _norm_key(key2)), default)


_DEFAULT_STORE = LocalSummaryStore()


def get_pair_evidence_text(pair_type: str, key1: str, key2: str, default: str = "") -> str:
    """Lookup evidence text from local precomputed JSONL files.

    pair_type should match the JSONL filename prefix, e.g. "slc_pathway" or "pathway_disease".
    """

    return _DEFAULT_STORE.get(pair_type, key1, key2, default=default)
