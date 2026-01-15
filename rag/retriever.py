from typing import List, Dict, Optional
from collections import Counter, defaultdict

# =========================================================
# CONFIG
# =========================================================

DEFAULT_TOP_K = 6
OVERFETCH_K = 30

MIN_ADJUSTED_SCORE = 0.30
WEAK_RETRIEVAL_THRESHOLD = 0.40
MIN_COVERAGE_CHUNKS = 2
MAX_MERGED_CANDIDATES = 50

MAX_CHUNKS_PER_PAGE = 2
MAX_CHUNKS_PER_SOURCE = 3

MIN_METADATA_CONFIDENCE = 0.25   # 🔑 hard floor

CONTENT_TYPE_WEIGHT = {
    "table_row": 1.35,
    "procedure": 1.15,
    "text": 1.0,
    "ocr": 0.6,
}

INTENT_REQUIREMENTS = {
    "numeric": {"table_row": 1},
    "eligibility": {"table_row": 1},
    "procedure": {"procedure": 1},
}

SINGLE_SOURCE_PENALTY = 0.85
SINGLE_PAGE_PENALTY = 0.90


# =========================================================
# RETRIEVER
# =========================================================

class Retriever:
    """
    Evidence-first retriever.

    STRICT CONTRACT:
    - Accepts embeddings ONLY
    - Never embeds text
    - Never depends on raw query text
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    # -----------------------------------------------------
    # NORMALIZATION
    # -----------------------------------------------------

    def _normalize(self, matches: List[Dict]) -> None:
        scores = [m["score"] for m in matches]
        hi = max(scores)
        lo = min(scores)

        if hi == lo:
            for m in matches:
                m["norm_score"] = 1.0
            return

        denom = hi - lo
        for m in matches:
            m["norm_score"] = (m["score"] - lo) / denom

    # -----------------------------------------------------
    # SCORE ADJUSTMENT
    # -----------------------------------------------------

    def _adjust(self, m: Dict) -> float:
        meta = m["metadata"]
        score = m["norm_score"]

        score *= CONTENT_TYPE_WEIGHT.get(
            meta.get("content_type", "text"), 1.0
        )

        conf = meta.get("confidence", 1.0)
        if conf < 0.6:
            score *= 0.75
        if conf < 0.4:
            score *= 0.5

        priority = meta.get("priority", 3)
        if priority != 3:
            score *= (1 + (priority - 3) * 0.05)

        return score

    # -----------------------------------------------------
    # INTENT CONSTRAINT
    # -----------------------------------------------------

    def _enforce_intent(
        self,
        chunks: List[Dict],
        intent: Optional[str],
    ) -> Optional[str]:

        if not intent or intent not in INTENT_REQUIREMENTS:
            return None

        counts = Counter(c["content_type"] for c in chunks)
        for ctype, required in INTENT_REQUIREMENTS[intent].items():
            if counts.get(ctype, 0) < required:
                return f"missing_required_{ctype}"

        return None

    # -----------------------------------------------------
    # MAIN RETRIEVAL
    # -----------------------------------------------------

    def retrieve(
        self,
        query_vectors: List[List[float]],
        intent: Optional[str] = None,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict:

        # ---------- FILTERS ----------
        filters = {}
        if language:
            filters["language"] = {"$eq": language}
        if domain:
            filters["domain"] = {"$eq": domain}

        filters = filters or None

        # ---------- QUERY ----------
        raw: List[Dict] = []
        for vec in query_vectors:
            res = self.vector_store.query(
                query_vector=vec,
                top_k=OVERFETCH_K,
                filters=filters,
            )
            if res:
                raw.extend(res)

        if not raw:
            return {
                "chunks": [],
                "diagnostics": {"status": "fail", "reason": "no_matches"},
            }

        # ---------- HARD CONFIDENCE FLOOR ----------
        raw = [
            m for m in raw
            if m["metadata"].get("confidence", 1.0) >= MIN_METADATA_CONFIDENCE
        ]

        if not raw:
            return {
                "chunks": [],
                "diagnostics": {"status": "fail", "reason": "all_low_confidence"},
            }

        # ---------- NORMALIZE ----------
        self._normalize(raw)

        # ---------- MERGE BEST PER CHUNK ----------
        merged: Dict[str, Dict] = {}
        for m in raw:
            cid = m["metadata"]["chunk_id"]
            prev = merged.get(cid)
            if not prev or m["norm_score"] > prev["norm_score"]:
                merged[cid] = m

        candidates = sorted(
            merged.values(),
            key=lambda x: x["norm_score"],
            reverse=True,
        )[:MAX_MERGED_CANDIDATES]

        # ---------- RANKING + DIVERSITY ----------
        page_count = defaultdict(int)
        source_count = defaultdict(int)
        ranked: List[Dict] = []

        for m in candidates:
            meta = m["metadata"]
            page = meta.get("page")
            source = meta.get("source")

            if page is not None and page_count[page] >= MAX_CHUNKS_PER_PAGE:
                continue
            if source and source_count[source] >= MAX_CHUNKS_PER_SOURCE:
                continue

            score = self._adjust(m)
            if score < MIN_ADJUSTED_SCORE:
                continue

            page_count[page] += 1
            if source:
                source_count[source] += 1

            ranked.append({
                "chunk_id": meta["chunk_id"],
                "score": round(score, 4),
                "content_type": meta.get("content_type"),
                "source": source,
                "page": page,
                "confidence": meta.get("confidence"),
                "text": m.get("text", ""),   # 🔑 FIXED
            })

            if len(ranked) >= top_k * 2:
                break

        if len(ranked) < MIN_COVERAGE_CHUNKS:
            return {
                "chunks": [],
                "diagnostics": {"status": "fail", "reason": "insufficient_coverage"},
            }

        ranked.sort(key=lambda x: x["score"], reverse=True)

        if ranked[0]["score"] < WEAK_RETRIEVAL_THRESHOLD:
            return {
                "chunks": [],
                "diagnostics": {"status": "fail", "reason": "low_confidence"},
            }

        intent_issue = self._enforce_intent(ranked, intent)
        if intent_issue:
            return {
                "chunks": [],
                "diagnostics": {"status": "fail", "reason": intent_issue},
            }

        # ---------- CONFIDENCE ----------
        top = ranked[:top_k]
        scores = [c["score"] for c in top]

        mx = scores[0]
        avg = sum(scores) / len(scores)
        agreement = min(1.0, len(scores) / 3)

        source_diversity = len({c["source"] for c in top if c["source"]})
        page_diversity = len({c["page"] for c in top if c["page"] is not None})

        penalty = 1.0
        if source_diversity <= 1:
            penalty *= SINGLE_SOURCE_PENALTY
        if page_diversity <= 1:
            penalty *= SINGLE_PAGE_PENALTY

        retrieval_confidence = round(
            (0.6 * mx + 0.3 * avg + 0.1 * agreement) * penalty,
            3,
        )

        return {
            "chunks": top,
            "diagnostics": {
                "status": "ok",
                "intent": intent,
                "domain": domain,
                "retrieval_confidence": retrieval_confidence,
                "content_mix": dict(
                    Counter(c["content_type"] for c in top)
                ),
                "source_diversity": source_diversity,
                "page_diversity": page_diversity,
            },
        }
