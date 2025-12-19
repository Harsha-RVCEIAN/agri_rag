from typing import List, Dict, Optional

# ---------------- CONFIG ----------------

DEFAULT_TOP_K = 6
OVERFETCH_K = 20

MIN_ADJUSTED_SCORE = 0.25
WEAK_RETRIEVAL_THRESHOLD = 0.35
MAX_MERGED_CANDIDATES = 40  # noise guard

CONTENT_TYPE_WEIGHT = {
    "table_row": 1.30,
    "procedure": 1.15,
    "text": 1.0,
    "ocr": 0.7,
}


class Retriever:
    """
    Retriever = truth filter.

    Responsibilities:
    - Rank chunks
    - Penalize low-confidence / OCR content
    - Prefer tables for numeric queries
    - Detect weak / irrelevant retrieval

    NOT responsible for:
    - Embedding
    - Answer generation
    - Persistence
    """

    def __init__(self, vector_store, embedder=None):
        self.vector_store = vector_store
        self.embedder = embedder  # kept for interface consistency

    # ---------- SCORE NORMALIZATION ----------

    def _normalize_scores(self, matches: List[Dict]) -> List[Dict]:
        if not matches:
            return matches

        scores = [m["score"] for m in matches]
        max_s, min_s = max(scores), min(scores)

        if max_s == min_s:
            for m in matches:
                m["norm_score"] = 1.0
            return matches

        for m in matches:
            m["norm_score"] = (m["score"] - min_s) / (max_s - min_s)

        return matches

    # ---------- SCORE ADJUSTMENT ----------

    def _adjust_score(self, match: Dict, intent: Optional[str]) -> float:
        base = match.get("norm_score", match["score"])
        meta = match["metadata"]

        content_type = meta.get("content_type", "text")
        confidence = meta.get("confidence", 1.0)
        priority = meta.get("priority", 3)

        # Base weighting by content type
        score = base * CONTENT_TYPE_WEIGHT.get(content_type, 1.0)

        # OCR / confidence penalties
        if confidence < 0.6:
            score *= 0.75
        if confidence < 0.4:
            score *= 0.5

        # Intent-based bias
        if intent == "eligibility" and content_type == "table_row":
            score *= 1.2
        elif intent == "procedure" and content_type == "procedure":
            score *= 1.15

        # Priority fine-tuning
        score *= (1.0 + (priority - 3) * 0.05)

        return score

    # ---------- MAIN RETRIEVAL ----------

    def retrieve(
        self,
        query: str,
        query_vectors: List[List[float]],
        intent: Optional[str] = None,
        language: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict:

        # ---------- METADATA FILTERS ----------
        filters = {}

        if language:
            filters["language"] = {"$eq": language}

        if intent == "eligibility":
            filters["content_type"] = {"$in": ["table_row", "procedure"]}
            top_k = min(top_k, 5)

        elif intent == "procedure":
            filters["content_type"] = {"$in": ["procedure", "text"]}
            top_k = min(top_k, 7)

        # ---------- DENSE RETRIEVAL ----------
        dense_matches: List[Dict] = []

        for v in query_vectors:
            matches = self.vector_store.query(
                query_vector=v,
                top_k=OVERFETCH_K,
                filters=filters,
            )
            if matches:
                dense_matches.extend(matches)

        if not dense_matches:
            return {
                "chunks": [],
                "diagnostics": {"reason": "no_matches"},
            }

        # ---------- NORMALIZE SCORES ----------
        dense_matches = self._normalize_scores(dense_matches)

        # ---------- MERGE (BEST MATCH PER CHUNK) ----------
        merged: Dict[str, Dict] = {}
        for m in dense_matches:
            cid = m["metadata"]["chunk_id"]
            if cid not in merged or merged[cid]["norm_score"] < m["norm_score"]:
                merged[cid] = m

        # ---------- NOISE GUARD (FIXED) ----------
        # Sort BEFORE slicing to avoid random loss of good chunks
        merged_values = sorted(
            merged.values(),
            key=lambda m: m["norm_score"],
            reverse=True
        )[:MAX_MERGED_CANDIDATES]

        # ---------- ADJUST & FILTER ----------
        ranked = []
        for m in merged_values:
            adjusted = self._adjust_score(m, intent)
            if adjusted < MIN_ADJUSTED_SCORE:
                continue

            meta = m["metadata"]
            ranked.append({
                "chunk_id": meta["chunk_id"],
                "score": adjusted,
                "content_type": meta.get("content_type"),
                "source": meta.get("source"),
                "page": meta.get("page"),
                "confidence": meta.get("confidence"),
                "text": meta.get("text", ""),
            })

        if not ranked:
            return {
                "chunks": [],
                "diagnostics": {"reason": "low_quality"},
            }

        ranked.sort(key=lambda x: x["score"], reverse=True)

        # ---------- WEAK / IRRELEVANT RETRIEVAL ----------
        if ranked[0]["score"] < WEAK_RETRIEVAL_THRESHOLD:
            return {
                "chunks": [],
                "diagnostics": {
                    "reason": "irrelevant_query",
                    "top_score": ranked[0]["score"],
                    "intent": intent,
                },
            }

        # ---------- SUCCESS ----------
        return {
            "chunks": ranked[:top_k],
            "diagnostics": {
                "reason": "ok",
                "returned": len(ranked),
                "used": top_k,
                "intent": intent,
            },
        }
