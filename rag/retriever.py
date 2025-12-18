from typing import List, Dict, Optional
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore


DEFAULT_TOP_K = 6
OVERFETCH_K = 20

MIN_ADJUSTED_SCORE = 0.25
WEAK_RETRIEVAL_THRESHOLD = 0.35

CONTENT_TYPE_WEIGHT = {
    "table_row": 1.30,
    "procedure": 1.15,
    "text": 1.0,
    "ocr": 0.7
}


class Retriever:

    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    # ---------- QUERY UTILS ----------

    def _expand_query(self, query: str) -> List[str]:
        q = query.lower()
        expansions = [query]

        if "scheme" in q:
            expansions.append(f"{query} government scheme")

        if "pm kisan" in q or "pm-kisan" in q:
            expansions.append("PM-Kisan Samman Nidhi eligibility application")

        if "insurance" in q:
            expansions.append("crop insurance scheme PMFBY")

        return list(set(expansions))

    def _embed_queries(self, queries: List[str]) -> List[List[float]]:
        records = [{"text": q, "metadata": {"chunk_id": "__query__"}} for q in queries]
        embedded = self.embedder.embed_chunks(records)
        return [r["vector"] for r in embedded]

    # ---------- SCORE NORMALIZATION ----------

    def _normalize_scores(self, matches: List[Dict]) -> List[Dict]:
        """
        Normalize scores to 0–1 range to make dense + keyword comparable.
        """
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

    # ---------- SCORING ----------

    def _adjust_score(self, match: Dict, intent: Optional[str]) -> float:
        base = match.get("norm_score", match["score"])
        meta = match["metadata"]

        content_type = meta.get("content_type", "text")
        confidence = meta.get("confidence", 1.0)
        priority = meta.get("priority", 3)

        score = base * CONTENT_TYPE_WEIGHT.get(content_type, 1.0)

        # Global confidence penalty (not OCR-only)
        if confidence < 0.6:
            score *= 0.75
        if confidence < 0.4:
            score *= 0.5

        # Intent bias
        if intent == "eligibility" and content_type == "table_row":
            score *= 1.2
        if intent == "procedure" and content_type == "procedure":
            score *= 1.15

        # Priority fine-tuning
        score *= (1.0 + (priority - 3) * 0.05)

        return score

    # ---------- MAIN ----------

    def retrieve(
        self,
        query: str,
        intent: Optional[str] = None,
        language: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K
    ) -> Dict:

        filters = {}
        if language:
            filters["language"] = {"$eq": language}

        if intent == "eligibility":
            filters["content_type"] = {"$in": ["table_row", "procedure"]}
            top_k = min(top_k, 5)

        elif intent == "procedure":
            filters["content_type"] = {"$in": ["procedure", "text"]}
            top_k = min(top_k, 7)

        expanded = self._expand_query(query)
        vectors = self._embed_queries(expanded)

        dense = []
        for v in vectors:
            dense.extend(
                self.vector_store.query(
                    query_vector=v,
                    top_k=OVERFETCH_K,
                    filters=filters
                ) or []
            )

        keyword = self.vector_store.keyword_query(
            query=query,
            top_k=OVERFETCH_K,
            filters=filters
        ) or []

        if not dense and not keyword:
            return {"chunks": [], "diagnostics": {"reason": "no_matches"}}

        # ---------- normalize scores separately ----------
        dense = self._normalize_scores(dense)
        keyword = self._normalize_scores(keyword)

        # ---------- merge ----------
        merged = {}
        for m in dense + keyword:
            cid = m["metadata"]["chunk_id"]
            if cid not in merged or merged[cid]["norm_score"] < m["norm_score"]:
                merged[cid] = m

        ranked = []
        for m in merged.values():
            adj = self._adjust_score(m, intent)
            if adj < MIN_ADJUSTED_SCORE:
                continue

            meta = m["metadata"]
            ranked.append({
                "chunk_id": meta["chunk_id"],
                "score": adj,
                "content_type": meta.get("content_type"),
                "source": meta.get("source"),
                "page": meta.get("page"),
                "confidence": meta.get("confidence"),
                "text": meta["text"]
            })

        if not ranked:
            return {"chunks": [], "diagnostics": {"reason": "low_quality"}}

        ranked.sort(key=lambda x: x["score"], reverse=True)

        if ranked[0]["score"] < WEAK_RETRIEVAL_THRESHOLD:
            return {
                "chunks": [],
                "diagnostics": {
                    "reason": "weak_retrieval",
                    "top_score": ranked[0]["score"],
                    "intent": intent
                }
            }

        return {
            "chunks": ranked[:top_k],
            "diagnostics": {
                "expanded_queries": expanded,
                "returned_chunks": len(ranked),
                "intent": intent
            }
        }
