# multi_stage_reranking.py
from sentence_transformers import CrossEncoder


class MultiStageReranker:
    def __init__(self, base_rag):
        self.base_rag = base_rag

        # Different rerankers for different stages
        self.stage_1_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # Fast
        self.stage_2_reranker = CrossEncoder('cross-encoder/ms-marco-electra-base')  # Accurate

    def multi_stage_search(self, query: str, top_k: int = 5):
        """Three-stage pipeline: Retrieve → Fast Rerank → Precise Rerank"""

        # Stage 1: Broad retrieval
        candidates = self.base_rag.hybrid_search(query, top_k=50)

        # Stage 2: Fast reranking (50 → 10)
        stage_1_pairs = [[query, self._extract_text(c)] for c in candidates]
        stage_1_scores = self.stage_1_reranker.predict(stage_1_pairs)

        stage_1_results = list(zip(candidates, stage_1_scores))
        stage_1_results.sort(key=lambda x: x[1], reverse=True)
        top_10_candidates = [r[0] for r in stage_1_results[:10]]

        # Stage 3: Precise reranking (10 → 5)
        stage_2_pairs = [[query, self._extract_text(c)] for c in top_10_candidates]
        stage_2_scores = self.stage_2_reranker.predict(stage_2_pairs)

        final_results = []
        for candidate, score in zip(top_10_candidates, stage_2_scores):
            final_results.append({
                'text': self._extract_text(candidate),
                'source': self._extract_field(candidate, 'source'),
                'final_score': float(score)
            })

        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        return final_results[:top_k]
