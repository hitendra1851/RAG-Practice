# domain_specific_reranking.py
class DomainSpecificReranker:
    def __init__(self, base_rag):
        self.base_rag = base_rag

        # Different rerankers for different document types
        self.rerankers = {
            'policy': CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'),
            'faq': CrossEncoder('cross-encoder/qnli-electra-base'),
            'procedure': CrossEncoder('cross-encoder/stsb-distilroberta-base')
        }

    def domain_aware_search(self, query: str, top_k: int = 5):
        """Use different rerankers based on document types in results"""

        candidates = self.base_rag.hybrid_search(query, top_k=20)

        # Group candidates by document type
        grouped_candidates = {}
        for candidate in candidates:
            doc_type = self._extract_field(candidate, 'doc_type')
            if doc_type not in grouped_candidates:
                grouped_candidates[doc_type] = []
            grouped_candidates[doc_type].append(candidate)

        # Rerank each group with appropriate reranker
        all_reranked = []
        for doc_type, type_candidates in grouped_candidates.items():
            if doc_type in self.rerankers:
                reranker = self.rerankers[doc_type]
                pairs = [[query, self._extract_text(c)] for c in type_candidates]
                scores = reranker.predict(pairs)

                for candidate, score in zip(type_candidates, scores):
                    all_reranked.append({
                        'text': self._extract_text(candidate),
                        'source': self._extract_field(candidate, 'source'),
                        'doc_type': doc_type,
                        'rerank_score': float(score),
                        'reranker_used': doc_type
                    })

        # Sort all results together
        all_reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        return all_reranked[:top_k]