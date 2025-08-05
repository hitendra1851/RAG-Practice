# production_reranking.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time


class ProductionReranker:
    def __init__(self, base_rag):
        self.base_rag = base_rag
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Performance optimizations
        self.reranker.max_length = 256  # Truncate long documents
        self.batch_size = 8  # Process in batches

    def optimized_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Optimized reranking with batching and truncation"""

        if not candidates:
            return []

        # Prepare pairs with truncation
        query_doc_pairs = []
        for candidate in candidates:
            text = self._extract_text(candidate)
            # Truncate documents to improve speed
            truncated_text = text[:500]  # Keep first 500 chars
            query_doc_pairs.append([query, truncated_text])

        # Batch prediction for better GPU utilization
        all_scores = []
        for i in range(0, len(query_doc_pairs), self.batch_size):
            batch = query_doc_pairs[i:i + self.batch_size]
            batch_scores = self.reranker.predict(batch)
            all_scores.extend(batch_scores)

        # Combine with results
        reranked = []
        for candidate, score in zip(candidates, all_scores):
            reranked.append({
                'text': self._extract_text(candidate),
                'source': self._extract_field(candidate, 'source'),
                'rerank_score': float(score),
                'original_score': self._extract_score(candidate, 'combined_score')
            })

        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked

    async def async_rerank(self, query: str, top_k: int = 5) -> List[Dict]:
        """Async reranking for better concurrency"""

        loop = asyncio.get_event_loop()

        # Run retrieval in thread pool
        candidates = await loop.run_in_executor(
            self.thread_pool,
            self.base_rag.hybrid_search,
            query,
            20
        )

        # Run reranking in thread pool
        reranked = await loop.run_in_executor(
            self.thread_pool,
            self.optimized_rerank,
            query,
            candidates
        )

        return reranked[:top_k]

    def cache_enabled_rerank(self, query: str, top_k: int = 5,
                             cache: Dict = None) -> List[Dict]:
        """Reranking with result caching"""

        if cache is None:
            cache = {}

        cache_key = f"{query}:{top_k}"

        # Check cache first
        if cache_key in cache:
            print("✅ Cache hit!")
            return cache[cache_key]

        # Compute results
        candidates = self.base_rag.hybrid_search(query, top_k=20)
        reranked = self.optimized_rerank(query, candidates)

        # Cache results
        cache[cache_key] = reranked[:top_k]

        return reranked[:top_k]