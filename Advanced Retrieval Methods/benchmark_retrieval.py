# benchmark_retrieval.py
import time
from typing import Dict, List


def benchmark_retrieval_methods(hybrid_rag, test_queries: List[str]):
    """Compare performance of different retrieval methods"""

    results = {
        'dense_only': {'times': [], 'avg_scores': []},
        'sparse_only': {'times': [], 'avg_scores': []},
        'hybrid': {'times': [], 'avg_scores': []}
    }

    for query in test_queries:
        print(f"🧪 Testing: '{query}'")

        # Test Dense Only
        start = time.time()
        dense_results = hybrid_rag.dense_search(query, top_k=5)
        dense_time = time.time() - start
        dense_avg_score = sum([score for _, score in dense_results]) / len(dense_results)

        # Test Sparse Only
        start = time.time()
        sparse_results = hybrid_rag.sparse_search(query, top_k=5)
        sparse_time = time.time() - start
        sparse_avg_score = sum([score for _, score in sparse_results]) / max(len(sparse_results), 1)

        # Test Hybrid
        start = time.time()
        hybrid_results = hybrid_rag.hybrid_search(query, top_k=5)
        hybrid_time = time.time() - start
        hybrid_avg_score = sum([r['combined_score'] for r in hybrid_results]) / len(hybrid_results)

        # Store results
        results['dense_only']['times'].append(dense_time)
        results['dense_only']['avg_scores'].append(dense_avg_score)
        results['sparse_only']['times'].append(sparse_time)
        results['sparse_only']['avg_scores'].append(sparse_avg_score)
        results['hybrid']['times'].append(hybrid_time)
        results['hybrid']['avg_scores'].append(hybrid_avg_score)

    # Print summary
    print(f"\n{'=' * 50}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'=' * 50}")

    for method, data in results.items():
        avg_time = sum(data['times']) / len(data['times'])
        avg_score = sum(data['avg_scores']) / len(data['avg_scores'])
        print(f"{method.upper():<12} | Avg Time: {avg_time:.4f}s | Avg Score: {avg_score:.3f}")
