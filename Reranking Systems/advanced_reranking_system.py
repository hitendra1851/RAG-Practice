# advanced_reranking_system.py
from sentence_transformers import CrossEncoder
import torch
import time
from typing import List, Dict, Tuple
import numpy as np


class AdvancedRerankedRAG:
    def __init__(self, base_hybrid_rag, reranker_models=None):
        self.base_rag = base_hybrid_rag

        # Default to multiple reranker models for different use cases
        if reranker_models is None:
            self.rerankers = {
                'general': CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'),
                'qa': CrossEncoder('cross-encoder/qnli-electra-base'),
                'semantic': CrossEncoder('cross-encoder/stsb-distilroberta-base')
            }
        else:
            self.rerankers = reranker_models

        self.default_reranker = 'general'
        print(f"✅ Loaded {len(self.rerankers)} reranker models")

    def search_with_reranking(self, query: str, top_k: int = 5, retrieve_k: int = 20,
                              reranker_type: str = None, explain: bool = False) -> List[Dict]:
        """
        Advanced two-stage search with detailed scoring

        Args:
            query: Search query
            top_k: Final number of results to return
            retrieve_k: Number of candidates to retrieve for reranking
            reranker_type: Which reranker model to use ('general', 'qa', 'semantic')
            explain: Whether to include detailed scoring explanations
        """

        # Choose reranker
        reranker_type = reranker_type or self.default_reranker
        reranker = self.rerankers[reranker_type]

        if explain:
            print(f"\n🔍 RERANKING PIPELINE EXPLANATION")
            print(f"{'=' * 60}")
            print(f"Query: '{query}'")
            print(f"Reranker: {reranker_type}")
            print(f"Retrieve K: {retrieve_k} → Rerank to Top K: {top_k}")

        # Stage 1: Broad retrieval
        start_time = time.time()
        candidates = self.base_rag.hybrid_search(query, top_k=retrieve_k)
        retrieval_time = time.time() - start_time

        if not candidates:
            return []

        if explain:
            print(f"\n📊 STAGE 1: RETRIEVAL ({retrieval_time:.3f}s)")
            print("Top 3 candidates from hybrid search:")
            for i, candidate in enumerate(candidates[:3], 1):
                score = self._extract_score(candidate, 'combined_score')
                text = self._extract_text(candidate)
                print(f"   {i}. Score: {score:.3f} | {text[:60]}...")

        # Stage 2: Precise reranking
        start_time = time.time()

        # Prepare query-document pairs
        query_doc_pairs = []
        for candidate in candidates:
            text = self._extract_text(candidate)
            query_doc_pairs.append([query, text])

        # Get reranking scores
        rerank_scores = reranker.predict(query_doc_pairs)
        reranking_time = time.time() - start_time

        # Combine results with scores
        reranked_results = []
        for candidate, rerank_score in zip(candidates, rerank_scores):
            original_score = self._extract_score(candidate, 'combined_score')

            result = {
                'text': self._extract_text(candidate),
                'source': self._extract_field(candidate, 'source'),
                'doc_type': self._extract_field(candidate, 'doc_type'),
                'id': self._extract_field(candidate, 'id'),
                'original_score': float(original_score),
                'rerank_score': float(rerank_score),
                'score_improvement': float(rerank_score) - float(original_score),
                'reranker_used': reranker_type
            }

            reranked_results.append(result)

        # Sort by rerank score
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

        if explain:
            print(f"\n🎯 STAGE 2: RERANKING ({reranking_time:.3f}s)")
            print("Top 3 after reranking:")
            for i, result in enumerate(reranked_results[:3], 1):
                improvement = result['score_improvement']
                improvement_str = f"(+{improvement:.3f})" if improvement > 0 else f"({improvement:.3f})"
                print(f"   {i}. Score: {result['rerank_score']:.3f} {improvement_str}")
                print(f"      Text: {result['text'][:60]}...")

            # Show position changes
            self._show_position_changes(candidates, reranked_results, top_k)

        return reranked_results[:top_k]

    def _extract_text(self, candidate) -> str:
        """Extract text from different candidate formats"""
        if isinstance(candidate, dict):
            if 'doc' in candidate:
                return candidate['doc'].get('text', '')
            return candidate.get('text', '')
        return str(candidate)

    def _extract_score(self, candidate, score_key: str) -> float:
        """Extract score from different candidate formats"""
        if isinstance(candidate, dict):
            if 'doc' in candidate:
                return candidate.get(score_key, 0.0)
            return candidate.get(score_key, 0.0)
        return 0.0

    def _extract_field(self, candidate, field: str):
        """Extract field from different candidate formats"""
        if isinstance(candidate, dict):
            if 'doc' in candidate:
                return candidate['doc'].get(field, 'unknown')
            return candidate.get(field, 'unknown')
        return 'unknown'

    def _show_position_changes(self, original_candidates, reranked_results, top_k):
        """Show how reranking changed result positions"""
        print(f"\n📈 POSITION CHANGES (Top {top_k}):")

        # Create mapping of original positions
        original_positions = {}
        for i, candidate in enumerate(original_candidates):
            doc_id = self._extract_field(candidate, 'id')
            original_positions[doc_id] = i + 1

        # Show new positions
        for new_pos, result in enumerate(reranked_results[:top_k], 1):
            doc_id = result['id']
            old_pos = original_positions.get(doc_id, 'N/A')

            if old_pos != 'N/A':
                if new_pos < old_pos:
                    change = f"↑ {old_pos}→{new_pos}"
                elif new_pos > old_pos:
                    change = f"↓ {old_pos}→{new_pos}"
                else:
                    change = f"= {new_pos}"
            else:
                change = f"NEW → {new_pos}"

            print(f"   {change} | Score: {result['rerank_score']:.3f} | {result['text'][:50]}...")

    def compare_rerankers(self, query: str, top_k: int = 3) -> Dict:
        """Compare different reranker models on the same query"""

        print(f"\n{'=' * 80}")
        print(f"🧪 COMPARING RERANKER MODELS")
        print(f"Query: '{query}'")
        print(f"{'=' * 80}")

        results = {}

        for reranker_name in self.rerankers.keys():
            print(f"\n🔧 Testing {reranker_name.upper()} reranker:")

            reranked_results = self.search_with_reranking(
                query, top_k=top_k, reranker_type=reranker_name
            )

            results[reranker_name] = reranked_results

            for i, result in enumerate(reranked_results, 1):
                print(f"   {i}. Score: {result['rerank_score']:.3f}")
                print(f"      Text: {result['text'][:70]}...")

        return results

    def benchmark_reranking_impact(self, test_queries: List[str]) -> Dict:
        """Benchmark the impact of reranking across multiple queries"""

        print(f"\n{'=' * 80}")
        print(f"📊 RERANKING IMPACT BENCHMARK")
        print(f"Testing {len(test_queries)} queries...")
        print(f"{'=' * 80}")

        metrics = {
            'queries_tested': len(test_queries),
            'average_improvements': {},
            'top_1_accuracy_improvement': 0,
            'timing': {'retrieval': [], 'reranking': []},
            'detailed_results': []
        }

        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")

            # Get hybrid results (baseline)
            start_time = time.time()
            hybrid_results = self.base_rag.hybrid_search(query, top_k=5)
            retrieval_time = time.time() - start_time

            # Get reranked results
            start_time = time.time()
            reranked_results = self.search_with_reranking(query, top_k=5)
            total_time = time.time() - start_time
            reranking_time = total_time - retrieval_time

            # Calculate improvements
            if hybrid_results and reranked_results:
                hybrid_top_score = self._extract_score(hybrid_results[0], 'combined_score')
                reranked_top_score = reranked_results[0]['rerank_score']

                improvement = reranked_top_score - hybrid_top_score

                print(f"   Hybrid top score: {hybrid_top_score:.3f}")
                print(f"   Reranked top score: {reranked_top_score:.3f}")
                print(f"   Improvement: {improvement:+.3f}")

                metrics['detailed_results'].append({
                    'query': query,
                    'hybrid_score': hybrid_top_score,
                    'reranked_score': reranked_top_score,
                    'improvement': improvement,
                    'retrieval_time': retrieval_time,
                    'reranking_time': reranking_time
                })

            metrics['timing']['retrieval'].append(retrieval_time)
            metrics['timing']['reranking'].append(reranking_time)

        # Calculate summary metrics
        if metrics['detailed_results']:
            improvements = [r['improvement'] for r in metrics['detailed_results']]
            metrics['average_improvements']['mean'] = np.mean(improvements)
            metrics['average_improvements']['median'] = np.median(improvements)
            metrics['average_improvements']['positive_count'] = sum(1 for i in improvements if i > 0)
            metrics['average_improvements']['positive_percentage'] = (
                    metrics['average_improvements']['positive_count'] / len(improvements) * 100
            )

        metrics['timing']['average_retrieval'] = np.mean(metrics['timing']['retrieval'])
        metrics['timing']['average_reranking'] = np.mean(metrics['timing']['reranking'])

        # Print summary
        print(f"\n📊 BENCHMARK SUMMARY:")
        print(f"   Average improvement: {metrics['average_improvements']['mean']:+.3f}")
        print(f"   Queries improved: {metrics['average_improvements']['positive_percentage']:.1f}%")
        print(f"   Average retrieval time: {metrics['timing']['average_retrieval']:.3f}s")
        print(f"   Average reranking time: {metrics['timing']['average_reranking']:.3f}s")

        return metrics