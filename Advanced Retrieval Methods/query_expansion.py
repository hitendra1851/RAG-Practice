# query_expansion.py - Fixed version
from typing import List, Dict


class QueryExpansionRAG:
    def __init__(self, base_rag_system):
        self.base_rag = base_rag_system
        self.synonyms = {
            'vacation': ['PTO', 'time off', 'leave', 'holiday', 'days off'],
            'password': ['login', 'credentials', 'authentication', 'access'],
            'computer': ['laptop', 'workstation', 'PC', 'machine'],
            'reset': ['change', 'update', 'recover', 'restore'],
            'policy': ['rule', 'guideline', 'procedure', 'regulation'],
            'employee': ['worker', 'staff', 'personnel', 'team member'],
            'portal': ['website', 'platform', 'system', 'site'],
            'VPN': ['virtual private network', 'remote access', 'secure connection']
        }

    def expand_query(self, query: str) -> List[str]:
        """Generate multiple query variations"""
        expanded_queries = [query]

        words = query.lower().split()
        for word in words:
            if word in self.synonyms:
                for synonym in self.synonyms[word]:
                    expanded_query = query.lower().replace(word, synonym)
                    expanded_queries.append(expanded_query)

        # Remove duplicates
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q not in seen:
                unique_queries.append(q)
                seen.add(q)

        return unique_queries

    def search_with_expansion(self, query: str, top_k: int = 5) -> Dict:
        """Search using multiple query variations - FIXED VERSION"""

        try:
            # Generate expanded queries
            expanded_queries = self.expand_query(query)
            print(f"\n🔍 Query Expansion for: '{query}'")
            print(f"📝 Generated {len(expanded_queries)} variations:")
            for i, eq in enumerate(expanded_queries, 1):
                print(f"   {i}. '{eq}'")

            # Collect all results
            all_results = []

            for expanded_query in expanded_queries:
                try:
                    # Try hybrid search first
                    if hasattr(self.base_rag, 'hybrid_search'):
                        results = self.base_rag.hybrid_search(expanded_query, top_k=top_k * 2)
                    else:
                        # Fallback to regular search
                        results = self.base_rag.search(expanded_query, top_k=top_k * 2)

                    # Ensure results is a list
                    if results and isinstance(results, list):
                        all_results.extend(results)

                except Exception as e:
                    print(f"⚠️ Error searching with query '{expanded_query}': {e}")
                    continue

            # Remove duplicates
            seen_docs = {}
            for result in all_results:
                if not isinstance(result, dict):
                    continue

                # Create unique identifier
                if 'id' in result:
                    doc_id = result['id']
                else:
                    doc_id = f"{result.get('source', 'unknown')}_{hash(result.get('text', ''))}"

                # Determine score key
                score_key = None
                for key in ['combined_score', 'similarity_score', 'score']:
                    if key in result:
                        score_key = key
                        break

                if score_key is None:
                    continue

                # Keep result with higher score
                if doc_id in seen_docs:
                    if result[score_key] > seen_docs[doc_id][score_key]:
                        seen_docs[doc_id] = result
                else:
                    seen_docs[doc_id] = result

            # Sort by score
            final_results = list(seen_docs.values())
            if final_results:
                # Determine score key for sorting
                score_key = None
                for key in ['combined_score', 'similarity_score', 'score']:
                    if key in final_results[0]:
                        score_key = key
                        break

                if score_key:
                    final_results.sort(key=lambda x: x[score_key], reverse=True)

            # ALWAYS return dictionary format
            return {
                'original_query': query,
                'expanded_queries': expanded_queries,
                'results': final_results[:top_k],
                'total_found': len(final_results)
            }

        except Exception as e:
            print(f"❌ Error in search_with_expansion: {e}")
            # Return empty but valid dictionary format
            return {
                'original_query': query,
                'expanded_queries': [query],
                'results': [],
                'error': str(e)
            }