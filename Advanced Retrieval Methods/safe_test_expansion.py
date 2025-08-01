# safe_test_expansion.py
def safe_test_query_expansion():
    """Safely test query expansion with error handling"""

    from hybrid_retrieval_system import HybridRetrievalRAG
    from query_expansion import QueryExpansionRAG

    # Setup
    hybrid_rag = HybridRetrievalRAG()

    documents = [
        {
            'id': 'hr_001',
            'text': 'Employees receive 15-25 vacation days based on years of service.',
            'source': 'HR_Policy',
            'doc_type': 'policy'
        },
        {
            'id': 'hr_002',
            'text': 'PTO requests must be submitted through the HR portal 14 days in advance.',
            'source': 'HR_Policy',
            'doc_type': 'policy'
        }
    ]

    hybrid_rag.add_documents(documents)
    query_expansion_rag = QueryExpansionRAG(hybrid_rag)

    # Test query
    query = "How much time off do I get?"

    print(f"🧪 Testing Query Expansion: '{query}'")
    print("=" * 60)

    try:
        # Get expansion results
        expanded_results = query_expansion_rag.search_with_expansion(query, top_k=3)

        print(f"✅ Results type: {type(expanded_results)}")

        # Safe extraction of results
        if isinstance(expanded_results, dict):
            results_list = expanded_results.get('results', [])
            original_query = expanded_results.get('original_query', query)
            expanded_queries = expanded_results.get('expanded_queries', [query])

            print(f"📝 Original query: {original_query}")
            print(f"🔄 Expanded to {len(expanded_queries)} variations")
            print(f"📊 Found {len(results_list)} results")

        elif isinstance(expanded_results, list):
            results_list = expanded_results
            print(f"📊 Got list directly with {len(results_list)} results")

        else:
            print(f"❌ Unexpected result type: {type(expanded_results)}")
            return

        # Display results safely
        print(f"\n🎯 RESULTS:")
        for i, result in enumerate(results_list, 1):
            if isinstance(result, dict):
                # Find score field
                score = result.get('combined_score') or result.get('similarity_score') or result.get('score', 0.0)
                text = result.get('text', 'No text available')
                source = result.get('source', 'Unknown source')

                print(f"   {i}. Score: {score:.3f} | Source: {source}")
                print(f"      Text: {text[:60]}...")
            else:
                print(f"   {i}. Unexpected result format: {type(result)}")

    except Exception as e:
        print(f"❌ Error during expansion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    safe_test_query_expansion()