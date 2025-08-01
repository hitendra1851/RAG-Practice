# test_hybrid_retrieval_fixed.py
from hybrid_retrieval_system import HybridRetrievalRAG
from query_expansion import QueryExpansionRAG


def test_retrieval_methods():
    # Initialize hybrid system
    hybrid_rag = HybridRetrievalRAG()

    # Sample documents with IDs
    documents = [
        {
            'id': 'hr_001',
            'text': 'Employees receive 15-25 vacation days based on years of service. New employees get 15 days.',
            'source': 'HR_Policy',
            'doc_type': 'policy'
        },
        {
            'id': 'hr_002',
            'text': 'PTO requests must be submitted through the HR portal 14 days in advance.',
            'source': 'HR_Policy',
            'doc_type': 'policy'
        },
        {
            'id': 'it_001',
            'text': 'Password reset: Visit company portal, enter employee ID, follow email instructions.',
            'source': 'IT_FAQ',
            'doc_type': 'faq'
        },
        {
            'id': 'it_002',
            'text': 'VPN connection: Download Cisco AnyConnect, use network credentials.',
            'source': 'IT_FAQ',
            'doc_type': 'faq'
        }
    ]

    # Add documents to system
    hybrid_rag.add_documents(documents)

    # Test queries that benefit from different search types
    test_queries = [
        "How many vacation days do employees get?",  # Should favor semantic
        "PTO portal submission",  # Should favor keyword
        "Cisco AnyConnect VPN",  # Should favor exact match
        "time off policy for new hires"  # Should favor semantic
    ]

    print("🧪 TESTING HYBRID RETRIEVAL METHODS")
    print("=" * 80)

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"🔍 Query: '{query}'")
        print(f"{'=' * 60}")

        # Compare all three methods
        dense_results = hybrid_rag.dense_search(query, top_k=3)
        sparse_results = hybrid_rag.sparse_search(query, top_k=3)
        hybrid_results = hybrid_rag.hybrid_search(query, top_k=3, alpha=0.7)

        print("\n🧠 DENSE (Semantic) Results:")
        for i, (doc, score) in enumerate(dense_results, 1):
            print(f"   {i}. Score: {score:.3f} | {doc['text'][:60]}...")

        print("\n🔤 SPARSE (Keyword) Results:")
        for i, (doc, score) in enumerate(sparse_results, 1):
            print(f"   {i}. Score: {score:.3f} | {doc['text'][:60]}...")

        print("\n⚡ HYBRID Results:")
        for i, result in enumerate(hybrid_results, 1):
            print(f"   {i}. Combined: {result['combined_score']:.3f} "
                  f"(Dense: {result['dense_score']:.3f}, Sparse: {result['sparse_score']:.3f})")
            print(f"      Text: {result['text'][:60]}...")

    # Test Query Expansion (FIXED - using the same hybrid_rag with documents)
    print(f"\n{'=' * 80}")
    print("🧪 TESTING QUERY EXPANSION")
    print("=" * 80)

    # Use the SAME hybrid_rag that already has documents loaded
    query_expansion_rag = QueryExpansionRAG(hybrid_rag)  # ✅ Fixed!

    # Test expansion queries
    expansion_test_queries = [
        "How much time off do I get?",
        "How to change my login?",
        "What are the leave rules?"
    ]

    for query in expansion_test_queries:
        print(f"\n{'*' * 50}")
        print(f"🔍 Testing Query Expansion")
        print(f"{'*' * 50}")

        # Compare regular hybrid search vs expanded search
        regular_results = hybrid_rag.hybrid_search(query, top_k=3)
        expanded_results = query_expansion_rag.search_with_expansion(query, top_k=3)

        print(f"\n📊 REGULAR HYBRID SEARCH:")
        for i, result in enumerate(regular_results, 1):
            print(f"   {i}. Score: {result['combined_score']:.3f}")
            print(f"      Text: {result['text'][:70]}...")

        print(f"\n🚀 EXPANDED SEARCH RESULTS:")
        for i, result in enumerate(expanded_results['results'], 1):
            score_key = 'combined_score' if 'combined_score' in result else 'similarity_score'
            print(f"   {i}. Score: {result[score_key]:.3f}")
            print(f"      Text: {result['text'][:70]}...")

        # Test Query Expansion with error handling
        print(f"\n{'=' * 80}")
        print("🧪 TESTING QUERY EXPANSION")
        print("=" * 80)

        query_expansion_rag = QueryExpansionRAG(hybrid_rag)

        expansion_test_queries = [
            "How much time off do I get?",
            "How to change my login?",
            "What are the leave rules?"
        ]

        for query in expansion_test_queries:
            print(f"\n{'*' * 50}")
            print(f"🔍 Testing Query Expansion")
            print(f"{'*' * 50}")

            # Debug what search_with_expansion returns
            expanded_results = query_expansion_rag.search_with_expansion(query, top_k=3)

            print(f"🐛 DEBUG: expanded_results type = {type(expanded_results)}")
            print(f"🐛 DEBUG: expanded_results = {expanded_results}")

            # Handle both list and dictionary formats
            if isinstance(expanded_results, dict):
                # Expected dictionary format
                results_list = expanded_results.get('results', [])
                print(f"\n🚀 EXPANDED SEARCH RESULTS (Dictionary format):")
            else:
                # If it's a list directly
                results_list = expanded_results
                print(f"\n🚀 EXPANDED SEARCH RESULTS (List format):")

            # Display results
            for i, result in enumerate(results_list, 1):
                # Handle different result formats
                if isinstance(result, dict):
                    score_key = 'combined_score' if 'combined_score' in result else 'similarity_score'
                    score = result.get(score_key, 0.0)
                    text = result.get('text', 'No text available')
                    print(f"   {i}. Score: {score:.3f}")
                    print(f"      Text: {text[:70]}...")
                else:
                    print(f"   {i}. Unexpected result format: {result}")


def test_alpha_tuning():
    """Test different alpha values for hybrid search"""
    print(f"\n{'=' * 80}")
    print("🧪 TESTING ALPHA TUNING (Dense vs Sparse Weight)")
    print("=" * 80)

    # Initialize and add documents
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
            'text': 'PTO portal submission requires 14 days advance notice.',
            'source': 'HR_Policy',
            'doc_type': 'policy'
        }
    ]

    hybrid_rag.add_documents(documents)

    # Test query
    query = "PTO portal advance notice"

    # Test different alpha values
    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]

    for alpha in alpha_values:
        print(
            f"\n🎯 Alpha = {alpha} ({'Pure Sparse' if alpha == 0 else 'Pure Dense' if alpha == 1 else f'{int(alpha * 100)}% Dense, {int((1 - alpha) * 100)}% Sparse'})")
        results = hybrid_rag.hybrid_search(query, top_k=2, alpha=alpha)

        for i, result in enumerate(results, 1):
            print(f"   {i}. Combined: {result['combined_score']:.3f} "
                  f"(D: {result['dense_score']:.3f}, S: {result['sparse_score']:.3f})")
            print(f"      Text: {result['text'][:60]}...")


if __name__ == "__main__":
    test_retrieval_methods()
    test_alpha_tuning()
