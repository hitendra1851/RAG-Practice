# test_advanced_reranking.py
from hybrid_retrieval_system import HybridRetrievalRAG
from advanced_reranking_system import AdvancedRerankedRAG


def setup_test_system():
    """Setup test system with realistic business documents"""

    # Initialize hybrid RAG
    hybrid_rag = HybridRetrievalRAG()

    # Realistic business documents
    documents = [
        {
            'id': 'hr_001',
            'text': 'New employees receive 15 vacation days in their first year. After 2 years of service, this increases to 20 days.',
            'source': 'Employee_Handbook_2024',
            'doc_type': 'policy'
        },
        {
            'id': 'hr_002',
            'text': 'Senior employees with 5+ years of service receive maximum vacation allowance of 25 days per year.',
            'source': 'Senior_Benefits_Guide',
            'doc_type': 'policy'
        },
        {
            'id': 'hr_003',
            'text': 'Vacation policy overview: All full-time employees are eligible for paid time off based on tenure and position level.',
            'source': 'Benefits_Overview',
            'doc_type': 'overview'
        },
        {
            'id': 'hr_004',
            'text': 'Vacation request procedures: Submit requests 2 weeks in advance through HR portal. Manager approval required.',
            'source': 'HR_Procedures',
            'doc_type': 'procedure'
        },
        {
            'id': 'it_001',
            'text': 'Password reset instructions: Visit company portal, click "Forgot Password", enter employee ID, follow email instructions.',
            'source': 'IT_Support_FAQ',
            'doc_type': 'faq'
        },
        {
            'id': 'it_002',
            'text': 'VPN access setup: Download Cisco AnyConnect client, use network credentials, connect to vpn.company.com.',
            'source': 'VPN_Setup_Guide',
            'doc_type': 'guide'
        },
        {
            'id': 'finance_001',
            'text': 'Expense reporting: Submit receipts within 30 days via expense portal. Reimbursement processed within 5 business days.',
            'source': 'Finance_Policies',
            'doc_type': 'policy'
        },
        {
            'id': 'safety_001',
            'text': 'Emergency evacuation procedures: In case of fire alarm, exit via nearest stairwell, gather at designated meeting point.',
            'source': 'Safety_Manual',
            'doc_type': 'procedure'
        }
    ]

    hybrid_rag.add_documents(documents)

    # Initialize advanced reranking system
    reranked_rag = AdvancedRerankedRAG(hybrid_rag)

    return reranked_rag


def test_reranking_scenarios():
    """Test reranking on different types of queries"""

    reranked_rag = setup_test_system()

    # Test scenarios that showcase reranking strengths
    test_scenarios = [
        {
            'query': 'How many vacation days do senior employees get?',
            'expected_answer_contains': '25 days',
            'description': 'Specific question requiring precise answer'
        },
        {
            'query': 'What is the vacation policy for new hires?',
            'expected_answer_contains': '15 vacation days',
            'description': 'New employee specific information'
        },
        {
            'query': 'How do I reset my password?',
            'expected_answer_contains': 'company portal',
            'description': 'Procedural question with exact steps'
        },
        {
            'query': 'VPN setup instructions',
            'expected_answer_contains': 'Cisco AnyConnect',
            'description': 'Technical guidance query'
        }
    ]

    print(f"🧪 TESTING RERANKING SCENARIOS")
    print(f"{'=' * 80}")

    for scenario in test_scenarios:
        query = scenario['query']
        expected = scenario['expected_answer_contains']
        description = scenario['description']

        print(f"\n📋 Scenario: {description}")
        print(f"Query: '{query}'")
        print(f"Expected to find: '{expected}'")
        print("-" * 60)

        # Test with explanation
        results = reranked_rag.search_with_reranking(query, top_k=3, explain=True)

        # Check if expected answer is in top result
        if results:
            top_result = results[0]
            if expected.lower() in top_result['text'].lower():
                print(f"✅ SUCCESS: Expected answer found in top result!")
            else:
                print(f"❌ MISS: Expected answer not in top result")
                print(f"Top result: {top_result['text'][:100]}...")

        print("\n" + "=" * 80)


def test_reranker_comparison():
    """Compare different reranker models"""

    reranked_rag = setup_test_system()

    test_queries = [
        "How many vacation days for senior staff?",
        "Password reset procedure",
        "VPN connection setup"
    ]

    for query in test_queries:
        reranked_rag.compare_rerankers(query, top_k=3)


def test_performance_benchmark():
    """Benchmark reranking performance"""

    reranked_rag = setup_test_system()

    benchmark_queries = [
        "vacation days for new employees",
        "senior employee benefits",
        "password reset instructions",
        "VPN setup guide",
        "expense reporting process",
        "emergency procedures",
        "time off policy",
        "IT support contact"
    ]

    metrics = reranked_rag.benchmark_reranking_impact(benchmark_queries)

    return metrics


if __name__ == "__main__":
    print("🚀 Starting Advanced Reranking Tests...")

    # Test 1: Reranking scenarios
    test_reranking_scenarios()

    # Test 2: Compare reranker models
    print(f"\n{'=' * 80}")
    print("🔧 TESTING DIFFERENT RERANKER MODELS")
    print(f"{'=' * 80}")
    test_reranker_comparison()

    # Test 3: Performance benchmark
    print(f"\n{'=' * 80}")
    print("📊 PERFORMANCE BENCHMARK")
    print(f"{'=' * 80}")
    metrics = test_performance_benchmark()

    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Queries improved by reranking: {metrics['average_improvements']['positive_percentage']:.1f}%")
    print(f"   Average score improvement: {metrics['average_improvements']['mean']:+.3f}")
    print(f"   Total time overhead: {metrics['timing']['average_reranking']:.3f}s per query")

    if metrics['average_improvements']['positive_percentage'] > 70:
        print(f"✅ Reranking is working well!")
    else:
        print(f"⚠️ Consider tuning reranking parameters or trying different models")