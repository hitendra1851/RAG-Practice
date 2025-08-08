# test_complete_answer_system.py
from complete_answer_system import ProductionAnswerGenerator, TestLLMClient
from hybrid_retrieval_system import HybridRetrievalRAG
from advanced_reranking_system import AdvancedRerankedRAG
from typing import List, Dict


def setup_complete_system():
    """Setup the complete RAG pipeline for testing"""

    print("🔄 Setting up complete RAG system...")

    # Initialize base systems
    hybrid_rag = HybridRetrievalRAG()

    # Add comprehensive test documents
    documents = [
        {
            'id': 'hr_001',
            'text': 'New employees receive 5 sick days per year, accrued monthly at 0.42 days per month. Sick days can be used for personal illness, medical appointments, or caring for sick family members. Unused sick days do not roll over to the next year.',
            'source': 'Employee_Handbook_2024',
            'doc_type': 'policy'
        },
        {
            'id': 'hr_002',
            'text': 'Senior employees with 5+ years of continuous service receive 25 vacation days per year. Mid-level employees (2-5 years) receive 20 vacation days. New employees receive 15 vacation days in their first year.',
            'source': 'Vacation_Policy_2024',
            'doc_type': 'policy'
        },
        {
            'id': 'hr_003',
            'text': 'Benefits comparison by employment level: New employees - 15 vacation days, 5 sick days, basic health insurance. Senior employees - 25 vacation days, 7 sick days, premium health insurance plus dental.',
            'source': 'Benefits_Comparison_Guide',
            'doc_type': 'comparison'
        },
        {
            'id': 'it_001',
            'text': 'Password reset procedure: Step 1 - Visit company portal at portal.company.com. Step 2 - Click the "Forgot Password" link. Step 3 - Enter your employee ID and email address. Step 4 - Check your email for reset instructions. Step 5 - Create a new password following security requirements.',
            'source': 'IT_Support_Manual',
            'doc_type': 'procedure'
        },
        {
            'id': 'it_002',
            'text': 'VPN access setup instructions: Download Cisco AnyConnect client from the IT portal. Install the software on your device. Use your network credentials (same as email login). Connect to vpn.company.com. For troubleshooting, contact IT support at extension 1234.',
            'source': 'VPN_Setup_Guide',
            'doc_type': 'guide'
        },
        {
            'id': 'finance_001',
            'text': 'Expense reporting process: All business expenses must be submitted within 30 days of purchase through the online expense portal. Include original receipts, business justification, and manager approval. Reimbursement is processed within 5 business days of approval.',
            'source': 'Finance_Procedures_2024',
            'doc_type': 'procedure'
        },
        {
            'id': 'hr_004',
            'text': 'Remote work policy: Employees may work remotely up to 3 days per week with manager approval. Home office setup stipend of $500 is available. VPN access required for all remote work. Monthly team meetings are mandatory in-person.',
            'source': 'Remote_Work_Policy',
            'doc_type': 'policy'
        },
        {
            'id': 'safety_001',
            'text': 'Emergency evacuation procedure: Upon hearing fire alarm, immediately stop work and exit via nearest stairwell. Do not use elevators. Proceed to designated meeting point in parking lot. Report to floor warden for attendance check. Wait for all-clear before returning to building.',
            'source': 'Emergency_Procedures',
            'doc_type': 'procedure'
        }
    ]

    hybrid_rag.add_documents(documents)

    # Initialize reranking system
    reranked_rag = AdvancedRerankedRAG(hybrid_rag)

    # Initialize answer generation with test LLM
    test_llm = TestLLMClient()
    answer_generator = ProductionAnswerGenerator(reranked_rag, test_llm)

    print("✅ Complete RAG system ready!")
    return answer_generator


def run_comprehensive_tests():
    """Run comprehensive test suite"""

    answer_generator = setup_complete_system()

    # Test scenarios covering all answer types
    test_scenarios = [
        {
            'query': 'How many sick days do new employees get?',
            'expected_type': 'factual',
            'should_contain': ['5 sick days', 'new employees'],
            'should_have_citations': True,
            'description': 'Direct factual question about specific policy'
        },
        {
            'query': 'How do I reset my password?',
            'expected_type': 'procedural',
            'should_contain': ['portal.company.com', 'step', 'employee ID'],
            'should_have_citations': True,
            'description': 'Step-by-step procedure question'
        },
        {
            'query': 'What is the difference between new employee and senior employee vacation benefits?',
            'expected_type': 'comparative',
            'should_contain': ['15 vacation days', '25 vacation days', 'senior'],
            'should_have_citations': True,
            'description': 'Comparison question requiring multiple sources'
        },
        {
            'query': 'Tell me about remote work options',
            'expected_type': 'general',
            'should_contain': ['3 days per week', 'manager approval'],
            'should_have_citations': True,
            'description': 'General information request'
        },
        {
            'query': 'What are the expense reporting deadlines?',
            'expected_type': 'factual',
            'should_contain': ['30 days', 'business days'],
            'should_have_citations': True,
            'description': 'Specific deadline question'
        }
    ]

    print(f"\n🧪 RUNNING COMPREHENSIVE ANSWER GENERATION TESTS")
    print(f"{'=' * 90}")

    test_results = []

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Test {i}/{len(test_scenarios)}: {scenario['description']}")
        print(f"Query: '{scenario['query']}'")
        print(f"Expected type: {scenario['expected_type']}")
        print("-" * 70)

        # Generate answer
        start_time = time.time()
        result = answer_generator.generate_answer(scenario['query'])

        # Evaluate result
        evaluation = evaluate_answer(result, scenario)
        test_results.append({
            'scenario': scenario,
            'result': result,
            'evaluation': evaluation,
            'test_time': time.time() - start_time
        })

        # Display results
        print(f"✅ Generated in {result.generation_time:.3f}s")
        print(
            f"📊 Answer type: {result.answer_type} ({'✅' if result.answer_type == scenario['expected_type'] else '❌'})")
        print(f"🎯 Confidence: {result.confidence_score:.3f}")
        print(f"📝 Citations: {len(result.citations)} ({'✅' if len(result.citations) > 0 else '❌'})")
        print(f"🔍 Content check: {'✅' if evaluation['content_correct'] else '❌'}")
        print(f"⭐ Overall score: {evaluation['score']:.1f}/10")

        print(f"\n💬 Generated Answer:")
        print(f"   {result.answer}")

        if result.citations:
            print(f"\n📚 Sources cited:")
            for citation in result.citations:
                print(f"   - {citation['source_name']} (ID: {citation['source_id']})")

        print("\n" + "=" * 90)

    # Print summary
    print_test_summary(test_results)

    return test_results


def evaluate_answer(result, scenario) -> Dict:
    """Evaluate answer quality against scenario expectations"""

    evaluation = {
        'type_correct': result.answer_type == scenario['expected_type'],
        'has_citations': len(result.citations) > 0,
        'content_correct': all(
            phrase.lower() in result.answer.lower()
            for phrase in scenario['should_contain']
        ),
        'appropriate_length': 30 <= len(result.answer) <= 800,
        'high_confidence': result.confidence_score >= 0.7,
        'fast_generation': result.generation_time <= 1.0
    }

    # Calculate overall score
    score = 0
    if evaluation['type_correct']: score += 2
    if evaluation['content_correct']: score += 3
    if evaluation['has_citations']: score += 2
    if evaluation['appropriate_length']: score += 1
    if evaluation['high_confidence']: score += 1
    if evaluation['fast_generation']: score += 1

    evaluation['score'] = score
    evaluation['max_score'] = 10

    return evaluation


def print_test_summary(test_results: List[Dict]):
    """Print comprehensive test summary"""

    print(f"\n📊 COMPREHENSIVE TEST SUMMARY")
    print(f"{'=' * 60}")

    total_tests = len(test_results)

    # Calculate metrics
    metrics = {
        'type_correct': sum(1 for r in test_results if r['evaluation']['type_correct']),
        'has_citations': sum(1 for r in test_results if r['evaluation']['has_citations']),
        'content_correct': sum(1 for r in test_results if r['evaluation']['content_correct']),
        'high_confidence': sum(1 for r in test_results if r['evaluation']['high_confidence']),
        'fast_generation': sum(1 for r in test_results if r['evaluation']['fast_generation'])
    }

    avg_confidence = sum(r['result'].confidence_score for r in test_results) / total_tests
    avg_generation_time = sum(r['result'].generation_time for r in test_results) / total_tests
    avg_score = sum(r['evaluation']['score'] for r in test_results) / total_tests

    # Print metrics
    print(f"Tests completed: {total_tests}")
    print(
        f"Query type classification: {metrics['type_correct']}/{total_tests} ({metrics['type_correct'] / total_tests * 100:.1f}%)")
    print(
        f"Content accuracy: {metrics['content_correct']}/{total_tests} ({metrics['content_correct'] / total_tests * 100:.1f}%)")
    print(
        f"Citation inclusion: {metrics['has_citations']}/{total_tests} ({metrics['has_citations'] / total_tests * 100:.1f}%)")
    print(
        f"High confidence (≥0.7): {metrics['high_confidence']}/{total_tests} ({metrics['high_confidence'] / total_tests * 100:.1f}%)")
    print(
        f"Fast generation (≤1s): {metrics['fast_generation']}/{total_tests} ({metrics['fast_generation'] / total_tests * 100:.1f}%)")

    print(f"\nAverage confidence: {avg_confidence:.3f}")
    print(f"Average generation time: {avg_generation_time:.3f}s")
    print(f"Average quality score: {avg_score:.1f}/10")

    # Overall assessment
    if avg_score >= 8.0:
        print("\n🎉 Excellent! Answer generation system is performing very well.")
    elif avg_score >= 6.0:
        print("\n👍 Good performance. Some areas for improvement identified.")
    else:
        print("\n⚠️ Answer generation needs optimization. Check individual test results.")


def test_edge_cases():
    """Test edge cases and error handling"""

    answer_generator = setup_complete_system()

    edge_cases = [
        {
            'query': 'nonexistent topic xyz123',
            'description': 'Query with no matching documents'
        },
        {
            'query': 'a',
            'description': 'Very short query (single character)'
        },
        {
            'query': '',
            'description': 'Empty query'
        },
        {
            'query': 'What is the meaning of life and how does it relate to our vacation policy and also quantum physics?',
            'description': 'Very complex multi-part query'
        },
        {
            'query': 'How many sick days vacation policy remote work expense reporting?',
            'description': 'Multiple disconnected topics'
        }
    ]

    print(f"\n🔍 TESTING EDGE CASES")
    print(f"{'=' * 60}")

    for case in edge_cases:
        print(f"\nEdge case: {case['description']}")
        print(f"Query: '{case['query']}'")

        try:
            result = answer_generator.generate_answer(case['query'])
            print(f"✅ Handled gracefully")
            print(f"   Answer type: {result.answer_type}")
            print(f"   Answer: {result.answer[:100]}...")
            print(f"   Confidence: {result.confidence_score}")
            print(f"   Generation time: {result.generation_time:.3f}s")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    import time

    print("🚀 Starting Complete Answer Generation System Tests...")
    print("=" * 80)

    # Test 1: Comprehensive scenarios
    test_results = run_comprehensive_tests()

    # Test 2: Edge cases
    test_edge_cases()

    print(f"\n🎯 ALL TESTS COMPLETED!")
    print("Check the results above to see how your answer generation system performs.")
    print("You can now integrate this with different LLM backends (OpenAI, Ollama, etc.)")