# test_local_rag.py
from advanced_rag_system import LocalRAGSystem


def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()


# Sample documents (same as before)
files = {
    "hr_policy": r"docs\hr_policy.txt",
    "it_faq": r"docs\it_faq.txt",
    "safety_manual": r"docs\safety_manual.txt",
    "customer_service": r"docs\Customer_Service_Knowledge_Base.txt",
}

hr_policy = read_file(files["hr_policy"])
it_faq = read_file(files["it_faq"])
safety_manual = read_file(files["safety_manual"])
customer_service = read_file(files["customer_service"])


def test_local_rag():
    # Test different embedding models
    models_to_test = [
        "all-MiniLM-L6-v2",  # Fast & good
        "all-mpnet-base-v2",  # Better quality
        "multi-qa-MiniLM-L6-cos-v1"  # Optimized for Q&A
    ]

    for model_name in models_to_test[:1]:  # Test first model only
        print(f"\n{'=' * 60}")
        print(f"🧪 TESTING MODEL: {model_name}")
        print(f"{'=' * 60}")

        # Initialize RAG system

        rag = LocalRAGSystem(model_name=model_name)

        # Add documents
        print("\n📁 Adding documents...")
        rag.add_document(hr_policy, "policy", "HR_Employee_Handbook", {"department": "HR"})
        rag.add_document(it_faq, "faq", "IT_Support_FAQ", {"department": "IT"})
        rag.add_document(safety_manual, "manual", "Safety_Procedures", {"department": "Safety"})
        rag.add_document(customer_service, "faq", "Customer_Service_Knowledge_Base", {"department": "Customer Service"})

        # Show model info
        print(f"\n🤖 MODEL INFO:")
        model_info = rag.get_model_info()
        for key, value in model_info.items():
            print(f"   {key}: {value}")

        # Test queries
        test_queries = [
            "How many vacation days do I get after 4 years?",
            "How do I reset my password?",
            "What should I do in a fire emergency?",
            "Can I work remotely?",
            "What's the guest WiFi password?",
            "How can customer cancel the order?",
            "What is the Refund Process time?"
        ]

        print(f"\n🔍 TESTING SEARCH:")
        for query in test_queries:
            print(f"\n❓ Query: '{query}'")
            results = rag.search(query, top_k=2)

            for i, result in enumerate(results, 1):
                print(f"   {i}. [{result['doc_type'].upper()}] {result['source']}")
                print(f"      Score: {result['similarity_score']:.3f}")
                print(f"      Text: {result['text'][:100]}...")


if __name__ == "__main__":
    test_local_rag()
