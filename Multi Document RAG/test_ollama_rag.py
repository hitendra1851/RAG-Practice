# test_ollama_rag.py
from ollama_rag_system import OllamaRAGSystem

hr_policy = """
ACME CORPORATION EMPLOYEE HANDBOOK

SECTION 1: VACATION POLICY
Full-time employees receive paid vacation based on years of service:
- Year 1-2: 15 days annually
- Year 3-5: 20 days annually  
- Year 6+: 25 days annually

Vacation requests must be submitted 14 days in advance through the HR portal.

SECTION 2: SICK LEAVE POLICY
All employees receive 10 sick days per year.
Sick days accumulate up to a maximum of 30 days.
Medical documentation required for sick leave exceeding 3 consecutive days.

SECTION 3: REMOTE WORK POLICY
Employees may work remotely up to 2 days per week with manager approval.
Remote work equipment provided by company IT department.
"""

it_faq = """
Q: How do I reset my network password?
A: Visit https://password.acme.com, enter your employee ID, and follow the email instructions. Password must be 12+ characters.

Q: What software can I install on my work computer?
A: Only pre-approved software from the company catalog. Submit installation requests through ServiceNow.

Q: How do I connect to the VPN?
A: Download Cisco AnyConnect from the IT portal. Use your network credentials. Contact IT helpdesk at extension 2468 for help.

Q: What's the Wi-Fi password for guests?
A: Guest network: "ACME-Guest" | Password: "Welcome2024!" | Valid for 24 hours per session.
"""

safety_manual = """
PROCEDURE 1: FIRE EMERGENCY RESPONSE
1. Immediately stop work and alert others by shouting "FIRE!"
2. Activate nearest fire alarm pull station
3. Evacuate using closest safe exit - DO NOT use elevators
4. Proceed to designated assembly area in north parking lot

PROCEDURE 2: MEDICAL EMERGENCY
1. Call 911 immediately for serious injuries
2. Contact on-site medical personnel at extension 911
3. Do not move injured person unless in immediate danger
4. Provide first aid only if trained and certified
"""



def test_ollama_rag():
    print(f"\n{'=' * 60}")
    print(f"🦙 TESTING OLLAMA RAG SYSTEM")
    print(f"{'=' * 60}")

    # Initialize Ollama RAG system
    try:
        rag = OllamaRAGSystem(
            model_name="llama2",  # For generation
            embedding_model="nomic-embed-text"  # For embeddings
        )
    except Exception as e:
        print(f"❌ Failed to initialize Ollama RAG: {e}")
        return

    # Add documents
    print("\n📁 Adding documents...")
    try:
        rag.add_document(hr_policy, "policy", "HR_Employee_Handbook", {"department": "HR"})
        rag.add_document(it_faq, "faq", "IT_Support_FAQ", {"department": "IT"})
        rag.add_document(safety_manual, "manual", "Safety_Procedures", {"department": "Safety"})
    except Exception as e:
        print(f"❌ Failed to add documents: {e}")
        return

    # Show model info
    print(f"\n🤖 MODEL INFO:")
    model_info = rag.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")

    # Test search only
    print(f"\n🔍 TESTING SEARCH:")
    test_queries = [
        "How many vacation days do I get after 4 years?",
        "How do I reset my password?",
        "What should I do in a fire emergency?",
    ]

    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        results = rag.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"   {i}. [{result['doc_type'].upper()}] {result['source']}")
            print(f"      Score: {result['similarity_score']:.3f}")
            print(f"      Text: {result['text'][:100]}...")

    # Test full RAG pipeline (search + generate)
    print(f"\n🤖 TESTING FULL RAG PIPELINE:")

    rag_queries = [
        "How many vacation days do I get after 4 years?",
        "What's the guest WiFi password?"
    ]

    for query in rag_queries:
        print(f"\n❓ Query: '{query}'")
        try:
            result = rag.query_with_generation(query)

            print(f"🤖 Generated Answer:")
            print(f"   {result['answer']}")
            print(f"📚 Sources: {', '.join(result['sources'])}")

        except Exception as e:
            print(f"❌ RAG generation failed: {e}")


if __name__ == "__main__":
    test_ollama_rag()