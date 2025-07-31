# debug_chunks.py
from advanced_rag_system import LocalRAGSystem
def debug_chunks(rag_system, doc_type=None):
    """Debug function to see what chunks were created"""
    print(f"\n🔍 DEBUG: Chunks created from your documents:")
    print("=" * 80)

    for i, chunk in enumerate(rag_system.knowledge_base):
        if doc_type and chunk["doc_type"] != doc_type:
            continue

        print(f"\n📄 CHUNK {i + 1}:")
        print(f"   Document: {chunk['source']}")
        print(f"   Type: {chunk['doc_type']}")
        print(f"   Chunk Index: {chunk['chunk_index']}")
        print(f"   Text Length: {len(chunk['text'])} characters")
        print(f"   Text Preview: {chunk['text'][:200]}...")
        print("-" * 60)


# Add this to your test script
def test_with_debug():
    # Your existing RAG setup...
    rag = LocalRAGSystem("all-MiniLM-L6-v2")  # or OllamaRAGSystem()

    # Add your customer service document
    customer_service_kb = """
    CUSTOMER SERVICE KNOWLEDGE BASE

    TOPIC: ORDER PROCESSING

    Issue: Customer wants to cancel order
    Resolution: Orders can be cancelled within 2 hours of placement. After 2 hours, order may be in fulfillment. Check order status in system. If "Processing" - contact warehouse at ext. 1234. If "Shipped" - advise customer to refuse delivery.

    Issue: Customer reporting damaged product
    Resolution: Apologize for inconvenience. Verify order number and product details. Create return label immediately. Offer replacement or refund. Escalate to manager if order value exceeds $500.

    TOPIC: BILLING INQUIRIES

    Issue: Customer doesn't recognize charge
    Resolution: Verify customer identity with order email. Explain charges including shipping and tax. If customer still disputes, escalate to billing department.

    Issue: Refund processing time
    Resolution: Refunds process within 3-5 business days to original payment method. Credit card refunds may take additional 1-2 billing cycles to appear. Provide refund confirmation number for tracking.
    """

    rag.add_document(customer_service_kb, "faq", "Customer_Service_Knowledge_Base", {"department": "Customer_Service"})

    # DEBUG: See what chunks were created
    debug_chunks(rag, doc_type="faq")

    # Test search
    results = rag.search("How can customer cancel the order?", top_k=3)
    print(f"\n🔍 SEARCH RESULTS:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['similarity_score']:.3f}")
        print(f"   Text: {result['text']}")
        print()


test_with_debug()