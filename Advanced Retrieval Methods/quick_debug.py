# quick_debug.py
from hybrid_retrieval_system import HybridRetrievalRAG
from query_expansion import QueryExpansionRAG

hybrid_rag = HybridRetrievalRAG()

documents = [{'id': 'test', 'text': 'Test document', 'source': 'Test', 'doc_type': 'test'}]
hybrid_rag.add_documents(documents)

query_expansion_rag = QueryExpansionRAG(hybrid_rag)
result = query_expansion_rag.search_with_expansion("test")

print(f"Result type: {type(result)}")
print(f"Result: {result}")