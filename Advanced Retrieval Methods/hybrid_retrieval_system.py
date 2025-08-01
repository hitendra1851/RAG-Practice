# hybrid_retrieval_system.py
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple


class HybridRetrievalRAG:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        # Dense retrieval (semantic)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Sparse retrieval (keyword-based)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )

        self.knowledge_base = []
        self.tfidf_matrix = None
        self.is_fitted = False

    def add_documents(self, documents: List[Dict]):
        """Add documents and build both dense and sparse indexes"""
        self.knowledge_base = documents

        # Extract texts for indexing
        texts = [doc['text'] for doc in documents]

        print("🔄 Building dense embeddings...")
        # Create dense embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Store embeddings with documents
        for doc, embedding in zip(self.knowledge_base, embeddings):
            doc['dense_embedding'] = embedding

        print("🔄 Building sparse TF-IDF index...")
        # Create sparse TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.is_fitted = True

        print(f"✅ Indexed {len(documents)} documents with hybrid search")

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Semantic vector search"""
        query_embedding = self.embedding_model.encode([query])

        similarities = []
        for doc in self.knowledge_base:
            similarity = cosine_similarity(query_embedding, [doc['dense_embedding']])[0][0]
            similarities.append((doc, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Keyword-based TF-IDF search"""
        if not self.is_fitted:
            return []

        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([query])

        # Calculate similarities with all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                results.append((self.knowledge_base[idx], similarities[idx]))

        return results

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict]:
        """
        Combine dense and sparse search results

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense search (0.0 = only sparse, 1.0 = only dense)
        """

        # Get results from both methods
        dense_results = self.dense_search(query, top_k=20)
        sparse_results = self.sparse_search(query, top_k=20)

        # Combine scores using weighted average
        combined_scores = {}

        # Add dense scores
        for doc, score in dense_results:
            doc_id = doc['id']
            combined_scores[doc_id] = {
                'doc': doc,
                'dense_score': score,
                'sparse_score': 0.0,
                'combined_score': alpha * score
            }

        # Add sparse scores
        for doc, score in sparse_results:
            doc_id = doc['id']
            if doc_id in combined_scores:
                combined_scores[doc_id]['sparse_score'] = score
                combined_scores[doc_id]['combined_score'] += (1 - alpha) * score
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'dense_score': 0.0,
                    'sparse_score': score,
                    'combined_score': (1 - alpha) * score
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )

        # Format results
        final_results = []
        for result in sorted_results[:top_k]:
            final_results.append({
                'text': result['doc']['text'],
                'source': result['doc']['source'],
                'doc_type': result['doc']['doc_type'],
                'dense_score': float(result['dense_score']),
                'sparse_score': float(result['sparse_score']),
                'combined_score': float(result['combined_score'])
            })

        return final_results

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Basic vector search - fixed version"""
        if not self.knowledge_base:
            return []

        query_embedding = self.embedding_model.encode([query])

        similarities = []
        for doc in self.knowledge_base:
            # Fix: Use 'dense_embedding' instead of 'embedding'
            similarity = cosine_similarity(query_embedding, [doc['dense_embedding']])[0][0]
            similarities.append({
                'id': doc['id'],
                'text': doc['text'],
                'source': doc['source'],
                'doc_type': doc['doc_type'],
                'similarity_score': float(similarity)
            })

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]