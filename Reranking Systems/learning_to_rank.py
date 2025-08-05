# learning_to_rank.py
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple

class LearningToRankReranker:
    def __init__(self, base_rag):
        self.base_rag = base_rag
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.ltr_model = RandomForestRegressor(n_estimators=100)
        self.is_trained = False

    def extract_features(self, query: str, candidate) -> np.ndarray:
        """Extract features for learning-to-rank"""

        text = self._extract_text(candidate)

        features = [
            # Original hybrid scores
            self._extract_score(candidate, 'dense_score'),
            self._extract_score(candidate, 'sparse_score'),
            self._extract_score(candidate, 'combined_score'),

            # Text features
            len(text.split()),  # Document length
            len(query.split()),  # Query length

            # Overlap features
            query_words = set(query.lower().split())
        doc_words = set(text.lower().split())
        len(query_words.intersection(doc_words)) / len(query_words),  # Query coverage

            # Cross-encoder score
        self.cross_encoder.predict([[query, text]])[0]]

        return np.array(features)

    def train_ltr_model(self, training_queries: List[Dict]):
        """Train learning-to-rank model on labeled data"""

        X_features = []
        y_relevance = []

        for item in training_queries:
            query = item['query']
            candidates = self.base_rag.hybrid_search(query, top_k=20)

            for candidate in candidates:
                features = self.extract_features(query, candidate)
                X_features.append(features)

                # Use relevance labels (you'd need to create these)
                relevance = item.get('relevance_scores', {}).get(
                    self._extract_field(candidate, 'id'), 0
                )
                y_relevance.append(relevance)

        X = np.array(X_features)
        y = np.array(y_relevance)

        self.ltr_model.fit(X, y)
        self.is_trained = True
        print(f"✅ Trained LTR model on {len(X)} examples")

    def ltr_rerank(self, query: str, top_k: int = 5):
        """Rerank using trained learning-to-rank model"""

        if not self.is_trained:
            raise ValueError("LTR model not trained yet!")

        candidates = self.base_rag.hybrid_search(query, top_k=20)

        # Extract features and predict relevance
        reranked_results = []
        for candidate in candidates:
            features = self.extract_features(query, candidate)
            ltr_score = self.ltr_model.predict([features])[0]

            reranked_results.append({
                'text': self._extract_text(candidate),
                'source': self._extract_field(candidate, 'source'),
                'ltr_score': float(ltr_score),
                'original_score': self._extract_score(candidate, 'combined_score')
            })

        reranked_results.sort(key=lambda x: x['ltr_score'], reverse=True)
        return reranked_results[:top_k]