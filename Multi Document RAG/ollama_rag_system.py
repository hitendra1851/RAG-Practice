# ollama_rag_system.py
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List, Dict, Any
import time


class OllamaRAGSystem:
    def __init__(self, model_name: str = "llama2", embedding_model: str = "nomic-embed-text"):
        """
        Initialize Ollama RAG System

        Args:
            model_name: Model for text generation (llama2, mistral, codellama, etc.)
            embedding_model: Model for embeddings (nomic-embed-text, all-minilm, etc.)
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_url = "http://localhost:11434"
        self.knowledge_base = []
        self.document_metadata = {}

        # Test connection and models
        self._check_ollama_connection()
        self._ensure_models_available()

    def _check_ollama_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                print(f"✅ Connected to Ollama")
                print(f"📋 Available models: {available_models}")
                return True
            else:
                print(f"❌ Ollama connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("🔧 Please ensure Ollama is installed and running:")
            print("   1. Install from: https://ollama.ai/")
            print("   2. Run: ollama serve")
            return False

    def _ensure_models_available(self):
        """Check and pull required models if needed"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            available_models = [model['name'] for model in response.json().get('models', [])]

            # Check embedding model
            if not any(self.embedding_model in model for model in available_models):
                print(f"📥 Pulling embedding model: {self.embedding_model}")
                self._pull_model(self.embedding_model)

            # Check generation model
            if not any(self.model_name in model for model in available_models):
                print(f"📥 Pulling generation model: {self.model_name}")
                self._pull_model(self.model_name)

        except Exception as e:
            print(f"⚠️ Could not check models: {e}")

    def _pull_model(self, model_name: str):
        """Pull a model from Ollama"""
        try:
            print(f"🔄 Pulling {model_name}... (this may take a few minutes)")
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300  # 5 minutes timeout
            )

            if response.status_code == 200:
                print(f"✅ Successfully pulled {model_name}")
            else:
                print(f"❌ Failed to pull {model_name}: {response.text}")
        except Exception as e:
            print(f"❌ Error pulling {model_name}: {e}")

    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding using Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=30
            )

            if response.status_code == 200:
                embedding = response.json()["embedding"]
                return embedding
            else:
                print(f"❌ Embedding failed: {response.text}")
                # Return random embedding as fallback
                return np.random.randn(384).tolist()  # Default embedding size

        except Exception as e:
            print(f"❌ Embedding request failed: {e}")
            return np.random.randn(384).tolist()

    def add_document(self, content: str, doc_type: str, source: str, metadata: Dict = None):
        """Add document with Ollama embeddings"""

        doc_id = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.document_metadata[doc_id] = {
            "source": source,
            "doc_type": doc_type,
            "created": datetime.now().isoformat(),
            "chunk_count": 0,
            **(metadata or {})
        }

        # Process document into chunks
        chunks = self._chunk_document(content, doc_type)

        # Create embeddings for each chunk
        print(f"🔄 Creating embeddings for {len(chunks)} chunks using {self.embedding_model}...")

        for i, chunk in enumerate(chunks):
            print(f"   Processing chunk {i + 1}/{len(chunks)}...", end='\r')

            embedding = self._create_embedding(chunk)

            chunk_data = {
                "id": f"{doc_id}_chunk_{i}",
                "document_id": doc_id,
                "text": chunk,
                "embedding": embedding,
                "chunk_index": i,
                "source": source,
                "doc_type": doc_type
            }

            self.knowledge_base.append(chunk_data)
            self.document_metadata[doc_id]["chunk_count"] += 1

            # Small delay to avoid overwhelming Ollama
            time.sleep(0.1)

        print(f"\n✅ Added document '{source}' with {len(chunks)} chunks")

    def _chunk_document(self, content: str, doc_type: str) -> List[str]:
        """Smart chunking based on document type"""

        if doc_type == "policy":
            return self._chunk_by_sections(content)
        elif doc_type == "faq":
            return self._chunk_by_qa_pairs(content)
        elif doc_type == "manual":
            return self._chunk_by_procedures(content)
        else:
            return self._chunk_by_sentences(content)

    def _chunk_by_sections(self, content: str) -> List[str]:
        """Policy documents: chunk by sections"""
        sections = []
        current_section = ""

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if (line.isupper() and len(line) > 5) or (line.startswith('SECTION') and ':' in line):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'

        if current_section.strip():
            sections.append(current_section.strip())

        return sections if sections else self._chunk_by_sentences(content)

    def _chunk_by_qa_pairs(self, content: str) -> List[str]:
        """FAQ: each Q&A pair is a chunk"""
        chunks = []
        lines = content.split('\n')
        current_qa = ""

        for line in lines:
            line = line.strip()
            if line.startswith('Q:') and current_qa:
                chunks.append(current_qa.strip())
                current_qa = line + '\n'
            elif line:
                current_qa += line + '\n'

        if current_qa.strip():
            chunks.append(current_qa.strip())

        return chunks if chunks else [content]

    def _chunk_by_procedures(self, content: str) -> List[str]:
        """Manuals: chunk by procedures"""
        chunks = []
        current_chunk = ""

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            if (line.startswith('PROCEDURE') and ':' in line):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else self._chunk_by_sentences(content)

    def _chunk_by_sentences(self, content: str, max_sentences: int = 4) -> List[str]:
        """Fallback: chunk by sentences"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        chunks = []

        for i in range(0, len(sentences), max_sentences):
            chunk = '. '.join(sentences[i:i + max_sentences])
            if chunk:
                chunks.append(chunk + '.')

        return chunks if chunks else [content]

    def search(self, query: str, top_k: int = 5, filter_by: Dict = None) -> List[Dict]:
        """Search using Ollama embeddings"""

        if not self.knowledge_base:
            return []

        # Apply filters
        candidates = self.knowledge_base
        if filter_by:
            candidates = self._apply_filters(candidates, filter_by)

        if not candidates:
            return []

        print(f"🔍 Searching through {len(candidates)} chunks...")

        # Create query embedding
        query_embedding = self._create_embedding(query)

        # Calculate similarities
        similarities = []
        for chunk in candidates:
            try:
                chunk_embedding = np.array(chunk["embedding"])
                query_embedding_np = np.array(query_embedding)

                # Ensure same dimensions
                if len(chunk_embedding) == len(query_embedding_np):
                    similarity = cosine_similarity([query_embedding_np], [chunk_embedding])[0][0]
                    similarities.append((chunk, similarity))
                else:
                    print(f"⚠️ Dimension mismatch: query={len(query_embedding_np)}, chunk={len(chunk_embedding)}")

            except Exception as e:
                print(f"❌ Similarity calculation failed: {e}")
                continue

        # Sort and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in similarities[:top_k]:
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "doc_type": chunk["doc_type"],
                "similarity_score": float(score),
                "document_id": chunk["document_id"]
            })

        return results

    def _apply_filters(self, candidates: List[Dict], filters: Dict) -> List[Dict]:
        """Apply metadata filters"""
        filtered = []
        for chunk in candidates:
            include = True
            for key, value in filters.items():
                if key in chunk and chunk[key] != value:
                    include = False
                    break
            if include:
                filtered.append(chunk)
        return filtered

    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using Ollama LLM"""

        # Build context from retrieved chunks
        context = "\n\n".join([
            f"Source: {chunk['source']}\n{chunk['text']}"
            for chunk in context_chunks
        ])

        prompt = f"""Based on the following company documents, answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Answer based only on the provided context. If the answer is not in the context, say "I don't have that information in the provided documents."

Answer:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error generating response: {response.text}"

        except Exception as e:
            return f"Error: Could not generate response - {e}"

    def query_with_generation(self, query: str, top_k: int = 3) -> Dict:
        """Complete RAG pipeline: search + generate"""

        # Step 1: Search for relevant chunks
        search_results = self.search(query, top_k=top_k)

        if not search_results:
            return {
                "query": query,
                "answer": "I don't have any relevant information to answer your question.",
                "sources": [],
                "search_results": []
            }

        # Step 2: Generate response using top chunks
        print("🤖 Generating response with Ollama...")
        answer = self.generate_response(query, search_results)

        return {
            "query": query,
            "answer": answer,
            "sources": [result["source"] for result in search_results],
            "search_results": search_results
        }

    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = response.json().get('models', []) if response.status_code == 200 else []

            return {
                "generation_model": self.model_name,
                "embedding_model": self.embedding_model,
                "ollama_url": self.ollama_url,
                "available_models": [m['name'] for m in models],
                "total_chunks": len(self.knowledge_base),
                "total_documents": len(self.document_metadata)
            }
        except:
            return {
                "generation_model": self.model_name,
                "embedding_model": self.embedding_model,
                "total_chunks": len(self.knowledge_base),
                "total_documents": len(self.document_metadata)
            }