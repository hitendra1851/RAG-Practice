# local_rag_system.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List, Dict, Any


class LocalRAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with local embedding model

        Popular models:
        - "all-MiniLM-L6-v2": Fast, good quality (384 dimensions)
        - "all-mpnet-base-v2": Better quality, slower (768 dimensions)
        - "multi-qa-MiniLM-L6-cos-v1": Optimized for Q&A
        - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual support
        """
        print(f"🔄 Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.knowledge_base = []
        self.document_metadata = {}
        print(f"✅ Model loaded! Embedding dimensions: {self.embedding_model.get_sentence_embedding_dimension()}")

    def add_document(self, content: str, doc_type: str, source: str, metadata: Dict = None):
        """Add document with local embeddings"""

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

        # Create embeddings for all chunks at once (much faster!)
        print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
        chunk_texts = [chunk for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)

        # Store chunks with embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_data = {
                "id": f"{doc_id}_chunk_{i}",
                "document_id": doc_id,
                "text": chunk,
                "embedding": embedding.tolist(),  # Convert numpy to list
                "chunk_index": i,
                "source": source,
                "doc_type": doc_type
            }

            self.knowledge_base.append(chunk_data)
            self.document_metadata[doc_id]["chunk_count"] += 1

        print(f"✅ Added document '{source}' with {len(chunks)} chunks")

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

            if (line.startswith('PROCEDURE') and ':' in line) or \
                    line.upper().startswith(('STEP', 'PHASE', 'SECTION')):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else self._chunk_by_sentences(content)

    def _chunk_by_sentences(self, content: str, max_sentences: int = 3) -> List[str]:
        """Fallback: chunk by sentences"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        chunks = []

        for i in range(0, len(sentences), max_sentences):
            chunk = '. '.join(sentences[i:i + max_sentences])
            if chunk:
                chunks.append(chunk + '.')

        return chunks if chunks else [content]

    def search(self, query: str, top_k: int = 5, filter_by: Dict = None) -> List[Dict]:
        """Search using local embeddings"""

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
        query_embedding = self.embedding_model.encode([query])[0]

        # Calculate similarities
        similarities = []
        for chunk in candidates:
            chunk_embedding = np.array(chunk["embedding"])
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            similarities.append((chunk, similarity))

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

    def get_model_info(self) -> Dict:
        """Get information about the embedding model"""
        return {
            "model_name": self.embedding_model._modules['0'].auto_model.name_or_path,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "max_sequence_length": self.embedding_model.get_max_seq_length(),
            "total_chunks": len(self.knowledge_base),
            "total_documents": len(self.document_metadata)
        }