# Multi-Document RAG System 🚀

A practical implementation of Retrieval-Augmented Generation (RAG) that actually works with your business documents. Built while learning RAG from scratch - no fancy theoretical stuff, just working code that solves real problems.

## What This Does

Ever asked ChatGPT about your company's vacation policy and got "I don't have access to that information"? This fixes that. It gives AI systems access to your private documents and lets them answer questions based on YOUR data, not just generic training.

**Real example:**
- **Without RAG**: "I don't know your company's return policy"  
- **With RAG**: "Based on your customer service guidelines, orders can be cancelled within 2 hours of placement..."

## Why I Built This

Started with a simple problem: our team was wasting hours searching through policy documents, FAQs, and manuals. Thought there had to be a better way. Turns out there is - RAG systems that can instantly find and understand information across multiple document types.

## What's Included

### 🏗️ Core Systems
- **`advanced_rag_system.py`** - Full-featured RAG with local embeddings (Sentence-Transformers)
- **`ollama_rag_system.py`** - RAG using local Llama models via Ollama (100% private)

### 🧪 Testing & Utilities  
- **`test_multi_doc_rag.py`** - Test with sample HR, IT, and Safety documents
- **`test_ollama_rag.py`** - Test the Ollama-based system
- **`debug_chunks.py`** - See exactly how your documents get chunked
- **`benchmark_embeddings.py`** - Compare different embedding models

### 📄 Sample Documents
- Employee handbook sections
- IT support FAQs  
- Safety procedures
- Customer service knowledge base

## Key Features

✅ **Multiple Document Types** - Handles policies, FAQs, manuals, procedures  
✅ **Smart Chunking** - Different strategies for different document types  
✅ **Local & Cloud Options** - Use OpenAI, local models, or Ollama  
✅ **Metadata Filtering** - Search by department, date, document type  
✅ **Debug Tools** - See what's happening under the hood  
✅ **Production Ready** - Error handling, retries, performance monitoring  

## Quick Start

### Option 1: Local Models (Recommended)
```bash
# Install dependencies
pip install sentence-transformers scikit-learn numpy torch

# Run the system
python test_multi_doc_rag.py
