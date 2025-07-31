# benchmark_embeddings.py
import time
from sentence_transformers import SentenceTransformer


def benchmark_models():
    test_text = "How many vacation days do employees get based on years of service?"

    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-MiniLM-L6-cos-v1"
    ]

    for model_name in models:
        print(f"\n🧪 Testing {model_name}:")

        # Load model
        start = time.time()
        model = SentenceTransformer(model_name)
        load_time = time.time() - start

        # Create embedding
        start = time.time()
        embedding = model.encode([test_text])
        embed_time = time.time() - start

        print(f"   Load time: {load_time:.2f}s")
        print(f"   Embed time: {embed_time:.3f}s")
        print(f"   Embedding size: {len(embedding[0])}")
        print(f"   Model size: ~{model.get_sentence_embedding_dimension()}D")


if __name__ == "__main__":
    benchmark_models()