# ollama_integration.py
import requests
from complete_answer_system import ProductionAnswerGenerator


class OllamaAnswerSystem:
    def __init__(self, retrieval_system, model: str = "llama2", host: str = "http://localhost:11434"):
        self.answer_generator = ProductionAnswerGenerator(
            retrieval_system,
            OllamaLLMClient(model, host)
        )

    def generate_answer(self, query: str, **options):
        return self.answer_generator.generate_answer(query, options)


class OllamaLLMClient:
    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        try:
            response = requests.post(f"{self.host}/api/generate", json={
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            })

            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            raise Exception(f"Ollama connection error: {str(e)}")

# Usage example:
# ollama_system = OllamaAnswerSystem(your_retrieval_system, "llama2")
# result = ollama_system.generate_answer("How many sick days do new employees get?")