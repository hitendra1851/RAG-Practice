# openai_integration.py
import openai
from complete_answer_system import ProductionAnswerGenerator


class OpenAIAnswerSystem:
    def __init__(self, retrieval_system, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
        self.answer_generator = ProductionAnswerGenerator(
            retrieval_system,
            OpenAILLMClient(model)
        )

    def generate_answer(self, query: str, **options):
        return self.answer_generator.generate_answer(query, options)


class OpenAILLMClient:
    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

# Usage example:
# openai_system = OpenAIAnswerSystem(your_retrieval_system, "your-api-key")
# result = openai_system.generate_answer("How many sick days do new employees get?")