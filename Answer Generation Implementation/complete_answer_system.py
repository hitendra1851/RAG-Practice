# complete_answer_system.py
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import hashlib


@dataclass
class AnswerResult:
    """Structured answer result with all metadata"""
    query: str
    answer: str
    confidence_score: float
    sources: List[Dict]
    citations: List[Dict]
    answer_type: str
    generation_time: float
    metadata: Dict


class ProductionAnswerGenerator:
    """Production-ready answer generation system"""

    def __init__(self, retrieval_system, llm_client=None):
        self.retrieval_system = retrieval_system
        self.llm_client = llm_client

        # Configuration
        self.config = {
            'max_context_length': 4000,
            'max_answer_length': 800,
            'confidence_threshold': 0.6,
            'citation_required': True,
            'max_sources': 3,
            'temperature': 0.1  # Low for factual accuracy
        }

        # Initialize prompt templates
        self.prompt_templates = self._init_prompt_templates()

        # Answer quality rules
        self.quality_rules = self._init_quality_rules()

        print("✅ Production Answer Generator initialized")

    def generate_answer(self, query: str, options: Dict = None) -> AnswerResult:
        """Generate complete answer with full pipeline"""

        start_time = time.time()
        options = options or {}

        # Override config with options
        config = {**self.config, **options}

        try:
            # Step 1: Validate query
            if not query or len(query.strip()) < 3:
                return self._create_error_response(query, "Query too short", start_time)

            # Step 2: Retrieve and rerank documents
            search_results = self.retrieval_system.search_with_reranking(
                query,
                top_k=config['max_sources'] * 2
            )

            if not search_results:
                return self._create_no_results_response(query, start_time)

            # Step 3: Select and prepare context
            context_info = self._prepare_context(query, search_results, config)

            # Step 4: Classify query and select prompt
            answer_type = self._classify_query_type(query)
            prompt = self._build_prompt(query, context_info, answer_type)

            # Step 5: Generate raw answer
            raw_answer = self._generate_with_llm(prompt, config)

            # Step 6: Post-process and extract citations
            processed_answer = self._post_process_answer(raw_answer, context_info)

            # Step 7: Calculate confidence and apply quality rules
            confidence = self._calculate_confidence(query, processed_answer, context_info)
            enhanced_answer = self._enhance_answer_quality(
                processed_answer, query, context_info, answer_type
            )

            # Step 8: Build final result
            generation_time = time.time() - start_time

            return AnswerResult(
                query=query,
                answer=enhanced_answer['text'],
                confidence_score=confidence,
                sources=context_info['sources'],
                citations=enhanced_answer['citations'],
                answer_type=answer_type,
                generation_time=generation_time,
                metadata={
                    'context_length': context_info['total_length'],
                    'sources_used': len(context_info['sources']),
                    'prompt_length': len(prompt),
                    'raw_answer_length': len(raw_answer),
                    'enhancement_applied': enhanced_answer.get('enhancements', []),
                    'timestamp': datetime.now().isoformat()
                }
            )

        except Exception as e:
            return self._create_error_response(query, str(e), start_time)

    def _init_prompt_templates(self) -> Dict[str, str]:
        """Initialize all prompt templates"""
        return {
            'factual': """You are a helpful assistant providing factual information from company documents.

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Provide a direct, accurate answer based ONLY on the context documents
- Include specific numbers, dates, or details when relevant
- Use [Source X] to cite where information comes from
- Be concise but complete
- If information is missing, say "I don't have complete information about..."

ANSWER:""",

            'procedural': """You are a helpful assistant providing step-by-step instructions from company documents.

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Provide clear, actionable steps based ONLY on the context documents
- Number the steps in logical order
- Include important warnings or prerequisites
- Use [Source X] to cite where each step comes from
- If steps are unclear, say "Please refer to [Source X] for complete details"

ANSWER:""",

            'comparative': """You are a helpful assistant helping users understand comparisons from company documents.

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Compare items clearly based ONLY on the context documents
- Highlight key differences and similarities
- Use specific details from the documents
- Use [Source X] to cite comparison information
- Organize comparison logically

ANSWER:""",

            'general': """You are a helpful assistant providing information from company documents.

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Provide helpful, accurate information based ONLY on the context documents
- Be comprehensive but concise
- Use [Source X] to cite where information comes from
- If context doesn't fully answer the question, explain what information is available

ANSWER:"""
        }

    def _init_quality_rules(self) -> List:
        """Initialize answer quality enhancement rules"""
        return [
            self._add_context_clarification,
            self._improve_number_formatting,
            self._add_helpful_context,
            self._ensure_citation_quality
        ]

    def _prepare_context(self, query: str, search_results: List[Dict], config: Dict) -> Dict:
        """Prepare context from search results"""

        selected_sources = []
        combined_text = ""
        current_length = 0

        # Sort by rerank score
        sorted_results = sorted(search_results,
                                key=lambda x: x.get('rerank_score', x.get('similarity_score', 0)),
                                reverse=True)

        for i, result in enumerate(sorted_results):
            if len(selected_sources) >= config['max_sources']:
                break

            source_text = result.get('text', '')

            # Check context length limit
            if current_length + len(source_text) > config['max_context_length']:
                remaining_space = config['max_context_length'] - current_length
                if remaining_space > 150:  # Only if meaningful space left
                    source_text = source_text[:remaining_space - 10] + "..."
                else:
                    break

            source_id = i + 1
            selected_sources.append({
                'id': source_id,
                'text': source_text,
                'source': result.get('source', f'Document_{source_id}'),
                'doc_type': result.get('doc_type', 'document'),
                'score': result.get('rerank_score', result.get('similarity_score', 0))
            })

            combined_text += f"\n--- Source {source_id} ({result.get('source', 'Unknown')}) ---\n"
            combined_text += source_text + "\n"
            current_length += len(source_text)

        return {
            'sources': selected_sources,
            'combined_text': combined_text.strip(),
            'total_length': current_length
        }

    def _classify_query_type(self, query: str) -> str:
        """Classify query to select appropriate prompt template"""

        query_lower = query.lower().strip()

        # Procedural (how-to)
        procedural_indicators = ['how to', 'how do', 'how can', 'steps to', 'procedure', 'process']
        if any(indicator in query_lower for indicator in procedural_indicators):
            return 'procedural'

        # Comparative
        comparative_indicators = ['compare', 'comparison', 'difference', 'vs', 'versus', 'better than']
        if any(indicator in query_lower for indicator in comparative_indicators):
            return 'comparative'

        # Factual (specific facts)
        factual_starters = ['what is', 'when is', 'where is', 'who is', 'how many', 'how much']
        if any(query_lower.startswith(starter) for starter in factual_starters):
            return 'factual'

        return 'general'

    def _build_prompt(self, query: str, context_info: Dict, answer_type: str) -> str:
        """Build the final prompt for LLM generation"""

        template = self.prompt_templates.get(answer_type, self.prompt_templates['general'])

        return template.format(
            context=context_info['combined_text'],
            query=query
        )

    def _generate_with_llm(self, prompt: str, config: Dict) -> str:
        """Generate answer using LLM with error handling"""

        if self.llm_client is None:
            # Fallback for testing
            return self._generate_fallback_answer(prompt)

        try:
            response = self.llm_client.generate(
                prompt,
                max_tokens=config['max_answer_length'],
                temperature=config['temperature']
            )
            return response.strip() if response else "Unable to generate answer."

        except Exception as e:
            print(f"⚠️ LLM generation error: {e}")
            return f"Error generating answer: {str(e)}"

    def _generate_fallback_answer(self, prompt: str) -> str:
        """Generate a basic answer for testing without LLM"""

        # Extract query from prompt
        query_match = re.search(r'USER QUESTION: (.+)', prompt)
        query = query_match.group(1) if query_match else "the question"

        # Extract first source text for basic answer
        source_match = re.search(r'--- Source 1 \(.+?\) ---\n(.+?)\n', prompt, re.DOTALL)
        if source_match:
            source_text = source_match.group(1)[:200] + "..."
            return f"Based on the available information: {source_text} [Source 1]"

        return "I don't have enough information to answer this question."

    def _post_process_answer(self, raw_answer: str, context_info: Dict) -> Dict:
        """Post-process raw answer to extract citations and improve quality"""

        # Extract citations
        citation_pattern = r'\[Source (\d+)\]'
        found_citations = []

        for match in re.finditer(citation_pattern, raw_answer):
            source_num = int(match.group(1))
            if 1 <= source_num <= len(context_info['sources']):
                source = context_info['sources'][source_num - 1]
                found_citations.append({
                    'source_id': source_num,
                    'source_name': source['source'],
                    'doc_type': source['doc_type'],
                    'score': source['score']
                })

        # Remove duplicate citations
        unique_citations = []
        seen_ids = set()
        for citation in found_citations:
            if citation['source_id'] not in seen_ids:
                unique_citations.append(citation)
                seen_ids.add(citation['source_id'])

        # Clean up answer text
        cleaned_answer = raw_answer.strip()

        # Remove incomplete sentences at the end
        if cleaned_answer.endswith('...') and not cleaned_answer.endswith('...['):
            sentences = cleaned_answer.rsplit('.', 1)
            if len(sentences) > 1 and len(sentences[1].strip()) < 10:
                cleaned_answer = sentences[0] + '.'

        return {
            'text': cleaned_answer,
            'citations': unique_citations,
            'raw_text': raw_answer
        }

    def _calculate_confidence(self, query: str, processed_answer: Dict, context_info: Dict) -> float:
        """Calculate confidence score for the generated answer"""

        if not context_info['sources']:
            return 0.0

        # Base confidence from source quality
        avg_source_score = sum(s['score'] for s in context_info['sources']) / len(context_info['sources'])
        base_confidence = min(avg_source_score * 0.9, 0.85)  # Cap base confidence

        answer_text = processed_answer['text'].lower()
        adjustments = 0.0

        # Positive indicators
        if len(processed_answer['citations']) > 0:
            adjustments += 0.1

        if len(processed_answer['text']) > 50:
            adjustments += 0.05

        # Check for uncertainty phrases
        uncertainty_phrases = [
            'i don\'t know', 'not sure', 'unclear', 'might be', 'possibly',
            'i don\'t have', 'not enough information', 'missing'
        ]
        if not any(phrase in answer_text for phrase in uncertainty_phrases):
            adjustments += 0.05

        # Negative indicators
        if len(processed_answer['text']) < 20:
            adjustments -= 0.2

        if any(phrase in answer_text for phrase in uncertainty_phrases):
            adjustments -= 0.15

        # Question type specific adjustments
        if 'how many' in query.lower() or 'how much' in query.lower():
            if any(char.isdigit() for char in answer_text):
                adjustments += 0.1  # Has numbers for quantity question
            else:
                adjustments -= 0.1  # No numbers for quantity question

        final_confidence = max(0.0, min(1.0, base_confidence + adjustments))
        return round(final_confidence, 3)

    def _enhance_answer_quality(self, processed_answer: Dict, query: str,
                                context_info: Dict, answer_type: str) -> Dict:
        """Apply quality enhancement rules"""

        enhanced = processed_answer.copy()
        enhancements_applied = []

        for rule in self.quality_rules:
            try:
                result = rule(enhanced, query, context_info, answer_type)
                if result['text'] != enhanced['text']:
                    enhanced = result
                    enhancements_applied.append(rule.__name__)
            except Exception as e:
                print(f"⚠️ Enhancement rule {rule.__name__} failed: {e}")

        enhanced['enhancements'] = enhancements_applied
        return enhanced

    def _add_context_clarification(self, answer: Dict, query: str,
                                   context_info: Dict, answer_type: str) -> Dict:
        """Add helpful context clarification"""

        answer_text = answer['text']

        # Add policy date context
        policy_sources = [s for s in context_info['sources'] if 'policy' in s['doc_type'].lower()]
        if policy_sources and 'current' not in answer_text.lower():
            answer_text += " (Based on current company policies)"

        # Add scope clarification for employee questions
        if 'employee' in query.lower() and len(context_info['sources']) > 0:
            source_texts = [s['text'].lower() for s in context_info['sources']]
            if any('full-time' in text for text in source_texts):
                if 'full-time' not in answer_text.lower():
                    answer_text += " This applies to full-time employees."

        return {**answer, 'text': answer_text}

    def _improve_number_formatting(self, answer: Dict, query: str,
                                   context_info: Dict, answer_type: str) -> Dict:
        """Improve formatting of numbers and important details"""

        answer_text = answer['text']

        # Bold important numbers
        number_patterns = [
            (r'(\d+\.?\d*)\s*(days?)', r'**\1** \2'),
            (r'(\d+\.?\d*)\s*(hours?)', r'**\1** \2'),
            (r'(\d+\.?\d*)\s*(years?)', r'**\1** \2'),
            (r'(\$\d+\.?\d*)', r'**\1**'),
            (r'(\d+\.?\d*)%', r'**\1%**')
        ]

        for pattern, replacement in number_patterns:
            answer_text = re.sub(pattern, replacement, answer_text, flags=re.IGNORECASE)

        return {**answer, 'text': answer_text}

    def _add_helpful_context(self, answer: Dict, query: str,
                             context_info: Dict, answer_type: str) -> Dict:
        """Add helpful additional context"""

        answer_text = answer['text']

        # Add related information hints
        if answer_type == 'factual' and len(answer_text) < 100:
            # Check if there's related info in other sources
            unused_sources = context_info['sources'][len(answer['citations']):]
            if unused_sources:
                answer_text += f" For additional details, see {unused_sources[0]['source']}."

        return {**answer, 'text': answer_text}

    def _ensure_citation_quality(self, answer: Dict, query: str,
                                 context_info: Dict, answer_type: str) -> Dict:
        """Ensure citations are properly formatted and complete"""

        answer_text = answer['text']

        # If no citations but should have them, add reference
        if not answer['citations'] and context_info['sources']:
            if not re.search(r'\[Source \d+\]', answer_text):
                answer_text += f" [Source 1]"
                answer['citations'] = [{
                    'source_id': 1,
                    'source_name': context_info['sources'][0]['source'],
                    'doc_type': context_info['sources'][0]['doc_type'],
                    'score': context_info['sources'][0]['score']
                }]

        return {**answer, 'text': answer_text}

    def _create_error_response(self, query: str, error_msg: str, start_time: float) -> AnswerResult:
        """Create structured error response"""
        return AnswerResult(
            query=query,
            answer=f"I apologize, but I encountered an error: {error_msg}",
            confidence_score=0.0,
            sources=[],
            citations=[],
            answer_type='error',
            generation_time=time.time() - start_time,
            metadata={'error': error_msg, 'timestamp': datetime.now().isoformat()}
        )

    def _create_no_results_response(self, query: str, start_time: float) -> AnswerResult:
        """Create response when no relevant documents found"""
        return AnswerResult(
            query=query,
            answer="I couldn't find any relevant information to answer this question in the available documents.",
            confidence_score=0.0,
            sources=[],
            citations=[],
            answer_type='no_results',
            generation_time=time.time() - start_time,
            metadata={'reason': 'no_results', 'timestamp': datetime.now().isoformat()}
        )


# LLM Client Implementations
class OpenAIClient:
    """OpenAI API client wrapper"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        # You'd import openai here and initialize

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        # Implementation would use OpenAI API
        # For demo purposes, return placeholder
        return "OpenAI response would go here based on the prompt"


class OllamaClient:
    """Ollama local LLM client wrapper"""

    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        # Implementation would use Ollama API
        # For demo purposes, return placeholder
        return "Ollama response would go here based on the prompt"


class TestLLMClient:
    """Test LLM client that generates realistic responses for demo"""

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        # Parse the query from prompt
        query_match = re.search(r'USER QUESTION: (.+)', prompt)
        if not query_match:
            return "I cannot determine what question was asked."

        query = query_match.group(1).lower()

        # Generate contextual response based on query type
        if 'sick days' in query and 'new employee' in query:
            return "New employees receive **5** sick days per year, accrued monthly at 0.42 days per month [Source 1]. These days can be used for personal illness or medical appointments."

        elif 'vacation days' in query and 'senior' in query:
            return "Senior employees with 5+ years of service receive **25** vacation days per year [Source 1]. Vacation requests must be submitted 2 weeks in advance."

        elif 'password reset' in query or 'reset password' in query:
            return "To reset your password: 1) Visit the company portal at portal.company.com 2) Click 'Forgot Password' 3) Enter your employee ID 4) Follow the email instructions [Source 1]."

        elif 'expense' in query:
            return "Submit expense receipts through the expense portal within **30 days** of purchase [Source 1]. Include business justification and manager approval. Reimbursement is processed within 5 business days."

        else:
            # Generic response using first source
            source_match = re.search(r'--- Source 1 \(.+?\) ---\n(.+?)\n', prompt, re.DOTALL)
            if source_match:
                source_text = source_match.group(1)[:150]
                return f"Based on the available information: {source_text}... [Source 1]"

            return "I don't have enough information to provide a complete answer to this question."