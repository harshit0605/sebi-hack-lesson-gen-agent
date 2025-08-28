"""
LangGraph-based TranslationOrchestrator Agent using StateGraph patterns.
Provides LLM-based translation with caching and quality scoring for financial content.
"""
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import asyncio
import logging
import hashlib

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from base_agent import (
    BaseLangGraphAgent,
    TranslationState,
    LangGraphConfig,
    AgentStatus
)
from llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

class TranslationOrchestratorAgent(BaseLangGraphAgent):
    """LangGraph-based TranslationOrchestrator with LLM translation and caching."""
    
    def __init__(self, llm_adapter: LLMAdapter, config: Optional[LangGraphConfig] = None):
        super().__init__("TranslationOrchestrator", config)
        self.llm_adapter = llm_adapter
        self.semaphore = asyncio.Semaphore(3)  # Concurrent translation limit
        
        # Translation cache (in production, use Redis or persistent storage)
        self.translation_cache = {}
        
        # Supported languages with financial terminology mappings
        self.language_configs = {
            'hi': {
                'name': 'Hindi',
                'financial_terms': {
                    'investment': 'निवेश',
                    'portfolio': 'पोर्टफोलियो', 
                    'dividend': 'लाभांश',
                    'mutual fund': 'म्यूचुअल फंड',
                    'stock': 'स्टॉक',
                    'bond': 'बांड',
                    'risk': 'जोखिम',
                    'return': 'रिटर्न'
                }
            },
            'ta': {
                'name': 'Tamil',
                'financial_terms': {
                    'investment': 'முதலீடு',
                    'portfolio': 'போர்ட்ஃபோலியோ',
                    'dividend': 'பங்கு லாபம்',
                    'mutual fund': 'பரஸ்பர நிதி',
                    'stock': 'பங்கு',
                    'bond': 'பத்திரம்',
                    'risk': 'ஆபத்து',
                    'return': 'வருமானம்'
                }
            },
            'te': {
                'name': 'Telugu',
                'financial_terms': {
                    'investment': 'పెట్టుబడి',
                    'portfolio': 'పోర్ట్‌ఫోలియో',
                    'dividend': 'డివిడెంట్',
                    'mutual fund': 'మ్యూచువల్ ఫండ్',
                    'stock': 'స్టాక్',
                    'bond': 'బాండ్',
                    'risk': 'ప్రమాదం',
                    'return': 'రిటర్న్'
                }
            }
        }
    
    def build_graph(self) -> StateGraph:
        """Build the TranslationOrchestrator LangGraph workflow."""
        
        def prepare_translation(state: TranslationState) -> TranslationState:
            """Prepare translation request and validate inputs."""
            source_text = state.get('source_text', '')
            target_languages = state.get('target_languages', [])
            
            if not source_text:
                state['messages'].append(
                    AIMessage(content="No source text provided for translation")
                )
                state['status'] = AgentStatus.FAILED
                state['last_error'] = "No source text provided"
                return state
            
            if not target_languages:
                state['messages'].append(
                    AIMessage(content="No target languages specified")
                )
                state['status'] = AgentStatus.FAILED
                state['last_error'] = "No target languages specified"
                return state
            
            # Initialize translation tracking
            state['translations'] = {}
            state['quality_scores'] = {}
            state['cached_translations'] = {}
            
            # Check cache for existing translations
            source_hash = self._generate_text_hash(source_text)
            cached_count = 0
            
            for lang in target_languages:
                cache_key = f"{source_hash}_{lang}"
                if cache_key in self.translation_cache:
                    state['cached_translations'][lang] = self.translation_cache[cache_key]['text']
                    state['translations'][lang] = self.translation_cache[cache_key]['text']
                    state['quality_scores'][lang] = self.translation_cache[cache_key]['quality_score']
                    cached_count += 1
            
            state['messages'].append(
                AIMessage(content=f"Prepared translation for {len(target_languages)} languages. Found {cached_count} cached translations.")
            )
            
            return state
        
        async def perform_translations(state: TranslationState) -> TranslationState:
            """Perform LLM-based translations for uncached languages."""
            source_text = state.get('source_text', '')
            target_languages = state.get('target_languages', [])
            translations = state.get('translations', {})
            
            # Find languages that need translation (not cached)
            pending_languages = [lang for lang in target_languages if lang not in translations]
            
            if not pending_languages:
                state['messages'].append(
                    AIMessage(content="All translations found in cache")
                )
                return state
            
            try:
                # Perform translations concurrently
                translation_tasks = [
                    self._translate_to_language(source_text, lang)
                    for lang in pending_languages
                ]
                
                translation_results = await asyncio.gather(*translation_tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(translation_results):
                    lang = pending_languages[i]
                    
                    if isinstance(result, Exception):
                        self.logger.error(f"Translation failed for {lang}: {result}")
                        state['translations'][lang] = f"Translation failed: {str(result)}"
                        state['quality_scores'][lang] = 0.0
                    else:
                        translated_text, quality_score = result
                        state['translations'][lang] = translated_text
                        state['quality_scores'][lang] = quality_score
                        
                        # Cache successful translation
                        source_hash = self._generate_text_hash(source_text)
                        cache_key = f"{source_hash}_{lang}"
                        self.translation_cache[cache_key] = {
                            'text': translated_text,
                            'quality_score': quality_score,
                            'cached_at': datetime.now().isoformat()
                        }
                
                successful_translations = sum(1 for lang in pending_languages 
                                            if state['quality_scores'].get(lang, 0) > 0.5)
                
                state['messages'].append(
                    AIMessage(content=f"Completed {successful_translations}/{len(pending_languages)} new translations")
                )
                
            except Exception as e:
                state['last_error'] = f"Translation processing error: {str(e)}"
                state['status'] = AgentStatus.FAILED
                self.logger.error(f"Translation processing failed: {e}")
            
            return state
        
        async def validate_quality(state: TranslationState) -> TranslationState:
            """Validate translation quality and consistency."""
            quality_scores = state.get('quality_scores', {})
            translations = state.get('translations', {})
            
            if not quality_scores:
                state['messages'].append(
                    AIMessage(content="No quality scores available for validation")
                )
                return state
            
            # Calculate overall quality metrics
            avg_quality = sum(quality_scores.values()) / len(quality_scores)
            high_quality_count = sum(1 for score in quality_scores.values() if score >= 0.8)
            low_quality_langs = [lang for lang, score in quality_scores.items() if score < 0.5]
            
            # Check for terminology consistency
            consistency_scores = await self._check_terminology_consistency(translations)
            
            state['metadata'] = {
                'average_quality': avg_quality,
                'high_quality_translations': high_quality_count,
                'low_quality_languages': low_quality_langs,
                'terminology_consistency': consistency_scores,
                'total_translations': len(translations)
            }
            
            state['messages'].append(
                AIMessage(content=f"Quality validation: Avg score {avg_quality:.2f}, {high_quality_count} high quality translations")
            )
            
            return state
        
        def check_completion(state: TranslationState) -> Literal["completed", "retry", "failed"]:
            """Check if translation process is complete."""
            translations = state.get('translations', {})
            target_languages = state.get('target_languages', [])
            quality_scores = state.get('quality_scores', {})
            retry_count = state.get('retry_count', 0)
            
            if not translations or not target_languages:
                return "failed"
            
            # Check if we have translations for all target languages
            missing_languages = [lang for lang in target_languages if lang not in translations]
            
            if missing_languages:
                if retry_count < 2:
                    return "retry"
                else:
                    return "failed"
            
            # Check quality threshold
            low_quality_count = sum(1 for score in quality_scores.values() if score < 0.3)
            
            if low_quality_count > len(target_languages) * 0.5 and retry_count < 2:
                return "retry"
            
            return "completed"
        
        def finalize_success(state: TranslationState) -> TranslationState:
            """Finalize successful translation."""
            translations = state.get('translations', {})
            metadata = state.get('metadata', {})
            
            avg_quality = metadata.get('average_quality', 0)
            
            state['messages'].append(
                AIMessage(content=f"TranslationOrchestrator completed successfully. "
                                f"Translated to {len(translations)} languages with avg quality {avg_quality:.2f}")
            )
            state['status'] = AgentStatus.COMPLETED
            return state
        
        def handle_retry(state: TranslationState) -> TranslationState:
            """Handle retry for failed translations."""
            retry_count = state.get('retry_count', 0) + 1
            state['retry_count'] = retry_count
            
            # Reset low-quality translations for retry
            quality_scores = state.get('quality_scores', {})
            translations = state.get('translations', {})
            
            retry_languages = []
            for lang, score in quality_scores.items():
                if score < 0.5:
                    retry_languages.append(lang)
                    if lang in translations:
                        del translations[lang]
                    if lang in quality_scores:
                        del quality_scores[lang]
            
            state['messages'].append(
                AIMessage(content=f"Retrying translation for {len(retry_languages)} languages (attempt {retry_count})")
            )
            state['status'] = AgentStatus.RETRYING
            return state
        
        def handle_failure(state: TranslationState) -> TranslationState:
            """Handle final failure state."""
            error_msg = state.get('last_error', 'Translation quality below threshold')
            
            state['messages'].append(
                AIMessage(content=f"TranslationOrchestrator failed: {error_msg}")
            )
            state['status'] = AgentStatus.FAILED
            return state
        
        # Build the StateGraph
        workflow = StateGraph(TranslationState)
        
        # Add nodes
        workflow.add_node("prepare_translation", prepare_translation)
        workflow.add_node("perform_translations", perform_translations)
        workflow.add_node("validate_quality", validate_quality)
        workflow.add_node("finalize_success", finalize_success)
        workflow.add_node("handle_retry", handle_retry)
        workflow.add_node("handle_failure", handle_failure)
        
        # Add edges
        workflow.add_edge(START, "prepare_translation")
        workflow.add_edge("prepare_translation", "perform_translations")
        workflow.add_edge("perform_translations", "validate_quality")
        
        # Conditional edges based on completion check
        workflow.add_conditional_edges(
            "validate_quality",
            check_completion,
            {
                "completed": "finalize_success",
                "retry": "handle_retry", 
                "failed": "handle_failure"
            }
        )
        
        workflow.add_edge("handle_retry", "perform_translations")
        workflow.add_edge("finalize_success", END)
        workflow.add_edge("handle_failure", END)
        
        return workflow
    
    async def _translate_to_language(self, source_text: str, target_language: str) -> tuple[str, float]:
        """Translate text to a specific language using LLM."""
        async with self.semaphore:
            try:
                lang_config = self.language_configs.get(target_language, {})
                lang_name = lang_config.get('name', target_language.upper())
                financial_terms = lang_config.get('financial_terms', {})
                
                # Create context-aware prompt
                terminology_context = ""
                if financial_terms:
                    term_examples = [f"'{eng}' -> '{local}'" for eng, local in list(financial_terms.items())[:5]]
                    terminology_context = f"\n\nImportant financial terminology mappings:\n{', '.join(term_examples)}"
                
                prompt = f"""
Translate the following financial education content from English to {lang_name}.

Instructions:
1. Maintain accuracy of financial concepts and terminology
2. Preserve the educational tone and structure
3. Use appropriate financial terminology in {lang_name}
4. Ensure cultural sensitivity for Indian financial markets context{terminology_context}

Content to translate:
{source_text[:1500]}

Provide only the translation without any explanations.
"""
                
                response = await self.llm_adapter.generate(
                    prompt=prompt,
                    max_tokens=2000,
                    temperature=0.2
                )
                
                translated_text = response.text.strip()
                
                # Calculate quality score
                quality_score = self._calculate_translation_quality(
                    source_text, translated_text, target_language
                )
                
                return translated_text, quality_score
                
            except Exception as e:
                self.logger.error(f"Translation failed for {target_language}: {e}")
                raise e
    
    def _calculate_translation_quality(self, source_text: str, translated_text: str, language: str) -> float:
        """Calculate quality score for translation."""
        if not translated_text or translated_text == source_text:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Length ratio check (reasonable expansion/contraction)
        length_ratio = len(translated_text) / max(len(source_text), 1)
        if 0.5 <= length_ratio <= 2.0:
            quality_score += 0.2
        
        # Check for financial terminology preservation  
        lang_config = self.language_configs.get(language, {})
        financial_terms = lang_config.get('financial_terms', {})
        
        term_matches = 0
        for eng_term in financial_terms.keys():
            if eng_term.lower() in source_text.lower():
                term_matches += 1
        
        if term_matches > 0:
            quality_score += 0.2
        
        # Structure preservation (bullet points, numbers, etc.)
        if source_text.count('\n') > 0 and translated_text.count('\n') > 0:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    async def _check_terminology_consistency(self, translations: Dict[str, str]) -> Dict[str, float]:
        """Check terminology consistency across translations."""
        consistency_scores = {}
        
        for lang, translation in translations.items():
            lang_config = self.language_configs.get(lang, {})
            financial_terms = lang_config.get('financial_terms', {})
            
            if not financial_terms:
                consistency_scores[lang] = 0.5  # Neutral score for unsupported languages
                continue
            
            consistent_terms = 0
            total_terms = 0
            
            for eng_term, local_term in financial_terms.items():
                if eng_term.lower() in translation.lower():
                    total_terms += 1
                    if local_term in translation:
                        consistent_terms += 1
            
            if total_terms > 0:
                consistency_scores[lang] = consistent_terms / total_terms
            else:
                consistency_scores[lang] = 1.0  # No terms to check
        
        return consistency_scores
    
    def _generate_text_hash(self, text: str) -> str:
        """Generate hash for text caching."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

# Example usage and testing  
async def test_translation_orchestrator():
    """Test the LangGraph-based TranslationOrchestrator agent."""
    from llm_adapter import LLMAdapter, LLMConfig, LLMProvider
    
    # Setup LLM adapter
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
    
    llm_adapter = LLMAdapter(config)
    translator = TranslationOrchestratorAgent(llm_adapter=llm_adapter)
    
    # Compile the graph
    translator.compile()
    
    # Create initial state
    initial_state = TranslationState(
        messages=[HumanMessage(content="Translate content to multiple Indian languages")],
        status=AgentStatus.READY,
        agent_name="TranslationOrchestrator",
        source_text="""
Investment diversification is a fundamental principle of portfolio management. 
By spreading investments across different asset classes, sectors, and geographical regions, 
investors can reduce overall portfolio risk while maintaining potential for returns.
Key strategies include:
1. Asset allocation across stocks, bonds, and commodities
2. Sector diversification within equity investments  
3. International exposure through global funds
4. Regular portfolio rebalancing
        """,
        source_language="en",
        target_languages=["hi", "ta", "te"]
    )
    
    # Execute the workflow
    final_state = await translator.execute(initial_state)
    
    # Print results
    print(f"Final Status: {final_state['status']}")
    print(f"Translations: {list(final_state.get('translations', {}).keys())}")
    print(f"Quality Scores: {final_state.get('quality_scores', {})}")
    print(f"Messages: {[msg.content for msg in final_state['messages']]}")
    
    return final_state

if __name__ == "__main__":
    # asyncio.run(test_translation_orchestrator())
    pass
