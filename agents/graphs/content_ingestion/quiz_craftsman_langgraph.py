"""
LangGraph-based QuizCraftsman Agent using StateGraph patterns.
Provides intelligent quiz generation with adaptive difficulty and multiple question types.
"""
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import asyncio
import logging
import json
import re

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from base_agent import (
    BaseLangGraphAgent,
    QuizState,
    LangGraphConfig,
    AgentStatus
)
from llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

class QuizCraftsmanAgent(BaseLangGraphAgent):
    """LangGraph-based QuizCraftsman with intelligent question generation."""
    
    def __init__(self, llm_adapter: LLMAdapter, config: Optional[LangGraphConfig] = None):
        super().__init__("QuizCraftsman", config)
        self.llm_adapter = llm_adapter
        self.semaphore = asyncio.Semaphore(2)  # Concurrent processing limit
        
        # Quiz type configurations
        self.quiz_configs = {
            'multiple_choice': {
                'name': 'Multiple Choice Questions',
                'question_count': 5,
                'options_per_question': 4,
                'style': 'clear options with one correct answer'
            },
            'true_false': {
                'name': 'True/False Questions', 
                'question_count': 8,
                'options_per_question': 2,
                'style': 'statements requiring true/false evaluation'
            },
            'scenario_based': {
                'name': 'Scenario-Based Questions',
                'question_count': 3,
                'options_per_question': 4,
                'style': 'real-world investment scenarios with analysis'
            },
            'myth_vs_fact': {
                'name': 'Myth vs Fact',
                'question_count': 6,
                'options_per_question': 2,
                'style': 'common misconceptions vs established facts'
            }
        }
        
        # Difficulty level configurations
        self.difficulty_configs = {
            'easy': {
                'name': 'Beginner Level',
                'complexity': 'basic concepts and definitions',
                'target_audience': 'new investors'
            },
            'medium': {
                'name': 'Intermediate Level',
                'complexity': 'practical applications and analysis',
                'target_audience': 'investors with some experience'
            },
            'hard': {
                'name': 'Advanced Level',
                'complexity': 'complex scenarios and strategic thinking',
                'target_audience': 'experienced investors and advisors'
            }
        }
    
    def build_graph(self) -> StateGraph:
        """Build the QuizCraftsman LangGraph workflow."""
        
        def prepare_quiz_generation(state: QuizState) -> QuizState:
            """Prepare quiz generation request and validate inputs."""
            content = state.get('content', '')
            quiz_type = state.get('quiz_type', 'multiple_choice')
            difficulty_level = state.get('difficulty_level', 'medium')
            
            if not content:
                state['messages'].append(
                    AIMessage(content="No content provided for quiz generation")
                )
                state['status'] = AgentStatus.FAILED
                state['last_error'] = "No content provided"
                return state
            
            # Validate quiz type and difficulty
            if quiz_type not in self.quiz_configs:
                quiz_type = 'multiple_choice'  # Default fallback
                state['quiz_type'] = quiz_type
            
            if difficulty_level not in self.difficulty_configs:
                difficulty_level = 'medium'  # Default fallback
                state['difficulty_level'] = difficulty_level
            
            # Initialize tracking
            state['questions'] = []
            state['quiz_metadata'] = {
                'quiz_type': quiz_type,
                'difficulty_level': difficulty_level,
                'target_question_count': self.quiz_configs[quiz_type]['question_count'],
                'generated_at': datetime.now().isoformat()
            }
            
            state['messages'].append(
                AIMessage(content=f"Prepared {quiz_type} quiz generation at {difficulty_level} difficulty")
            )
            
            return state
        
        async def analyze_content(state: QuizState) -> QuizState:
            """Analyze content to identify key topics and concepts for quiz questions."""
            content = state.get('content', '')
            difficulty_level = state.get('difficulty_level', 'medium')
            
            try:
                analysis_prompt = f"""
Analyze the following financial education content for quiz question generation:

{content[:2000]}

Identify:
1. Main topics and concepts (5-7 key areas)
2. Important facts and figures 
3. Regulatory aspects and SEBI guidelines mentioned
4. Common misconceptions that could be addressed
5. Practical applications and scenarios
6. Risk factors and considerations

Focus on content suitable for {difficulty_level} level questions.
Format as a structured analysis with clear sections.
"""
                
                response = await self.llm_adapter.generate(
                    prompt=analysis_prompt,
                    max_tokens=500,
                    temperature=0.3
                )
                
                # Parse the analysis
                analysis_text = response.text.strip()
                
                state['quiz_metadata']['content_analysis'] = analysis_text
                state['quiz_metadata']['topics_identified'] = self._extract_topics_from_analysis(analysis_text)
                
                state['messages'].append(
                    AIMessage(content=f"Analyzed content and identified {len(state['quiz_metadata']['topics_identified'])} key topics")
                )
                
            except Exception as e:
                self.logger.warning(f"Content analysis failed: {e}")
                # Continue with basic topic extraction
                state['quiz_metadata']['topics_identified'] = ['investment basics', 'risk management', 'portfolio management']
            
            return state
        
        async def generate_questions(state: QuizState) -> QuizState:
            """Generate quiz questions based on content analysis."""
            content = state.get('content', '')
            quiz_type = state.get('quiz_type', 'multiple_choice')
            difficulty_level = state.get('difficulty_level', 'medium')
            topics = state['quiz_metadata'].get('topics_identified', [])
            
            try:
                quiz_config = self.quiz_configs[quiz_type]
                difficulty_config = self.difficulty_configs[difficulty_level]
                
                question_generation_prompt = f"""
Generate {quiz_config['question_count']} {quiz_config['name']} based on this financial content:

{content[:1500]}

Requirements:
- Difficulty: {difficulty_config['name']} - {difficulty_config['complexity']}
- Target audience: {difficulty_config['target_audience']}
- Style: {quiz_config['style']}
- Focus topics: {', '.join(topics[:5])}

{self._get_quiz_type_instructions(quiz_type)}

Ensure questions:
- Are factually accurate and comply with SEBI guidelines
- Test understanding rather than memorization
- Include practical applications where appropriate
- Avoid ambiguous wording
- Have clear, unambiguous correct answers

Format each question as:
QUESTION [number]: [question text]
A) [option 1]
B) [option 2]  
C) [option 3]
D) [option 4]
CORRECT: [letter]
EXPLANATION: [brief explanation of correct answer]

---
"""
                
                response = await self.llm_adapter.generate(
                    prompt=question_generation_prompt,
                    max_tokens=2000,
                    temperature=0.4
                )
                
                # Parse questions from response
                questions = self._parse_questions_from_response(response.text, quiz_type)
                
                state['questions'] = questions
                state['quiz_metadata']['questions_generated'] = len(questions)
                
                state['messages'].append(
                    AIMessage(content=f"Generated {len(questions)} quiz questions")
                )
                
            except Exception as e:
                state['last_error'] = f"Question generation error: {str(e)}"
                state['status'] = AgentStatus.FAILED
                self.logger.error(f"Question generation failed: {e}")
            
            return state
        
        async def validate_questions(state: QuizState) -> QuizState:
            """Validate generated questions for quality and accuracy."""
            questions = state.get('questions', [])
            
            if not questions:
                state['messages'].append(
                    AIMessage(content="No questions to validate")
                )
                return state
            
            try:
                validated_questions = []
                validation_issues = []
                
                for i, question in enumerate(questions):
                    issues = self._validate_single_question(question, i+1)
                    
                    if not issues:
                        validated_questions.append(question)
                    else:
                        validation_issues.extend(issues)
                        # Try to fix common issues
                        fixed_question = self._fix_question_issues(question, issues)
                        if fixed_question:
                            validated_questions.append(fixed_question)
                
                state['questions'] = validated_questions
                state['quiz_metadata']['validation_issues'] = validation_issues
                state['quiz_metadata']['validated_questions'] = len(validated_questions)
                
                state['messages'].append(
                    AIMessage(content=f"Validated {len(validated_questions)} questions. Found {len(validation_issues)} issues")
                )
                
            except Exception as e:
                self.logger.warning(f"Question validation failed: {e}")
                # Continue with original questions
            
            return state
        
        def check_completion(state: QuizState) -> Literal["completed", "retry", "failed"]:
            """Check if quiz generation is complete."""
            questions = state.get('questions', [])
            quiz_type = state.get('quiz_type', 'multiple_choice')
            target_count = self.quiz_configs[quiz_type]['question_count']
            retry_count = state.get('retry_count', 0)
            
            if not questions:
                return "failed" if retry_count >= 2 else "retry"
            
            # Check if we have sufficient questions (at least 60% of target)
            min_required = max(1, int(target_count * 0.6))
            
            if len(questions) >= min_required:
                return "completed"
            elif retry_count < 2:
                return "retry"
            else:
                return "failed"
        
        def finalize_success(state: QuizState) -> QuizState:
            """Finalize successful quiz generation."""
            questions = state.get('questions', [])
            quiz_metadata = state.get('quiz_metadata', {})
            
            # Update final metadata
            quiz_metadata.update({
                'final_question_count': len(questions),
                'completion_timestamp': datetime.now().isoformat(),
                'quiz_ready': True
            })
            
            state['quiz_metadata'] = quiz_metadata
            
            state['messages'].append(
                AIMessage(content=f"QuizCraftsman completed successfully. Generated {len(questions)} high-quality quiz questions.")
            )
            state['status'] = AgentStatus.COMPLETED
            return state
        
        def handle_retry(state: QuizState) -> QuizState:
            """Handle retry for insufficient question generation."""
            retry_count = state.get('retry_count', 0) + 1
            state['retry_count'] = retry_count
            
            # Reset questions for fresh generation
            state['questions'] = []
            
            state['messages'].append(
                AIMessage(content=f"Retrying quiz generation with adjusted parameters (attempt {retry_count})")
            )
            state['status'] = AgentStatus.RETRYING
            return state
        
        def handle_failure(state: QuizState) -> QuizState:
            """Handle final failure state."""
            error_msg = state.get('last_error', 'Insufficient quiz questions generated')
            
            state['messages'].append(
                AIMessage(content=f"QuizCraftsman failed: {error_msg}")
            )
            state['status'] = AgentStatus.FAILED
            return state
        
        # Build the StateGraph
        workflow = StateGraph(QuizState)
        
        # Add nodes
        workflow.add_node("prepare_quiz_generation", prepare_quiz_generation)
        workflow.add_node("analyze_content", analyze_content)
        workflow.add_node("generate_questions", generate_questions)
        workflow.add_node("validate_questions", validate_questions)
        workflow.add_node("finalize_success", finalize_success)
        workflow.add_node("handle_retry", handle_retry)
        workflow.add_node("handle_failure", handle_failure)
        
        # Add edges
        workflow.add_edge(START, "prepare_quiz_generation")
        workflow.add_edge("prepare_quiz_generation", "analyze_content")
        workflow.add_edge("analyze_content", "generate_questions")
        workflow.add_edge("generate_questions", "validate_questions")
        
        # Conditional edges based on completion check
        workflow.add_conditional_edges(
            "validate_questions",
            check_completion,
            {
                "completed": "finalize_success",
                "retry": "handle_retry",
                "failed": "handle_failure"
            }
        )
        
        workflow.add_edge("handle_retry", "generate_questions")
        workflow.add_edge("finalize_success", END)
        workflow.add_edge("handle_failure", END)
        
        return workflow
    
    def _extract_topics_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract key topics from content analysis."""
        topics = []
        
        # Look for numbered or bulleted lists
        lines = analysis_text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.|\*|-|•', line):
                # Extract the topic after the marker
                topic = re.sub(r'^\d+\.|\*|-|•\s*', '', line).strip()
                if len(topic) > 5 and len(topic) < 100:  # Reasonable topic length
                    topics.append(topic.lower())
        
        return topics[:10]  # Limit to top 10 topics
    
    def _get_quiz_type_instructions(self, quiz_type: str) -> str:
        """Get specific instructions for different quiz types."""
        instructions = {
            'multiple_choice': """
For Multiple Choice Questions:
- Create 4 distinct options (A, B, C, D)
- Only one option should be clearly correct
- Distractors should be plausible but incorrect
- Avoid "all of the above" or "none of the above" options
""",
            'true_false': """
For True/False Questions:
- Create clear, unambiguous statements
- Avoid absolute terms like "always" or "never" unless factually correct
- Focus on specific facts or principles
- Ensure statements can be definitively classified as true or false
""",
            'scenario_based': """
For Scenario-Based Questions:
- Present realistic investment situations
- Include relevant context and constraints
- Ask about best practices or optimal decisions
- Consider risk-return trade-offs in scenarios
""",
            'myth_vs_fact': """
For Myth vs Fact Questions:
- Present common investment misconceptions
- Contrast with established financial principles
- Explain why the myth is incorrect
- Provide the factual alternative
"""
        }
        return instructions.get(quiz_type, "")
    
    def _parse_questions_from_response(self, response_text: str, quiz_type: str) -> List[Dict[str, Any]]:
        """Parse questions from LLM response."""
        questions = []
        
        # Split by question markers
        question_blocks = re.split(r'QUESTION\s*\d+:|---', response_text)
        
        for block in question_blocks[1:]:  # Skip first empty split
            if not block.strip():
                continue
                
            question_data = self._parse_single_question_block(block.strip(), quiz_type)
            if question_data:
                questions.append(question_data)
        
        return questions
    
    def _parse_single_question_block(self, block: str, quiz_type: str) -> Optional[Dict[str, Any]]:
        """Parse a single question block."""
        try:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            
            if not lines:
                return None
            
            # Extract question text (first non-empty line)
            question_text = lines[0]
            
            # Extract options
            options = []
            correct_answer = None
            explanation = ""
            
            for line in lines[1:]:
                if line.startswith(('A)', 'B)', 'C)', 'D)')):
                    option_text = line[2:].strip()
                    options.append(option_text)
                elif line.startswith('CORRECT:'):
                    correct_answer = line.replace('CORRECT:', '').strip().upper()
                elif line.startswith('EXPLANATION:'):
                    explanation = line.replace('EXPLANATION:', '').strip()
            
            if not question_text or not options:
                return None
            
            return {
                'question': question_text,
                'options': options,
                'correct_answer': correct_answer,
                'explanation': explanation,
                'question_type': quiz_type,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse question block: {e}")
            return None
    
    def _validate_single_question(self, question: Dict[str, Any], question_num: int) -> List[str]:
        """Validate a single question for quality issues."""
        issues = []
        
        # Check required fields
        if not question.get('question'):
            issues.append(f"Question {question_num}: Missing question text")
        
        options = question.get('options', [])
        if len(options) < 2:
            issues.append(f"Question {question_num}: Insufficient answer options")
        
        correct_answer = question.get('correct_answer')
        if not correct_answer:
            issues.append(f"Question {question_num}: No correct answer specified")
        
        # Check answer validity
        if correct_answer and len(options) > 0:
            valid_answers = ['A', 'B', 'C', 'D'][:len(options)]
            if correct_answer not in valid_answers:
                issues.append(f"Question {question_num}: Invalid correct answer '{correct_answer}'")
        
        # Check for duplicate options
        if len(options) > len(set(options)):
            issues.append(f"Question {question_num}: Duplicate answer options")
        
        # Check question quality
        question_text = question.get('question', '')
        if len(question_text) < 10:
            issues.append(f"Question {question_num}: Question text too short")
        
        return issues
    
    def _fix_question_issues(self, question: Dict[str, Any], issues: List[str]) -> Optional[Dict[str, Any]]:
        """Attempt to fix common question issues."""
        fixed_question = question.copy()
        
        # Fix missing correct answer
        if 'No correct answer specified' in ' '.join(issues):
            options = question.get('options', [])
            if options:
                fixed_question['correct_answer'] = 'A'  # Default to first option
        
        # Fix invalid correct answer
        correct_answer = fixed_question.get('correct_answer', '')
        options = fixed_question.get('options', [])
        if options and correct_answer not in ['A', 'B', 'C', 'D'][:len(options)]:
            fixed_question['correct_answer'] = 'A'
        
        return fixed_question

# Example usage and testing
async def test_quiz_craftsman():
    """Test the LangGraph-based QuizCraftsman agent."""
    from llm_adapter import LLMAdapter, LLMConfig, LLMProvider
    
    # Setup LLM adapter
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
    
    llm_adapter = LLMAdapter(config)
    quiz_agent = QuizCraftsmanAgent(llm_adapter=llm_adapter)
    
    # Compile the graph
    quiz_agent.compile()
    
    # Create initial state
    initial_state = QuizState(
        messages=[HumanMessage(content="Generate quiz questions from financial content")],
        status=AgentStatus.READY,
        agent_name="QuizCraftsman",
        content="""
Systematic Investment Plan (SIP) is one of the most effective ways to build long-term wealth through mutual fund investments. SEBI regulations ensure that SIPs provide a disciplined approach to investing, allowing investors to benefit from rupee cost averaging and the power of compounding.

Key Benefits of SIP:
1. Rupee Cost Averaging: By investing a fixed amount regularly, investors buy more units when prices are low and fewer units when prices are high, averaging out the cost over time.

2. Power of Compounding: Regular investments allow earnings to generate their own earnings, creating exponential growth over long periods.

3. Financial Discipline: Automated monthly investments create a savings habit and prevent emotional investment decisions.

4. Flexibility: SIPs can be started with as little as ₹500 per month and can be increased, paused, or stopped as per investor needs.

5. Professional Management: Mutual fund managers handle portfolio diversification and security selection.

Risk Considerations:
- Market volatility can impact short-term returns
- Fund manager performance affects overall returns  
- Exit loads may apply for early withdrawals
- Tax implications on gains need to be considered

SEBI Guidelines for SIP:
- Mandatory disclosure of all charges and fees
- Flexibility to modify or discontinue SIP without penalties after initial period
- Clear communication of risk factors and past performance
- Regular portfolio disclosure requirements

Investment Strategy:
For optimal results, investors should maintain SIPs for at least 5-7 years, choose funds aligned with their risk tolerance and goals, and review portfolio performance annually while avoiding frequent changes based on short-term market movements.
""",
        quiz_type="multiple_choice",
        difficulty_level="medium"
    )
    
    # Execute the workflow
    final_state = await quiz_agent.execute(initial_state)
    
    # Print results
    print(f"Final Status: {final_state['status']}")
    print(f"Questions Generated: {len(final_state.get('questions', []))}")
    print(f"Quiz Metadata: {final_state.get('quiz_metadata', {})}")
    print(f"Messages: {[msg.content for msg in final_state['messages']]}")
    
    # Print sample questions
    questions = final_state.get('questions', [])
    if questions:
        print("\nSample Question:")
        q = questions[0]
        print(f"Q: {q.get('question', '')}")
        for i, option in enumerate(q.get('options', [])):
            print(f"{chr(65+i)}) {option}")
        print(f"Correct Answer: {q.get('correct_answer', '')}")
        print(f"Explanation: {q.get('explanation', '')}")
    
    return final_state

if __name__ == "__main__":
    # asyncio.run(test_quiz_craftsman())
    pass
