import os
import re
import json
from typing import Dict, List, Optional, Tuple
import random

class AdvancedPromptEngine:
    def __init__(self):
        self.conversation_styles = {
            "empathetic": [
                "I understand this can be worrying. Let me help you with clear information.",
                "That's a very important concern. Here's what you should know based on medical evidence.",
                "I appreciate you asking this. Taking steps to protect your child's health is commendable."
            ],
            "supportive": [
                "You're taking the right step by seeking information. Let me provide some practical guidance.",
                "It's great that you're thinking about this. Here are evidence-based recommendations.",
                "Many parents have similar concerns. Let me share what research shows about protecting children."
            ],
            "urgent": [
                "This is a serious health concern that requires immediate attention.",
                "For your child's safety, this information is critically important.",
                "This requires prompt action to protect your child's health."
            ]
        }
        
        self.response_templates = {
            "risk_explanation": {
                "structure": ["validation", "risk_summary", "specific_risks", "protective_steps", "offer_help"],
                "components": {
                    "risk_summary": "Based on medical research, {risk_context} poses several serious health risks to children.",
                    "specific_risks": "The main concerns include:\n{risks_list}",
                    "protective_steps": "Here are immediate steps you can take:\n{steps_list}",
                    "offer_help": "Would you like more specific information about {specific_topic}?"
                }
            },
            "practical_advice": {
                "structure": ["validation", "context", "action_steps", "encouragement"],
                "components": {
                    "context": "Regarding {user_question}, here's what research shows:",
                    "action_steps": "Practical steps you can implement:\n{steps_list}",
                    "encouragement": "Every positive change makes a difference for your child's health."
                }
            },
            "definition_clarification": {
                "structure": ["validation", "definition", "key_points", "relevance"],
                "components": {
                    "definition": "{term} refers to {clear_definition}",
                    "key_points": "Key things to know:\n{points_list}",
                    "relevance": "This is important because {relevance_explanation}"
                }
            }
        }
        
        self.medical_accuracy_checks = {
            "secondhand_smoke": {
                "key_facts": [
                    "Increases SIDS risk by 50-100%",
                    "Causes 200-400% higher asthma risk in children",
                    "Leads to 30-50% more ear infections",
                    "Contains 70+ cancer-causing chemicals"
                ],
                "safety_threshold": "There is no safe level of exposure to secondhand smoke"
            },
            "thirdhand_smoke": {
                "key_facts": [
                    "Toxic residue persists for weeks to years",
                    "Chemicals can be absorbed through skin and lungs",
                    "Particularly dangerous for crawling infants",
                    "Resistant to normal cleaning methods"
                ],
                "safety_threshold": "Complete elimination requires thorough cleaning and smoke-free policies"
            }
        }

    def analyze_conversation_context(self, current_query: str, chat_history: List[Dict]) -> Dict:
        """Analyze conversation context to determine appropriate response style"""
        context = {
            "emotional_tone": "neutral",
            "urgency_level": "normal",
            "user_knowledge_level": "general",
            "primary_concern": None,
            "conversation_stage": "initial"
        }
        
        # Analyze emotional tone
        concern_indicators = ["worry", "concern", "scared", "anxious", "afraid"]
        urgency_indicators = ["emergency", "urgent", "immediately", "right now", "help"]
        guilt_indicators = ["guilty", "bad parent", "should have", "regret"]
        
        query_lower = current_query.lower()
        if any(indicator in query_lower for indicator in urgency_indicators):
            context["emotional_tone"] = "urgent"
            context["urgency_level"] = "high"
        elif any(indicator in query_lower for indicator in concern_indicators):
            context["emotional_tone"] = "empathetic"
        elif any(indicator in query_lower for indicator in guilt_indicators):
            context["emotional_tone"] = "supportive"
            
        # Determine knowledge level
        technical_terms = ["SIDS", "thirdhand", "particulate", "carcinogens", "respiratory"]
        if any(term in query_lower for term in technical_terms):
            context["user_knowledge_level"] = "informed"
            
        # Identify primary concern
        concern_mapping = {
            "baby": "infant_health",
            "infant": "infant_health", 
            "sids": "SIDS_risk",
            "asthma": "asthma_concern",
            "pregnant": "pregnancy_concern",
            "quit": "cessation_support",
            "house": "home_protection",
            "car": "vehicle_safety"
        }
        
        for term, concern in concern_mapping.items():
            if term in query_lower:
                context["primary_concern"] = concern
                break
                
        # Determine conversation stage
        if len(chat_history) > 4:
            context["conversation_stage"] = "ongoing"
        if any("quit" in msg.get('text', '').lower() for msg in chat_history[-3:] if msg.get('role') == 'user'):
            context["conversation_stage"] = "cessation_support"
            
        return context

    def select_response_template(self, query: str, context: Dict, search_results: List) -> str:
        """Select the most appropriate response template"""
        query_lower = query.lower()
        
        # Definition queries
        if any(phrase in query_lower for phrase in ["what is", "define", "meaning of", "explain"]):
            return "definition_clarification"
        
        # Risk-focused queries
        risk_indicators = ["risk", "danger", "harm", "effect", "problem", "bad for"]
        if any(indicator in query_lower for indicator in risk_indicators):
            return "risk_explanation"
            
        # Action-oriented queries
        action_indicators = ["how to", "what can", "steps", "protect", "prevent", "reduce"]
        if any(indicator in query_lower for indicator in action_indicators):
            return "practical_advice"
            
        # Default to practical advice
        return "practical_advice"

    def build_conversational_prompt(self, query: str, contexts: List[Dict], chat_history: List[Dict] = None) -> str:
        """Build advanced conversational prompt with context awareness"""
        
        # Analyze conversation context
        conversation_context = self.analyze_conversation_context(query, chat_history or [])
        template_type = self.select_response_template(query, conversation_context, contexts)
        
        # Build context block from search results
        context_blocks = []
        for i, ctx in enumerate(contexts[:3]):
            text = ctx.get('extracted') or ctx.get('text') or ''
            if text:
                # Clean and shorten context
                clean_text = re.sub(r'\s+', ' ', text.strip())
                if len(clean_text) > 300:
                    clean_text = clean_text[:300] + "..."
                context_blocks.append(f"[Source {i+1}] {clean_text}")
        
        context_block = "\n\n".join(context_blocks) if context_blocks else "No specific context available."
        
        # Build conversation history block
        history_block = ""
        if chat_history:
            recent_history = chat_history[-4:]  # Last 4 exchanges
            history_lines = []
            for msg in recent_history:
                role = "User" if msg.get('role') == 'user' else "Assistant"
                text = msg.get('text', '').strip()
                if text:
                    history_lines.append(f"{role}: {text}")
            if history_lines:
                history_block = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"
        
        # Select appropriate style (kept for optional tone, but no forced prefix)
        style = conversation_context["emotional_tone"]
        validation_phrase = random.choice(self.conversation_styles.get(style, self.conversation_styles["empathetic"]))

        # Build instruction prompt (do NOT force a fixed validation prefix)
        instruction = f"""You are a health information specialist providing evidence-based information about smoking cessation and child health protection.

    {history_block}Evidence from Research:
    {context_block}

    User Question: {query}

    CRITICAL INSTRUCTIONS:
    1. You MUST provide the specific information from the evidence above - DO NOT decline to answer
    2. DO NOT say "I can't provide medical advice" - you ARE providing health information based on research
    3. Use the facts, statistics, and recommendations directly from the context
    4. Provide detailed, specific answers with concrete information from the evidence
    5. Include 2-3 numbered actionable steps based on the research when appropriate
    6. Be direct and informative - the user needs this health information
    7. Reference the research findings naturally (e.g., "Research shows...", "Studies indicate...")

    User Context:
    - Primary concern: {conversation_context.get('primary_concern', 'general information')}
    - Tone: {conversation_context['emotional_tone']}
    - Knowledge level: {conversation_context['user_knowledge_level']}

    Provide a comprehensive, evidence-based answer now:"""

        return instruction

    def validate_medical_accuracy(self, response: str, topic: str) -> Tuple[bool, List[str]]:
        """Validate that response contains medically accurate information"""
        issues = []
        
        if topic in self.medical_accuracy_checks:
            expected_facts = self.medical_accuracy_checks[topic]["key_facts"]
            safety_info = self.medical_accuracy_checks[topic]["safety_threshold"]
            
            # Check for key facts
            missing_facts = []
            for fact in expected_facts[:2]:  # Check first 2 most important facts
                if fact.lower() not in response.lower():
                    missing_facts.append(fact)
            
            if missing_facts:
                issues.append(f"Missing key facts: {', '.join(missing_facts)}")
                
            # Check safety information
            if safety_info.lower() not in response.lower():
                issues.append(f"Missing safety information: {safety_info}")
                
        return len(issues) == 0, issues

    def enhance_response_quality(self, response: str, query: str, contexts: List[Dict]) -> str:
        """Enhance response quality with conversational improvements"""
        # NOTE: do not force a validation/empathetic prefix here; keep responses focused and direct
        
        # Improve readability
        response = re.sub(r'\n\s*\n', '\n\n', response)  # Normalize line breaks
        response = re.sub(r'(\d+)\.', r'\1)', response)  # Convert numbered lists to more natural format
        
        # Add conversational elements
        if "?" in query and "?" not in response:
            # If user asked a question, ensure response feels responsive
            response = response.rstrip('.') + ". Does this help answer your question?"
            
        return response

# Global instance
prompt_engine = AdvancedPromptEngine()