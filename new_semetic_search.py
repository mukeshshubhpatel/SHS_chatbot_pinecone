import sys
import os
from typing import List, Dict
import logging
import re
import random

# Import conversation management and orchestration
try:
    from conversation_manager import conversation_manager, ConversationState
except Exception as e:
    logging.warning(f"conversation_manager import failed: {e}")
    conversation_manager = None
    ConversationState = None

# Import the prompt engineering module for advanced prompt building
try:
    from prompt_engine import prompt_engine
except Exception as e:
    logging.warning(f"prompt_engine import failed: {e}. Some features may be unavailable.")
    prompt_engine = None

# Friendly import-time check to catch the common transformers/tokenizers mismatch
try:
    # Import tokenizers version early to help diagnose version mismatches
    import tokenizers as _tokenizers
    from transformers import AutoTokenizer, AutoModel
    import torch
    from pinecone import Pinecone
except Exception as e:
    # Attempt to report helpful, actionable guidance rather than the raw traceback
    try:
        tver = _tokenizers.__version__
    except Exception:
        tver = None

    msg_lines = [
        "\nERROR: Failed to import Transformers or its dependencies.",
        f"Reason: {e}",
        "",
        "Common cause: a 'tokenizers' / 'transformers' version mismatch or running a different Python interpreter than the project's venv.",
        "Detected tokenizers version: " + (tver or 'not installed'),
        "",
        "Remediation options:",
        "1) Run the script with the project's venv Python (recommended). In PowerShell:\n   .\\venv\\Scripts\\Activate.ps1\n   python .\\new_semetic_search.py",
        "   Or without activation:\n   .\\venv\\Scripts\\python .\\new_semetic_search.py",
        "",
        "2) If you prefer to fix your conda / system Python, install compatible versions in that environment:\n   python -m pip install \"tokenizers>=0.22.0,<0.24.0\" --upgrade\n   python -m pip install --upgrade transformers",
        "",
        "3) If you want, I can add a simple environment-check helper at the top of this file to print the active Python and package versions before proceeding (safe).",
        "",
        "After fixing, re-run the script with the same Python you used to install the packages.",
    ]

    sys.stderr.write('\n'.join(msg_lines) + '\n')
    raise

import chatbot_orchestrator
from print_matched_outputs_income import get_outputs_for_texts

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class SemanticSearch:
    def __init__(self):
        self.api_key = "#######"
        self.index_name = "chatbot"
        self.model_name = "BAAI/bge-large-en-v1.5"  # Updated to 1024D model for consistency
        self.model = None
        self.tokenizer = None
        self.index = None
        self.device = None
        self.conversation_manager = conversation_manager
        self.init_connections()

    def init_connections(self):
        """Initialize Pinecone and the embedding model."""
        try:
            logging.info("Initializing connections...")
            
            # Determine device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"Using device: {self.device}")
            
            # Initialize embedding model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Initialize Pinecone
            pc = Pinecone(api_key=self.api_key)
            self.index = pc.Index(self.index_name)
            
            logging.info("Initialization successful!")
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise

    def get_validation_sentence(self) -> str:
        """Return the exact validation sentence to prepend to replies.

        Can be configured via the `VALIDATION_SENTENCE` environment variable.
        """
        return os.environ.get('VALIDATION_SENTENCE', "I hear you — that sounds really difficult.")

    def ensure_validation_prefix(self, text: str) -> str:
        """Ensure the returned text starts with the exact validation sentence."""
        if not text:
            return "I couldn't find specific information about that. Could you try rephrasing your question?"
        return text

    def validation_strict(self) -> bool:
        """Return True when we should require the model to produce the
        validation sentence natively (do not auto-prefix model outputs).
        Controlled by env `VALIDATION_STRICT` (set to '1' to enable).
        """
        return os.environ.get('VALIDATION_STRICT', '').strip() in ('1', 'true', 'True')

    def build_definition_prompt(self, query: str, contexts: List[dict]) -> str:
        """Build specialized prompt for definition questions."""
        context_content = []
        for i, context in enumerate(contexts, start=1):
            text = (context.get('extracted') or context.get('text') or '').strip()
            src = context.get('source', 'Unknown')
            score = context.get('score', 0)
            excerpt = text if len(text) <= 300 else text[:300] + '...'
            context_content.append(f"[{i}] Source: {src} (score: {score:.3f})\n{excerpt}")

        context_block = "\n\n".join(context_content)

        prompt = f"""Based on the context below, provide a clear, concise definition of what SHS (Secondhand Smoke) is.

Context:
{context_block}

User Question: {query}

Please provide:
1. A clear definition of SHS (Secondhand Smoke)
2. Key components (what it contains)
3. Health risks associated with exposure
4. Who is most vulnerable

Keep the response factual, concise, and based solely on the provided context. If the context doesn't contain definition information, acknowledge this limitation."""
        return prompt

    def build_general_prompt(self, query: str, contexts: List[dict]) -> str:
        """Build prompt for general questions."""
        context_content = []
        for i, context in enumerate(contexts, start=1):
            text = (context.get('extracted') or context.get('text') or '').strip()
            src = context.get('source', 'Unknown')
            score = context.get('score', 0)
            excerpt = text if len(text) <= 300 else text[:300] + '...'
            context_content.append(f"[{i}] Source: {src} (score: {score:.3f})\n{excerpt}")

        context_block = "\n\n".join(context_content)

        prompt = f"""You are a helpful health advisor specializing in smoking cessation and secondhand smoke. 
Answer the user's question based on the provided context.

Context:
{context_block}

User Question: {query}

Guidelines:
- Provide accurate, evidence-based information
- Be empathetic and supportive
- If the context doesn't fully answer the question, acknowledge this
- Keep responses clear and concise
- Focus on practical advice when applicable

Response:"""
        return prompt

    def is_definition_query(self, query: str) -> bool:
        """Determine if the query is asking for a definition."""
        definition_phrases = [
            'what is', 'what are', 'define', 'definition of', 
            'meaning of', 'explain what', 'tell me about'
        ]
        query_lower = query.lower()
        return any(phrase in query_lower for phrase in definition_phrases)

    def generate_conversational_answer(self, query: str, contexts: List[dict] = None, chat_history: List[dict] = None) -> str:
        """Generate answers with conversation awareness using orchestrator."""
        # First, analyze the conversation context if available
        if self.conversation_manager and chat_history:
            conversation_state, context_info = self.conversation_manager.analyze_message(query, chat_history)
        elif self.conversation_manager:
            conversation_state, context_info = self.conversation_manager.analyze_message(query, [])
        else:
            conversation_state = None
            context_info = {}
        
        # If no contexts provided but we need database lookup, perform search
        if contexts is None and context_info.get('needs_database_lookup', False):
            contexts = self.search(query, top_k=3)
        
        # Use the enhanced prompt engineering for smoking questions
        if conversation_state and str(conversation_state) == "ConversationState.SMOKING_QUESTION":
            if prompt_engine:
                prompt = prompt_engine.build_conversational_prompt(query, contexts or [], chat_history)
                
                # Try to generate with Ollama/HF models
                try:
                    from ollama_test import ask_ollama
                    response = ask_ollama(query, contexts=contexts or [], chat_history=chat_history)
                    return response
                except Exception as e:
                    logging.warning(f"Model generation failed: {e}")
        
        # Fallback to smart response
        return self.smart_fallback_response(query, contexts or [], chat_history)

    def generate_answer(self, query: str, contexts: List[dict], chat_history: List[dict] = None) -> str:
        """Generate conversational, accurate answers using advanced prompt engineering."""
        
        # Build advanced prompt using prompt_engine if available
        if prompt_engine:
            prompt = prompt_engine.build_conversational_prompt(query, contexts, chat_history)
        else:
            # Fallback to basic prompt if prompt_engine not available
            prompt = self._build_basic_prompt(query, contexts, chat_history)
        
        # Try Ollama first if available
        try:
            from ollama_test import ask_ollama
            response = ask_ollama(query, contexts=contexts, chat_history=chat_history)
            
            # Validate medical accuracy
            primary_topic = self._detect_primary_topic(query, contexts)
            if prompt_engine:
                is_accurate, issues = prompt_engine.validate_medical_accuracy(response, primary_topic)
                
                if not is_accurate and contexts and len(issues) > 0:
                    # If inaccurate, regenerate with stricter guidance
                    strict_prompt = prompt + f"\n\nIMPORTANT: Ensure these key points are included: {', '.join(issues)}"
                    try:
                        response = self._regenerate_with_strict_prompt(strict_prompt, ask_ollama)
                    except Exception:
                        pass  # Keep the current response if regeneration fails
            
            # Enhance response quality
            if prompt_engine:
                response = prompt_engine.enhance_response_quality(response, query, contexts)
            return response
            
        except Exception as e:
            logging.warning(f"Ollama generation failed: {e}")
        
        # Fallback to improved deterministic response
        return self.smart_fallback_response(query, contexts, chat_history)
    
    def _build_basic_prompt(self, query: str, contexts: List[dict], chat_history: List[dict] = None) -> str:
        """Build basic prompt when prompt_engine is not available."""
        ctx_lines = []
        for i, c in enumerate(contexts[:3], start=1):
            excerpt = (c.get('extracted') or c.get('text') or '').strip()
            if not excerpt:
                continue
            ex_short = excerpt if len(excerpt) <= 800 else excerpt[:800] + '...'
            ctx_lines.append(f"[{i}] {ex_short}")
        context_block = "\n\n".join(ctx_lines)
        
        OPENING = self.get_validation_sentence()
        instruction = f"Start with: {OPENING}\nThen provide explanation and 3 numbered steps."
        
        return f"Context:\n{context_block}\n\nQuestion: {query}\n\nInstructions:\n{instruction}\n\nAnswer:"
    
    def _regenerate_with_strict_prompt(self, strict_prompt: str, ask_ollama_func) -> str:
        """Regenerate response with stricter medical accuracy guidance."""
        # Call Ollama with the strict prompt
        response = ask_ollama_func(strict_prompt, contexts=[])
        return response
    
    def _detect_primary_topic(self, query: str, contexts: List[dict]) -> str:
        """Detect primary topic for medical accuracy validation."""
        query_lower = query.lower()
        context_text = " ".join([ctx.get('text', '') for ctx in contexts]).lower()
        
        if any(term in query_lower or term in context_text for term in ['secondhand', 'shs', 'passive smoke']):
            return "secondhand_smoke"
        elif any(term in query_lower or term in context_text for term in ['thirdhand', 'residue', 'surface']):
            return "thirdhand_smoke"
        elif any(term in query_lower or term in context_text for term in ['quit', 'cessation', 'stop smoking']):
            return "cessation"
        
        return "general"

        # exact opening sentence required by product (configurable)
        OPENING = self.get_validation_sentence()

        # Few-shot examples (short) that show the exact required structure
        few_shot = (
            "Example 1:\nQuestion: What is secondhand smoke?\n"
            f"Answer: {OPENING} Secondhand smoke (SHS) is the smoke from burning tobacco and smoke exhaled by smokers. It contains many toxic chemicals and can harm people who do not smoke.\n\n"
            "Example 2:\nQuestion: How can I protect my child from SHS at home?\n"
            f"Answer: {OPENING} Keep your home and car 100% smoke-free, have a calm conversation with household members, offer cessation support, and lead by example.\n\n"
        )

        # Choose definition or general instruction but always enforce opening and numbered steps
        instruction = (
            f"You are a compassionate, concise health advisor. Use only the supplied context to answer.\n\n"
            f"Start the reply with exactly this sentence (do not omit or reword it):\n{OPENING}\n\n"
            "Then include one short sentence describing why (research shows harms), then provide 3-5 numbered, practical steps tailored to the user's question and the supplied context. End with one encouraging sentence offering further help.\n\n"
        )

        # Full prompt (include conversation block if available)
        prompt = "\n\n".join(["Conversation:", conv_block, "Context:", context_block, "", "Few-shot examples:", few_shot, "Instruction:", instruction, f"User question: {query}", "Answer:"])

        # Models to try
        hf_model = os.environ.get('HF_MODEL', 'microsoft/DialoGPT-large')
        hf_fallbacks = os.environ.get('HF_MODEL_FALLBACKS', '')
        models_to_try = [m.strip() for m in ([hf_model] + ([hf_fallbacks] if hf_fallbacks else [])) if m]
        # if fallbacks contains commas, expand
        if len(models_to_try) == 1 and ',' in models_to_try[0]:
            models_to_try = [m.strip() for m in models_to_try[0].split(',') if m.strip()]

        last_exc = None
        from transformers import AutoTokenizer, AutoModelForCausalLM

        for model_id in models_to_try:
            try:
                logging.info(f"Attempting generation with model: {model_id}")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model_obj = AutoModelForCausalLM.from_pretrained(model_id)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_obj.to(device)

                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                gen_kwargs = dict(
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=float(os.environ.get('GEN_TEMP', 0.3)),
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                if attention_mask is not None:
                    out_ids = model_obj.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
                else:
                    out_ids = model_obj.generate(input_ids, **gen_kwargs)

                full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                # Extract generated portion after the prompt (if present)
                if full.startswith(prompt):
                    generated = full[len(prompt):].strip()
                else:
                    # sometimes tokenizer returns prompt + generated, sometimes only generated
                    generated = full.replace(prompt, '').strip()

                generated = self.clean_response(generated)

                # Accept only if generated starts with the exact opening sentence and is long enough
                # If VALIDATION_STRICT is enabled, require the model to produce the sentence natively.
                if generated and generated.strip().startswith(OPENING) and len(generated) >= 60:
                    logging.info(f"Model {model_id} produced acceptable response.")
                    # Return the generated text as-is (native includes validation sentence)
                    return generated
                else:
                    # If strict mode is disabled, we could have prefixed the output, but we
                    # treat missing/short output as failure so the deterministic fallback runs.
                    logging.warning(f"Model {model_id} produced insufficiently-structured or too-short response; trying next model.")
                    last_exc = RuntimeError("Generated response too short or missing validation sentence")
                    continue

            except Exception as e:
                logging.error(f"Generation failed for {model_id}: {e}")
                last_exc = e
                continue

        logging.warning("All model attempts failed or produced insufficient output; using composed fallback.")
        return self.smart_fallback_response(query, contexts)

    def clean_response(self, text: str) -> str:
        """Clean up generated text."""
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove any incomplete sentences at the end
        text = re.sub(r'[^.!?]*$', '', text).strip()
        return text

    def smart_fallback_response(self, query: str, contexts: List[dict], chat_history: List[dict] = None) -> str:
        """Improved fallback with conversation awareness."""
        if not contexts:
            return self.ensure_validation_prefix("I couldn't find specific information about that in my knowledge base. Could you try rephrasing your question or ask about secondhand smoke health effects, protection strategies, or cessation resources?")

        # Extract meaningful content from contexts
        useful_info = []
        for context in contexts:
            text = (context.get('extracted') or context.get('text') or '').strip()
            if text and len(text) > 10:  # Only use substantial text
                useful_info.append(text)

        if not useful_info:
            return "I found some references but couldn't extract specific details. Please try asking about smoke-free policies, health risks of secondhand smoke, or quitting resources."

        # Analyze conversation context for better responses
        conversation_context = {}
        if prompt_engine:
            conversation_context = prompt_engine.analyze_conversation_context(query, chat_history or [])
        
        # Create a synthesized response based on found content
        if self.is_definition_query(query):
            response = self.create_definition_fallback(useful_info, query)
        else:
            response = self.create_conversational_fallback(useful_info, query, conversation_context)
        
        # Enhance response quality
        if prompt_engine:
            response = prompt_engine.enhance_response_quality(response, query, contexts)
        
        return response

    def create_definition_fallback(self, useful_info: List[str], query: str) -> str:
        """Create definition response from found content with varied openers."""
        # Prefer longer, sentence-like definitions from the useful info
        definition_keywords = ['secondhand smoke', 'shs', 'environmental tobacco', 'passive smok']
        primary_definition = None
        extra_sentences = []

        for info in useful_info:
            if not info:
                continue
            # Strip headings like 'Secondhand smoke health consequences:'
            candidate = re.sub(r':\s*$', '', info.split('\n')[0]).strip()
            sentences = re.split(r'(?<=[.!?])\s+', candidate)
            for sent in sentences:
                if any(k in sent.lower() for k in definition_keywords) and len(sent.strip()) > 30:
                    if primary_definition is None:
                        primary_definition = sent.strip().rstrip('.')
                    else:
                        extra_sentences.append(sent.strip().rstrip('.'))
            # also scan longer paragraphs for secondary supporting sentences
            if len(primary_definition or '') < 10 and len(info) > 60:
                # fallback: take the first long sentence containing a keyword
                for sent in re.split(r'(?<=[.!?])\s+', info):
                    if any(k in sent.lower() for k in definition_keywords) and len(sent) > 30:
                        primary_definition = (primary_definition or sent.strip().rstrip('.'))
                        break

        # Safe defaults if nothing found
        if not primary_definition:
            primary_definition = "Secondhand smoke (SHS) is the smoke from the burning end of tobacco products and the smoke exhaled by people who smoke"

        # Build structured reply: definition + why + numbered how + encouragement
        # One-line definition sentence
        definition_line = f"{primary_definition}."

        # Build a short 'why' line: look for health-risk phrases in useful_info
        why = None
        for info in useful_info:
            if not info:
                continue
            low = info.lower()
            if 'no safe level' in low or 'even brief exposure' in low or 'children' in low or 'increases risk' in low:
                why = 'Even brief exposure can be harmful; SHS contains toxic chemicals that increase risks for respiratory and cardiovascular problems.'
                break
        if not why:
            why = 'SHS contains many harmful chemicals and has been linked to serious health problems.'

        # Numbered practical steps (tailor lightly to children if query mentions children)
        steps = []
        if 'child' in query.lower() or 'children' in query.lower():
            steps = [
                'Make your home and car completely smoke-free — do not allow indoor smoking.',
                'Ask household and regular visitors to smoke outside and away from doors and windows.',
                'If a household member smokes, offer or connect them with quitline support, counseling, or nicotine replacement options.'
            ]
        else:
            steps = [
                'Make your home and car completely smoke-free — there is no safe level of SHS.',
                'Encourage smokers to quit by offering support and information about local cessation resources or quitlines.',
                'Use clear household rules (a written or spoken smoke-free policy) and lead by example.'
            ]

        encouragement = 'If you like, I can pull the exact lines from the sources I found or help you locate local quitline numbers and services.'

        # Assemble parts without validation sentence prefix
        parts = [definition_line, f"Why: {why}", 'How:']
        for i, s in enumerate(steps, start=1):
            parts.append(f"{i}) {s}")
        parts.append(encouragement)

        return '\n\n'.join(parts)

    def create_conversational_fallback(self, contexts: List[dict], query: str, conversation_context: Dict) -> str:
        """Create conversational fallback response aware of conversation history."""
        # Extract key information from contexts
        key_points = []
        for ctx in contexts[:3]:
            text = (ctx.get('extracted') or ctx.get('text') or '').strip() if isinstance(ctx, dict) else ctx
            if text and len(text) > 20:
                # Extract the most relevant sentence
                sentences = re.split(r'[.!?]+', text)
                if sentences:
                    key_points.append(sentences[0].strip() + '.')
        
        if not key_points:
            key_points = [
                "Secondhand smoke contains over 70 cancer-causing chemicals.",
                "Children exposed to smoke have higher rates of asthma and respiratory infections.",
                "Creating smoke-free environments is the most effective protection."
            ]
        
        # Build response based on conversation context
        if prompt_engine and hasattr(prompt_engine, 'conversation_styles'):
            tone = conversation_context.get('emotional_tone', 'supportive')
            validation = random.choice(prompt_engine.conversation_styles.get(tone, prompt_engine.conversation_styles['supportive']))
        else:
            validation = self.get_validation_sentence()
        
        response_parts = [validation]
        
        concern = conversation_context.get('primary_concern', 'general')
        if concern == "infant_health":
            response_parts.extend([
                "For infants, the most critical concern is Sudden Infant Death Syndrome (SIDS).",
                "Key protective steps:",
                "1) Keep your home and car completely smoke-free",
                "2) Ask smokers to change clothes before holding the baby", 
                "3) Use air purifiers and frequent cleaning to reduce residue",
                "Would you like more specific information about infant protection?"
            ])
        else:
            response_parts.extend([
                "Based on available information:",
                "\n".join([f"• {point}" for point in key_points[:2]]),
                "Practical steps you can take:",
                "1) Establish smoke-free rules for home and vehicles",
                "2) Seek smoking cessation support if needed", 
                "3) Educate family members about the risks",
                "I can provide more detailed guidance if you tell me about your specific situation."
            ])
        
        return "\n\n".join(response_parts)

    def create_general_fallback(self, useful_info: List[str], query: str) -> str:
        """Legacy method - delegates to create_conversational_fallback."""
        return self.create_conversational_fallback(useful_info, query, {})

        # Synthesis: detect common topics from excerpts
        joined = ' '.join([e.lower() for e in excerpts])
        synth_points = []
        if 'children' in joined or 'child' in joined:
            synth_points.append('protecting children')
        if 'smoke-free' in joined or 'smoke free' in joined or 'policy' in joined or 'law' in joined:
            synth_points.append('policy measures and smoke-free rules')
        if 'quit' in joined or 'quitline' in joined or 'cessation' in joined:
            synth_points.append('offering cessation support')

        if synth_points:
            synthesis = 'Overall, the top sources emphasize ' + ', '.join(synth_points) + '.'
        else:
            synthesis = 'Overall, the top sources emphasize that secondhand smoke causes serious harm and that clear protective steps are effective.'

        # Construct practical steps tailored lightly by synthesis
        steps = [
            'Decide that your entire home and car are 100% smoke-free — there is no safe level of secondhand smoke.',
            'Have a calm, short conversation with household members explaining this is about everyone’s health and well‑being.',
            'If someone smokes, help them find support: call the local quitline, consider counseling or nicotine replacement, or find online programs.',
            'Lead by example — your commitment helps make the new, healthy routine stick.'
        ]
        if 'policy measures' in synthesis and 'policy' not in steps[-1]:
            steps.append('Where appropriate, support smoke-free policies for multiunit housing and public spaces to reduce exposure at scale.')

        how_block = '\n'.join([f"{i+1}) {s}" for i, s in enumerate(steps)])

        parts = [OPENING]
        if personalized:
            parts.append(personalized)
        parts.append('Key findings from the top sources:')
        parts.append('\n'.join(key_lines))
        parts.append('Synthesis: ' + synthesis)
        parts.append('Based on the above, here are practical steps you can take now:')
        parts.append(how_block)
        parts.append('\nIf you want, I can show the exact lines from the sources or find local resources and quitline numbers.')

        return '\n\n'.join(parts)

    def get_embedding(self, text):
        """Create embedding for search."""
        try:
            # Tokenize and move to correct device
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract embedding and move to CPU for processing
            embedding = outputs.last_hidden_state[:, 0].squeeze()
            embedding = embedding.cpu().numpy().tolist()
            
            if not isinstance(embedding, list):
                embedding = [float(x) for x in embedding]
                
            return embedding
            
        except Exception as e:
            logging.error(f"Embedding creation failed: {e}")
            raise

    def search(self, query_text, top_k=5):  # Increased top_k for better results
        """Perform semantic search with improved query handling."""
        logging.info("Searching for: '%s'", query_text)

        # Clean and expand query for better search
        clean_query = self.clean_query(query_text)
        
        query_embedding = self.get_embedding(clean_query)

        # Truncate if necessary for Pinecone
        max_dimension = 1024
        if len(query_embedding) > max_dimension:
            query_embedding = query_embedding[:max_dimension]

        query_vector = [float(format(x, '.5f')) for x in query_embedding]

        try:
            results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            matches = results.get('matches', [])
            logging.info("Found %d matches", len(matches))

            formatted_results = []
            for match in matches:
                meta = match.get('metadata', {})
                stored_text = meta.get('text', '')
                # Only include reasonably good matches
                if match.get('score', 0) > 0.6:  # Adjust threshold as needed
                    formatted_results.append({
                        'id': match.get('id'),
                        'score': match.get('score'),
                        'source': meta.get('source', 'Unknown'),
                        'text': stored_text,
                        'raw_text': stored_text,
                    })

            return formatted_results[:3]  # Return top 3 good matches
            
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []

    def clean_query(self, query: str) -> str:
        """Clean and potentially expand the query for better search."""
        query = query.lower().strip()
        
        # Remove question marks and common question words for better embedding
        query = re.sub(r'[?]', '', query)
        
        # Expand abbreviations
        expansions = {
            'shs': 'secondhand smoke',
            'ets': 'environmental tobacco smoke',
            'second hand smoke': 'secondhand smoke',
            'passive smoke': 'secondhand smoke'
        }
        
        for abbr, expansion in expansions.items():
            if abbr in query:
                query = query.replace(abbr, expansion)
                
        return query

def main():
    """Main orchestrated chatbot loop with conversation awareness."""
    if not chatbot_orchestrator:
        print("⚠️  chatbot_orchestrator not available, using basic SemanticSearch mode")
        searcher = SemanticSearch()
        conversation_history = []
        orchestrator = None
    else:
        from chatbot_orchestrator import ChatbotOrchestrator
        orchestrator = ChatbotOrchestrator()
        conversation_history = []
    
    print("🚭 Smoking Cessation Chatbot Ready!")
    print("💬 I can help with greetings, general questions, and detailed smoking cessation advice")
    print("💡 Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("\n🧑‍💻 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for chatting! Stay healthy!")
                break
            
            # Process the message using orchestrator if available
            if chatbot_orchestrator:
                result = orchestrator.process_message(user_input, conversation_history)
                response = result.get('response', 'Unable to process message')
                conversation_state = result.get('conversation_state', 'UNKNOWN')
                search_performed = result.get('search_performed', False)
                search_results = result.get('search_results', [])
            else:
                # Fallback to basic SemanticSearch
                print("🔍 Searching for relevant information...")
                results = searcher.search(user_input)
                response = searcher.generate_answer(user_input, results, chat_history=conversation_history)
                conversation_state = "BASIC_MODE"
                search_performed = True
                search_results = results
            
            # Update conversation history
            conversation_history.append({'role': 'user', 'text': user_input})
            conversation_history.append({'role': 'assistant', 'text': response})
            
            # Display response
            print(f"\n🤖 Assistant: {response}")
            
            # Show debug info if needed
            if os.environ.get('DEBUG_CONVERSATION'):
                print(f"\n[DEBUG] State: {conversation_state}")
                print(f"[DEBUG] Search performed: {search_performed}")
                if search_results:
                    print(f"[DEBUG] Found {len(search_results)} relevant documents")
                    
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Thank you for using the chatbot!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":

    main()
