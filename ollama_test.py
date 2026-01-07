import requests
import json
import os
from typing import List, Dict

try:
    # Import the semantic search helper
    from new_semetic_search import SemanticSearch
    from print_matched_outputs_income import get_outputs_for_texts
except Exception:
    SemanticSearch = None
    get_outputs_for_texts = None

try:
    from prompt_engine import prompt_engine
except Exception:
    prompt_engine = None

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))


def get_validation_sentence() -> str:
    return os.environ.get('VALIDATION_SENTENCE', "")


def ensure_validation_prefix(text: str) -> str:
    """Only add validation prefix if explicitly configured in environment"""
    v = get_validation_sentence().strip()
    if not v or not text:
        return text
    if text.startswith(v):
        return text
    return v + " " + text


def build_prompt_from_context(question: str, contexts: List[dict], chat_history: List[dict] = None) -> str:
    """Build advanced conversational prompt using prompt_engine if available.
    
    Falls back to basic prompt if prompt_engine is not available.
    """
    if prompt_engine:
        return prompt_engine.build_conversational_prompt(question, contexts, chat_history)
    
    # Fallback to basic prompt
    ctx_lines = []
    for i, c in enumerate(contexts[:3], start=1):
        excerpt = (c.get('extracted') or c.get('text') or '').strip()
        if excerpt:
            short = excerpt if len(excerpt) <= 500 else excerpt[:500] + '...'
            ctx_lines.append(f"[{i}] {short}")

    context_block = "\n\n".join(ctx_lines) if ctx_lines else "(no supporting documents available)"

    # Conversation history block
    conv_block = ''
    if chat_history:
        recent = chat_history[-6:]
        conv_lines = []
        for turn in recent:
            role = turn.get('role', 'user')
            text = (turn.get('text') or '').strip()
            if not text:
                continue
            label = 'User' if role == 'user' else 'Assistant'
            conv_lines.append(f"{label}: {text}")
        if conv_lines:
            conv_block = "Conversation history:\n" + "\n".join(conv_lines) + "\n\n"

    instruction = (
        f"You are a health information assistant providing evidence-based information about smoking cessation and child health. "
        f"You MUST use ONLY the context provided below to answer questions. DO NOT say you cannot provide information. "
        f"DO NOT offer to provide general information - provide the SPECIFIC information from the context directly.\\n\\n"
        f"Based on the research and evidence in the context, provide a clear, detailed answer. "
        f"Include specific facts, statistics, and recommendations from the context. "
        f"Format your response with 2-3 numbered practical steps when appropriate. "
        f"Be informative, direct, and helpful. Use the conversation history to provide contextual follow-ups."
    )

    prompt = (
        f"{conv_block}Context:\n{context_block}\n\n"
        f"User question: {question}\n\n"
        f"Instructions:\n{instruction}\n\nAnswer:"
    )
    return prompt


def ask_ollama(question: str, contexts: List[dict] = None, chat_history: List[dict] = None):
    """Call Ollama with a grounded prompt built from semantic search results.
    
    Tries multiple models in sequence if configured, falls back to deterministic
    composer if all models fail.
    """
    searcher = None
    
    # If no contexts provided, try to run semantic search
    if contexts is None and SemanticSearch is not None:
        try:
            searcher = SemanticSearch()
            contexts = searcher.search(question, top_k=3)
            # Attach extracted outputs when available
            if get_outputs_for_texts is not None and contexts:
                raw_texts = [r.get('raw_text') or r.get('text') or '' for r in contexts]
                outputs_map = get_outputs_for_texts(raw_texts)
                for result in contexts:
                    key = (result.get('raw_text') or result.get('text') or '').strip().lower()
                    m = outputs_map.get(key)
                    if m and m.get('output'):
                        result['extracted'] = m.get('output')
        except Exception as e:
            print(f"Warning: semantic search failed: {e}")
            contexts = []

    # Build prompt (include chat_history)
    prompt = build_prompt_from_context(question, contexts or [], chat_history=chat_history)

    base_payload = {
        "prompt": prompt,
        "max_tokens": 400,
        "temperature": float(os.environ.get("OLLAMA_TEMP", 0.3)),
    }

    # Build ordered list of models to try: primary + any fallbacks from env
    primary = OLLAMA_MODEL
    fallbacks = os.environ.get("OLLAMA_MODEL_FALLBACKS", "")
    models_to_try = [primary]
    if fallbacks:
        extra = [m.strip() for m in fallbacks.split(',') if m.strip()]
        models_to_try.extend(extra)

    last_err_text = None
    for model_id in models_to_try:
        payload = dict(base_payload)
        payload["model"] = model_id
        try:
            # Use configurable timeout (seconds). ReadTimeouts for streaming can occur
            # if the model is slow to produce output — increase via env OLLAMA_TIMEOUT.
            resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=OLLAMA_TIMEOUT)
        except requests.exceptions.ReadTimeout as e:
            # Server accepted the connection but took too long to send data
            print(f"Warning: Ollama read timed out for model {model_id}: {e}")
            last_err_text = str(e)
            continue
        except requests.exceptions.RequestException as e:
            # Network or connection error — try next model
            print(f"Warning: failed to contact Ollama server for model {model_id}: {e}")
            last_err_text = str(e)
            continue

        if resp.status_code != 200:
            # Try to parse error JSON for known memory issue
            err_text = None
            try:
                err_json = resp.json()
                err_text = err_json.get('error') or json.dumps(err_json)
            except Exception:
                err_text = resp.text

            last_err_text = err_text
            # If Ollama indicates model too large for system memory, try next model
            if err_text and 'requires more system memory' in str(err_text).lower():
                print(f"Warning: model '{model_id}' could not be loaded due to insufficient system memory; trying next model if available.")
                continue
            else:
                # Non-memory error — raise immediately
                raise RuntimeError(f"Ollama request failed for model {model_id}: {resp.status_code} {err_text}")

        # Successful response; stream and return
        out = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode('utf-8'))
            except Exception:
                continue
            if "response" in data:
                chunk = data["response"]
                print(chunk, end="", flush=True)
                out.append(chunk)

        print()  # newline after stream
        result_text = ''.join(out)
        print(f"\n[DEBUG] Full Ollama response length: {len(result_text)} chars")
        print(f"[DEBUG] First 200 chars: {result_text[:200]}")
        OPENING = get_validation_sentence()
        # If strict validation is enabled, require the response to start with the validation sentence
        strict = os.environ.get('VALIDATION_STRICT', '').strip() in ('1', 'true', 'True')
        if strict:
            if result_text and result_text.strip().startswith(OPENING):
                return result_text
            else:
                print(f"Warning: model {model_id} did not produce the required validation sentence natively; trying next model.")
                last_err_text = 'missing validation sentence in model output'
                continue
        # non-strict: ensure the validation prefix is present
        final_response = ensure_validation_prefix(result_text)
        print(f"[DEBUG] Returning response with length: {len(final_response)} chars")
        return final_response

    # If we reach here, all models failed (likely due to memory or connection issues)
    print('\nWarning: all attempted Ollama models failed to load or respond.')
    if last_err_text and 'requires more system memory' in str(last_err_text).lower():
        print('It appears your machine cannot load these models into memory. Consider using a smaller model or running Ollama with CPU mode.')

    # Fall back to deterministic composer if available
    if searcher is not None:
        try:
            print('Falling back to deterministic composer from indexed sources...')
            return ensure_validation_prefix(searcher.smart_fallback_response(question, contexts or []))
        except Exception as e:
            print(f"Fallback composition failed: {e}")

    return 'All generation attempts failed. Please try a smaller Ollama model or check your Ollama server.'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Ask an Ollama model grounded in your semantic search index")
    parser.add_argument('question', nargs='?', help='Question to ask the chatbot')
    args = parser.parse_args()
    # Interactive loop: if an initial question is provided, use it first, then continue prompting
    initial = args.question
    try:
        while True:
            if initial:
                q = initial.strip()
                initial = None
            else:
                q = input('Enter your question (or type "quit" to exit): ').strip()

            if not q:
                continue
            if q.lower() in ('quit', 'exit'):
                print('Goodbye')
                break

            try:
                _ = ask_ollama(q)
            except Exception as e:
                print(f"Error calling Ollama: {e}")

    except (EOFError, KeyboardInterrupt):
        print('\nExiting...')

