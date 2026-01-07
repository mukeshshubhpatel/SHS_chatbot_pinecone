import os
import json
import traceback
from typing import List, Dict, Any
from datetime import datetime

import streamlit as st

# Import conversation management and orchestration
try:
    from conversation_manager import conversation_manager, ConversationState
except Exception as e:
    st.warning(f"conversation_manager import failed: {e}")
    conversation_manager = None
    ConversationState = None

try:
    from chatbot_orchestrator import chatbot_orchestrator
except Exception as e:
    st.warning(f"chatbot_orchestrator import failed: {e}")
    chatbot_orchestrator = None

# Try to import your project's search/generation helpers
_ss_import_err = None
_ollama_import_err = None
_prompt_engine_err = None
SemanticSearch = None
generate_answer = None
ask_ollama = None
prompt_engine = None

try:
    from new_semetic_search import SemanticSearch
except Exception as e:
    _ss_import_err = str(e)

try:
    from ollama_test import ask_ollama
except Exception as e:
    _ollama_import_err = str(e)

try:
    from prompt_engine import prompt_engine
except Exception as e:
    _prompt_engine_err = str(e)


def display_conversational_ui():
    """Enhanced UI with conversation features"""
    st.markdown("""
    <style>
    /* White background theme */
    .stApp {
        background-color: #ffffff;
    }
    
    .user-message {
        background-color: #ffffff;
        color: #000000;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        border: 1px solid #e0e0e0;
    }
    .assistant-message {
        background-color: #ffffff;
        color: #000000;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        border: 1px solid #e0e0e0;
    }
    .conversation-container {
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 20px;
        background-color: #ffffff;
    }
    
    /* Black text styling */
    body {
        color: #000000;
    }
    
    .stMarkdown, .stText {
        color: #000000 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced conversation display
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("### Conversation")
        
        # Create conversation container
        conversation_html = "<div class='conversation-container'>"
        
        for i, message in enumerate(st.session_state.history[-10:]):  # Show last 10 messages
            if message['role'] == 'user':
                conversation_html += f"""
                <div class='user-message'>
                    <strong>You:</strong> {message['text']}
                </div>
                """
            else:
                conversation_html += f"""
                <div class='assistant-message'>
                    <strong>Assistant:</strong> {message['text']}
                </div>
                """
        
        conversation_html += "</div>"
        st.markdown(conversation_html, unsafe_allow_html=True)
        
        # Quick action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 New Topic"):
                st.session_state.history = []
                st.rerun()
        with col2:
            if st.button("📋 Get Action Steps"):
                # Add a prompt to get actionable steps
                if st.session_state.history:
                    last_query = "Provide specific actionable steps based on our conversation"
                    st.session_state.history.append({'role': 'user', 'text': last_query})
                    st.rerun()
        with col3:
            if st.button("ℹ️ More Info"):
                # Add a prompt for more detailed information
                if st.session_state.history:
                    last_query = "Provide more detailed medical information"
                    st.session_state.history.append({'role': 'user', 'text': last_query})
                    st.rerun()


def get_searcher():
    """Return a SemanticSearch instance or None if import/init failed."""
    if SemanticSearch is None:
        return None
    try:
        return SemanticSearch()
    except Exception as e:
        st.error(f"Failed to initialize SemanticSearch: {e}")
        return None


def run_system_health_check():
    """Run system diagnostics."""
    with st.expander("System Diagnostics"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Check prompt engine
            try:
                from prompt_engine import prompt_engine as pe
                st.success("✅ Prompt Engine: Loaded")
            except Exception as e:
                st.error(f"❌ Prompt Engine: {e}")
            
            # Check Ollama
            try:
                from ollama_test import ask_ollama
                st.success("✅ Ollama: Available")
            except Exception as e:
                st.warning(f"⚠️ Ollama: {e}")
        
        with col2:
            # Check search
            try:
                from new_semetic_search import SemanticSearch
                searcher = SemanticSearch()
                st.success("✅ Semantic Search: Working")
            except Exception as e:
                st.error(f"❌ Semantic Search: {e}")
            
            # Check basic imports
            try:
                import torch
                st.success("✅ PyTorch: Available")
            except Exception as e:
                st.warning(f"⚠️ PyTorch: {e}")


def extract_output_from_match(m: Dict[str, Any]) -> str:
    """Extract stored OUTPUT paragraph from match metadata."""
    if not isinstance(m, dict):
        return ''
    if 'extracted' in m and m['extracted']:
        return m['extracted']
    meta = m.get('metadata') or m.get('meta') or {}
    if isinstance(meta, dict):
        for k in ('extracted', 'OUTPUT', 'output', 'extraction'):
            if k in meta and meta[k]:
                return meta[k]
    return (m.get('text') or m.get('raw_text') or meta.get('text') or '')[:500]


def summarize_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize search matches for display."""
    out = []
    for m in matches:
        score = m.get('score') if isinstance(m.get('score'), (int, float)) else m.get('similarity') or m.get('distance')
        out.append({
            'id': m.get('id') or m.get('doc_id'),
            'score': score,
            'text_snippet': (m.get('text') or m.get('raw_text') or '')[:400],
            'extracted_OUTPUT': extract_output_from_match(m)
        })
    return out


def run_query(question: str, backend_choice: str, top_k: int, chat_history: List[Dict[str, Any]] = None):
    """Enhanced run_query with conversation support."""
    searcher = get_searcher()
    if searcher is None:
        diag_msg = "`new_semetic_search.SemanticSearch` could not be imported or initialized."
        extra = {}
        try:
            extra['import_error'] = _ss_import_err
        except Exception:
            extra['import_error'] = 'No import traceback available.'
        reply_json = {
            'question': question,
            'reply': diag_msg + ' See Diagnostics in the sidebar for details.',
            'matches': [],
            'diagnostics': extra,
        }
        return reply_json

    matches = searcher.search(question, top_k=top_k)
    summarized = summarize_matches(matches or [])
    contexts = matches[:top_k] if matches else []

    reply = None
    reply_json = None

    if backend_choice == 'ollama' and ask_ollama is not None:
        try:
            reply = ask_ollama(question, contexts=contexts, chat_history=chat_history)
        except Exception as e:
            st.warning(f"Ollama generation failed: {e}. Falling back to deterministic composer.")
            try:
                reply = searcher.smart_fallback_response(question, contexts or [], chat_history=chat_history)
            except Exception as e2:
                reply = f"Fallback composition also failed: {e2}"
    elif backend_choice == 'force-fallback':
        try:
            reply = searcher.smart_fallback_response(question, contexts or [], chat_history=chat_history)
        except Exception as e:
            reply = f"Fallback composition failed: {e}"
    else:
        # Try HF path via searcher.generate_answer
        try:
            if hasattr(searcher, 'generate_answer'):
                reply = searcher.generate_answer(question, contexts=contexts, chat_history=chat_history)
            else:
                reply = searcher.smart_fallback_response(question, contexts or [], chat_history=chat_history)
        except Exception as e:
            st.warning(f"Generation failed: {e}. Using deterministic fallback.")
            try:
                reply = searcher.smart_fallback_response(question, contexts or [], chat_history=chat_history)
            except Exception as e2:
                reply = f"Fallback composition also failed: {e2}"

    # Build JSON response
    reply_json = {
        'question': question,
        'reply': reply,
        'matches': summarized,
    }
    return reply_json


def display_enhanced_response(resp: Dict, show_analysis: bool):
    """Display response with enhanced features."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💡 Response")
        st.markdown(resp.get('reply', 'No response generated.'))
        
        if show_analysis:
            st.markdown("---")
            st.markdown("#### 📈 Response Analysis")
            st.info("Response analysis features coming soon.")
            
    with col2:
        st.markdown("### 📚 Supporting Sources")
        if resp.get('matches'):
            for i, match in enumerate(resp['matches'][:3], 1):
                with st.expander(f"Source {i}"):
                    st.markdown(f"**Excerpt:** {match.get('text_snippet', '')}")
                    if match.get('extracted_OUTPUT'):
                        st.markdown(f"**Key Points:** {match.get('extracted_OUTPUT')}")
                    score = match.get('score')
                    if score is not None:
                        st.caption(f"Relevance score: {score:.3f}")
        else:
            st.info("No supporting sources found.")


def process_user_question(question: str, backend: str, top_k: int, show_analysis: bool):
    """Process user question with enhanced features."""
    with st.spinner("🔍 Searching medical information..."):
        try:
            # Append user message to history
            st.session_state.history.append({'role': 'user', 'text': question})
            
            # Get response
            resp = run_query(question, backend, top_k, chat_history=st.session_state.history)
            
            # Append assistant response
            assistant_text = resp.get('reply', 'No reply generated.')
            st.session_state.history.append({'role': 'assistant', 'text': assistant_text})
            
            # Display enhanced response
            display_enhanced_response(resp, show_analysis)
            
        except Exception as e:
            st.error(f"Error processing your question: {e}")
            st.code(traceback.format_exc())


def get_current_time():
    """Get current time string"""
    return datetime.now().strftime("%H:%M:%S")


def display_conversation_interface():
    """Display the main conversation interface"""
    
    # Display conversation first
    
    if not st.session_state.conversation_history:
        st.info("💡 Start by asking a question about smoking cessation!")
    else:
        display_conversation_history()
    
    # Input section after conversation
    st.markdown("---")
    
    # Add JavaScript for Enter key handling
    st.markdown("""
    <script>
    const doc = window.parent.document;
    const inputs = doc.querySelectorAll('input[type="text"]');
    inputs.forEach(input => {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                const sendBtn = doc.querySelector('button[kind="primary"]');
                if (sendBtn) sendBtn.click();
            }
        });
    });
    </script>
    """, unsafe_allow_html=True)
    
    user_input = st.text_input(
        "💬 Type your message here:",
        placeholder="Say hello, ask a question, or type your message... (Press Enter to send)",
        key="user_input",
        on_change=None
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_clicked = st.button("🚀 Send", use_container_width=True, type="primary")
        if send_clicked or (user_input and user_input != st.session_state.get('last_input', '')):
            if user_input.strip():
                st.session_state['last_input'] = user_input
                process_user_message(user_input.strip())
            else:
                st.warning("Please enter a message first.")
    with col2:
        if st.button("🎯 Analyze Last Message", use_container_width=True):
            if st.session_state.conversation_history:
                analyze_last_interaction()


def process_user_message(message: str):
    """Process user message with smart routing"""
    with st.spinner("🤔 Analyzing your message..."):
        try:
            if not chatbot_orchestrator:
                st.error("❌ Chatbot orchestrator not available")
                return
            
            # Process through orchestrator
            result = chatbot_orchestrator.process_message(
                message, 
                st.session_state.conversation_history
            )
            
            # Update conversation history
            st.session_state.conversation_history.append({
                'role': 'user', 
                'text': message,
                'timestamp': get_current_time()
            })
            
            st.session_state.conversation_history.append({
                'role': 'assistant', 
                'text': result['response'],
                'conversation_state': result['conversation_state'],
                'timestamp': get_current_time()
            })
            
            # Save sources from the result so the sidebar can display them
            try:
                st.session_state['last_response_sources'] = result.get('search_results') or result.get('sources') or []
            except Exception:
                st.session_state['last_response_sources'] = []

            # Show technical details if enabled
            if st.session_state.show_technical:
                with st.expander("🔧 Technical Details"):
                    st.json({
                        'conversation_state': result['conversation_state'],
                        'needs_database_lookup': result.get('needs_database_lookup', False),
                        'search_performed': result.get('search_performed', False),
                        'context': result.get('context', {})
                    })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing message: {e}")


def display_conversation_history():
    """Display formatted conversation history"""
    for i, message in enumerate(st.session_state.conversation_history):
        if message['role'] == 'user':
            st.markdown(f"""
            <div style='background-color: #ffffff; color: #000000; padding: 12px; border-radius: 15px; margin: 8px 0; border: 1px solid #e0e0e0;'>
                <strong style='color: #000000;'>👤 You:</strong> <span style='color: #000000;'>{message['text']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Add indicator for database usage
            state = message.get('conversation_state', 'unknown')
            indicator = "🔍" if state == "smoking_question" else "💬"
            
            st.markdown(f"""
            <div style='background-color: #ffffff; color: #000000; padding: 12px; border-radius: 15px; margin: 8px 0; border: 1px solid #e0e0e0;'>
                <strong style='color: #000000;'>{indicator} Assistant:</strong> <span style='color: #000000;'>{message['text']}</span>
            </div>
            """, unsafe_allow_html=True)


def analyze_last_interaction():
    """Analyze the last interaction"""
    if st.session_state.conversation_history and conversation_manager:
        last_user_msg = None
        for msg in reversed(st.session_state.conversation_history):
            if msg['role'] == 'user':
                last_user_msg = msg['text']
                break
        
        if last_user_msg:
            state, context = conversation_manager.analyze_message(
                last_user_msg, 
                st.session_state.conversation_history
            )
            
            st.info(f"**Last message analysis:**")
            st.write(f"**State:** {state}")
            st.write(f"**Intent:** {context.get('intent', 'unknown')}")
            if context.get('keywords'):
                st.write(f"**Keywords:** {', '.join(context['keywords'])}")


def run_query(question: str, backend_choice: str, top_k: int, chat_history: List[Dict[str, Any]] = None):
    """Enhanced run_query that uses the smart orchestrator"""
    try:
        if chatbot_orchestrator:
            result = chatbot_orchestrator.process_message(question, chat_history or [])
            return {
                'question': question,
                'reply': result['response'],
                'matches': result.get('search_results', []),
                'conversation_state': result['conversation_state']
            }
        else:
            return {
                'question': question,
                'reply': "I apologize, but the chatbot orchestrator is not available.",
                'matches': [],
                'conversation_state': 'error'
            }
    except Exception as e:
        return {
            'question': question,
            'reply': f"I apologize, but I encountered an error: {str(e)}",
            'matches': [],
            'conversation_state': 'error'
        }


# Main UI - Initialize session state if not already done
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = chatbot_orchestrator
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'show_technical' not in st.session_state:
    st.session_state.show_technical = False


# Page configuration
st.set_page_config(
    page_title="Smart Smoking Cessation Chatbot", 
    layout="wide", 
    page_icon="🚭"
)

st.title("🚭 Smoke-free Parents, Healthy Kids")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = chatbot_orchestrator
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'show_technical' not in st.session_state:
    st.session_state.show_technical = False

# Enhanced sidebar
with st.sidebar:
    st.header("🎛️ Chat Settings")
    
    st.markdown("### 💬 Conversation Mode")
    enable_smart_routing = st.checkbox("Enable Smart Conversation Routing", value=True)
    show_conversation_analysis = st.checkbox("Show Conversation Analysis", value=False)
    st.session_state.show_technical = show_conversation_analysis
    # Supporting Source shown near Conversation Mode for quick visibility
    sources = st.session_state.get("last_response_sources")
    if sources:
        st.markdown("### Supporting Source")
        for i, s in enumerate(sources[:5], start=1):
            # Show generic source labels instead of internal IDs
            title = f"Source {i}"
            snippet = s.get("text") or s.get("text_snippet") or s.get("extracted_OUTPUT") or ""
            score = s.get("score")
            with st.expander(title):
                if snippet:
                    st.write(snippet)
                if score is not None:
                    st.caption(f"Relevance score: {score}")
    
    st.markdown("### 🔍 Search Settings")
    search_aggressiveness = st.select_slider(
        "Search Aggressiveness",
        options=["Minimal", "Balanced", "Comprehensive"],
        value="Balanced"
    )
    
    st.markdown("---")
    st.header("📊 Conversation Insights")
    
    if st.button("🔄 Analyze Conversation"):
        if chatbot_orchestrator:
            summary = chatbot_orchestrator.get_conversation_summary(
                st.session_state.conversation_history
            )
            st.write("**Conversation Summary:**")
            st.json(summary)
    
    if st.button("🧹 Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

# Main chat interface
display_conversation_interface()

st.markdown("---")
st.markdown("**Smoking Cessation Bot** — *Evidence-based, compassionate guidance for smoking cessation.*")
