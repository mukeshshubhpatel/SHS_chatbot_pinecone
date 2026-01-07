from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import time
import logging

# Import orchestrator and conversation manager
try:
    from chatbot_orchestrator import chatbot_orchestrator
    logger_init = logging.getLogger(__name__)
    logger_init.info("chatbot_orchestrator imported successfully")
except Exception as e:
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Failed to import chatbot_orchestrator: {e}")
    chatbot_orchestrator = None

try:
    from conversation_manager import ConversationState
except Exception as e:
    ConversationState = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smoking Cessation Chatbot API",
    description="Advanced conversational AI for smoking cessation and child health",
    version="1.0.0"
)


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ConversationRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = None
    top_k: int = 5
    backend: Optional[str] = "ollama"  # "ollama", "hf", or "force-fallback"


class ConversationResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[Dict]
    confidence: float
    response_time: float
    status: str = "success"
    conversation_state: Optional[str] = None


# In-memory conversation store (use database in production)
conversations = {}

# Initialize searcher
_searcher = None
_searcher_error = None


def get_searcher():
    """Get or initialize the SemanticSearch instance."""
    global _searcher, _searcher_error
    if _searcher is None:
        try:
            from new_semetic_search import SemanticSearch
            _searcher = SemanticSearch()
            logger.info("SemanticSearch initialized successfully")
        except Exception as e:
            _searcher_error = str(e)
            logger.error(f"Failed to initialize SemanticSearch: {e}")
    return _searcher


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting up Smoking Cessation Chatbot API")
    get_searcher()


@app.post("/chat", response_model=ConversationResponse)
async def chat_endpoint(request: ConversationRequest):
    """
    Main chat endpoint for conversational interactions using the orchestrator.
    
    Accepts a message and optional conversation history, returns an evidence-based response.
    """
    start_time = datetime.now()
    
    try:
        # Convert to internal format
        chat_history = []
        if request.chat_history:
            for msg in request.chat_history:
                chat_history.append({
                    'role': msg.role,
                    'text': msg.content
                })
        
        # Process through orchestrator
        if not chatbot_orchestrator:
            return ConversationResponse(
                response="Chatbot orchestrator is not available",
                conversation_id=request.conversation_id or str(uuid.uuid4()),
                sources=[],
                confidence=0.0,
                response_time=0.0,
                status="error",
                conversation_state="error"
            )
        
        result = chatbot_orchestrator.process_message(request.message, chat_history)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare sources from search results
        sources = []
        if result.get('search_results'):
            sources = [{
                "text": r.get('text', '')[:200] + '...',
                "score": r.get('score', 0),
                "source": r.get('source', 'Unknown')
            } for r in result['search_results'][:3]]
        
        return ConversationResponse(
            response=result['response'],
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            sources=sources,
            confidence=0.8 if result.get('search_performed') else 0.5,
            response_time=response_time,
            status="success",
            conversation_state=result['conversation_state']
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return ConversationResponse(
            response=f"I apologize, but I encountered an error: {str(e)}",
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            sources=[],
            confidence=0.0,
            response_time=(datetime.now() - start_time).total_seconds(),
            status="error",
            conversation_state="error"
        )


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Retrieve the full conversation history for a given conversation ID.
    """
    if conversation_id in conversations:
        return {
            "conversation_id": conversation_id,
            "messages": conversations[conversation_id],
            "message_count": len(conversations[conversation_id])
        }
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "message_count": 0,
        "status": "not_found"
    }


@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and its history.
    """
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {
            "status": "success",
            "message": f"Conversation {conversation_id} deleted"
        }
    return {
        "status": "not_found",
        "message": f"Conversation {conversation_id} not found"
    }


@app.post("/conversation/analyze")
async def analyze_conversation(conversation_history: List[ChatMessage]):
    """
    Analyze conversation history and provide insights.
    
    Returns a summary of the conversation including topics discussed,
    message count, and conversation patterns.
    """
    try:
        if not chatbot_orchestrator:
            return {
                "error": "Chatbot orchestrator not available",
                "total_messages": len(conversation_history)
            }
        
        # Convert to internal format
        internal_history = []
        for msg in conversation_history:
            internal_history.append({
                'role': msg.role,
                'text': msg.content
            })
        
        # Get conversation summary from orchestrator
        summary = chatbot_orchestrator.get_conversation_summary(internal_history)
        
        return {
            "summary": summary,
            "total_messages": len(conversation_history),
            "primary_topics": summary.get('smoking_topics', []),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error analyzing conversation: {e}", exc_info=True)
        return {
            "error": str(e),
            "total_messages": len(conversation_history),
            "status": "error"
        }

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and dependencies are working.
    """
    searcher = get_searcher()
    status = {
        "api": "healthy",
        "semantic_search": "healthy" if searcher else "unhealthy",
        "conversations_stored": len(conversations),
        "timestamp": datetime.now().isoformat()
    }
    
    if not searcher and _searcher_error:
        status["semantic_search_error"] = _searcher_error
    
    return status


@app.get("/")
async def root():
    """
    API documentation and available endpoints.
    """
    return {
        "name": "Smoking Cessation Chatbot API",
        "version": "1.0.0",
        "description": "Advanced conversational AI for smoking cessation and child health",
        "endpoints": {
            "POST /chat": {
                "description": "Send a message and get an evidence-based response",
                "parameters": {
                    "message": "Your question (required)",
                    "conversation_id": "ID for conversation context (optional)",
                    "chat_history": "Previous messages for context (optional)",
                    "top_k": "Number of search results (default: 5)",
                    "backend": "Generation backend: 'ollama', 'hf', or 'force-fallback' (default: 'ollama')"
                }
            },
            "GET /conversation/{id}": {
                "description": "Retrieve conversation history"
            },
            "DELETE /conversation/{id}": {
                "description": "Delete a conversation"
            },
            "GET /health": {
                "description": "Health check endpoint"
            },
            "GET /": {
                "description": "This documentation"
            }
        },
        "features": [
            "Conversational context awareness",
            "Evidence-based responses",
            "Multiple generation backends",
            "Medical accuracy validation",
            "Response quality enhancement"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)