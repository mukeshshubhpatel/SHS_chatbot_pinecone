from sentence_transformers import SentenceTransformer
from upload_vector_db import Pinecone
import uuid
import logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Config ---
PINECONE_API_KEY = "pcsk_5Q3MCs_Asywku5gWesRHs5GedYGP3RSpQmQiTdjGNBD6EbhJfwjLJH2rgV6H6tssk4U2Mc"
INDEX_NAME = "chatbot"

pc = Pinecone(api_key="pcsk_5Q3MCs_Asywku5gWesRHs5GedYGP3RSpQmQiTdjGNBD6EbhJfwjLJH2rgV6H6tssk4U2Mc")
index = pc.Index("chatbot")

# --- Init ---
try:
    logging.info("Initializing Pinecone and embedding model...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Initialization successful.")
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    raise

# --- Functions ---
def upsert_texts(texts, metadatas=None):
    logging.info(f"Starting upsert for {len(texts)} texts.")
    try:
        if metadatas is None:
            metadatas = [{}] * len(texts)
        vectors = embedding_model.encode(texts, show_progress_bar=False)
        logging.info(f"Vector creation successful: {len(vectors)} vectors created.")
        payload = [
            {"id": str(uuid.uuid4()), "values": vec.tolist(), "metadata": meta}
            for vec, meta in zip(vectors, metadatas)
        ]
        logging.info(f"Payload prepared with {len(payload)} items. Upserting to Pinecone...")
        response = index.upsert(payload)
        logging.info(f"Upsert successful. Pinecone response: {response}")
    except Exception as e:
        logging.error(f"Error during upsert_texts: {e}")
        raise

def search_similar_docs(query, top_k=5):
    logging.info(f"Searching for similar docs for query: '{query}' (top_k={top_k})")
    try:
        vector = embedding_model.encode([query])[0]
        logging.info("Query vector created successfully.")
        results = index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True)
        logging.info(f"Search successful. {len(results['matches'])} matches found.")
        return [
            match["metadata"].get("text", "") for match in results["matches"]
        ]
    except Exception as e:
        logging.error(f"Error during search_similar_docs: {e}")
        raise
