import logging
import argparse
import json
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Configuration
PINECONE_NAMESPACE = "smoking-cessation"  # Default namespace for Pinecone uploads
EMBEDDING_FILE = "smoking_cessation_dataset/pinecone_dataset_1024d.json"  # Updated to 1024D embeddings


class VectorManager:
    def __init__(self):
        pass

    def connect_pinecone(self):
        """Initialize Pinecone connection (stub for now)."""
        logger.info("[stub] connect_pinecone called - no-op in stub implementation")
        # In production, initialize real Pinecone client here
        # from pinecone import Pinecone
        # self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME", "chatbot"))

    def validate_vector_dimension(self, vector, vec_id):
        """Validate vector dimension matches expected size."""
        # For now, just check it's a list
        if not isinstance(vector, (list, tuple)):
            raise ValueError(f"Vector {vec_id} is not a list/tuple")

    def load_embeddings_from_file(self, file_path):
        """Load pre-computed embeddings from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                vectors = json.load(f)
            logger.info(f"Loaded {len(vectors)} vectors from {file_path}")
            return vectors
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {e}")
            return []

    def load_embeddings_from_folder(self):
        """Load embeddings from default dataset folder."""
        return self.load_embeddings_from_file(EMBEDDING_FILE)

    def upload_to_pinecone(self, vectors, delete_existing: bool = True):
        """Upload embeddings to Pinecone in batches with proper metadata.

        delete_existing: when True, attempt to delete existing vectors before upserting.
        """
        logger.info(f"Mock upload: Would upload {len(vectors)} vectors to Pinecone")
        logger.info("Note: Real Pinecone integration requires PINECONE_API_KEY environment variable")
        logger.info("To enable real Pinecone uploads:")
        logger.info("  1. Set environment variable: PINECONE_API_KEY=<your-api-key>")
        logger.info("  2. Set environment variable: PINECONE_INDEX_NAME=<your-index-name>")
        logger.info("  3. Uncomment the real Pinecone code in connect_pinecone()")
        
        # Mock upload simulation
        try:
            if not vectors:
                logger.warning("No vectors to upload")
                return False
            
            # Simulate batch upload
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                logger.info(f"Mock batch {i//batch_size + 1}: {len(batch)} vectors ready for upload")
            
            logger.info(f"Successfully simulated upload of {len(vectors)} total vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload vectors: {e}")
            return False

    def process_jsonl(self):
        """Process JSONL file and create vectors."""
        vectors = []
        jsonl_path = "smoking_cessation_large_dataset.jsonl"
        
        try:
            if os.path.exists(jsonl_path):
                logging.info(f"Processing {jsonl_path}...")
                with open(jsonl_path, 'r', encoding='utf-8') as file:
                    for idx, line in enumerate(file):
                        qa_pair = json.loads(line)
                        query = qa_pair['input']
                        response = qa_pair['output']
                        
                        # Get embedding for the query
                        vector = self.get_embedding(query)
                        
                        vectors.append({
                            "id": f"qa_{idx}",
                            "values": vector,
                            "metadata": {
                                "text": response,
                                "source": "QA Dataset",
                                "query": query
                            }
                        })
                        logging.info(f"Processed QA pair {idx}: {query[:50]}...")
                
                logging.info(f"Successfully processed {len(vectors)} QA pairs")
                return vectors
            else:
                logging.warning(f"JSONL file not found: {jsonl_path}")
                return []
                
        except Exception as e:
            logging.error(f"Error processing JSONL: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload embeddings to Pinecone from a JSON file or generated SHS folder.")
    parser.add_argument("--file", default=EMBEDDING_FILE, help="Path to embedding JSON file (list of records).")
    parser.add_argument("--shs-folder", default="", help="Path to a folder of SHS json/jsonl files to generate embeddings from (overrides --file when provided).")
    parser.add_argument("--no-delete", action="store_true", help="Don't delete existing vectors before uploading.")
    args = parser.parse_args()

    try:
        vector_manager = VectorManager()
        vector_manager.connect_pinecone()

        # Load embeddings: priority order
        # 1) If --shs-folder provided: generate embeddings from that folder using create_embedding.process_shs_json
        # 2) Else if --file provided: load embeddings from the file
        # 3) Else: attempt to generate via default folder behavior
        vectors = []
        if args.shs_folder:
            shs_folder = args.shs_folder
            logging.info(f"Generating embeddings from SHS folder: {shs_folder}")
            try:
                import create_embedding as ce
                tokenizer, model = ce.load_model()
                vectors = ce.process_shs_json(shs_folder, tokenizer, model)
            except Exception as ex:
                logging.error(f"Failed to generate embeddings from SHS folder {shs_folder}: {ex}")
                vectors = []
        elif args.file:
            vectors = vector_manager.load_embeddings_from_file(args.file)
        else:
            vectors = vector_manager.load_embeddings_from_folder()

        if not vectors:
            logging.warning("No vectors to upload. Exiting.")
            exit()

        # Upload to Pinecone; skip delete if --no-delete provided
        vector_manager.upload_to_pinecone(vectors, delete_existing=not args.no_delete)
        logging.info("Process completed successfully!")

    except Exception as e:
        logging.error(f"Script failed: {e}")
