"""
Create embeddings from the smoking cessation dataset using sentence-transformers
and prepare for Pinecone upload.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATASET_DIR = "smoking_cessation_dataset"
OUTPUT_FILE = os.path.join(DATASET_DIR, "pinecone_dataset.json")
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, lightweight embedding model (384 dimensions)
BATCH_SIZE = 32


class EmbeddingGenerator:
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
        return embeddings.tolist()

    def create_vector_records(
        self,
        chunks: List[Dict[str, Any]],
        source_type: str = "pdf"
    ) -> List[Dict[str, Any]]:
        """
        Create vector records with embeddings from chunks.
        
        Args:
            chunks: List of chunk dicts with 'text', 'source', 'metadata' fields
            source_type: Type of source (pdf, web, image)
            
        Returns:
            List of records with id, values (embeddings), and metadata
        """
        if not chunks:
            logger.warning(f"No chunks provided for source type: {source_type}")
            return []

        # Extract text from chunks
        texts = [chunk.get("text", "") for chunk in chunks]
        texts = [t for t in texts if t.strip()]  # Filter empty texts

        if not texts:
            logger.warning(f"No valid texts found in {len(chunks)} chunks")
            return []

        # Generate embeddings
        embeddings = self.generate_embeddings(texts)

        # Create vector records
        vector_records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{source_type}_{len(vector_records)}"
            
            # Extract metadata
            text = chunk.get("text", "")
            source = chunk.get("source", f"{source_type}_document")
            metadata = chunk.get("metadata", {})
            
            record = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": text[:500],  # Truncate text for metadata
                    "source": source,
                    "source_type": source_type,
                    "chunk_index": metadata.get("chunk_index", i),
                    "document_name": metadata.get("document_name", ""),
                }
            }
            vector_records.append(record)

        logger.info(f"Created {len(vector_records)} vector records for {source_type} source")
        return vector_records

    def process_dataset(self) -> List[Dict[str, Any]]:
        """
        Process all extracted data from the dataset folder and create embeddings.
        
        Returns:
            List of all vector records ready for Pinecone upload
        """
        all_vectors = []

        # Process PDF chunks
        pdf_file = os.path.join(DATASET_DIR, "pdf_extracted.json")
        if os.path.exists(pdf_file):
            logger.info(f"Processing PDF chunks from {pdf_file}")
            try:
                with open(pdf_file, 'r', encoding='utf-8') as f:
                    pdf_chunks = json.load(f)
                logger.info(f"Loaded {len(pdf_chunks)} PDF chunks")
                pdf_vectors = self.create_vector_records(pdf_chunks, source_type="pdf")
                all_vectors.extend(pdf_vectors)
            except Exception as e:
                logger.error(f"Error processing PDF chunks: {e}")
        else:
            logger.warning(f"PDF chunks file not found: {pdf_file}")

        # Process web chunks
        web_file = os.path.join(DATASET_DIR, "web_scraped.json")
        if os.path.exists(web_file):
            logger.info(f"Processing web chunks from {web_file}")
            try:
                with open(web_file, 'r', encoding='utf-8') as f:
                    web_chunks = json.load(f)
                logger.info(f"Loaded {len(web_chunks)} web chunks")
                web_vectors = self.create_vector_records(web_chunks, source_type="web")
                all_vectors.extend(web_vectors)
            except Exception as e:
                logger.error(f"Error processing web chunks: {e}")
        else:
            logger.warning(f"Web chunks file not found: {web_file}")

        # Process image chunks (if available)
        image_file = os.path.join(DATASET_DIR, "image_extracted.json")
        if os.path.exists(image_file):
            logger.info(f"Processing image chunks from {image_file}")
            try:
                with open(image_file, 'r', encoding='utf-8') as f:
                    image_chunks = json.load(f)
                if image_chunks:
                    logger.info(f"Loaded {len(image_chunks)} image chunks")
                    image_vectors = self.create_vector_records(image_chunks, source_type="image")
                    all_vectors.extend(image_vectors)
                else:
                    logger.info("No image chunks available")
            except Exception as e:
                logger.error(f"Error processing image chunks: {e}")
        else:
            logger.info("Image chunks file not found (expected if no images extracted)")

        logger.info(f"Total vectors created: {len(all_vectors)}")
        return all_vectors

    def save_vectors(self, vectors: List[Dict[str, Any]], output_file: str = OUTPUT_FILE):
        """Save vectors to a JSON file for Pinecone upload."""
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        logger.info(f"Saving {len(vectors)} vectors to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vectors, f, indent=2)
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
        logger.info(f"Vectors saved. File size: {file_size:.2f} MB")
        return output_file



def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("STARTING EMBEDDING GENERATION")
    logger.info("=" * 70)

    try:
        # Initialize embedding generator
        generator = EmbeddingGenerator(MODEL_NAME)

        # Process dataset and generate embeddings
        logger.info("Processing dataset folder and generating embeddings...")
        vectors = generator.process_dataset()

        if not vectors:
            logger.error("No vectors were generated. Check dataset files.")
            return False

        # Save vectors to file
        output_file = generator.save_vectors(vectors, OUTPUT_FILE)

        logger.info("=" * 70)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total vectors created: {len(vectors)}")
        logger.info(f"Embedding dimension: {generator.embedding_dim}")
        logger.info(f"Output file: {output_file}")
        logger.info("=" * 70)

        return True

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
