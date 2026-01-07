from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class EnhancedSearch:
    def __init__(self):
        self.api_key = "pcsk_5Q3MCs_Asywku5gWesRHs5GedYGP3RSpQmQiTdjGNBD6EbhJfwjLJH2rgV6H6tssk4U2Mc"
        self.index_name = "chatbot"
        self.embedding_model_name = "BAAI/bge-large-en"
        self.llm_model_name = "Qwen/Qwen1.5-7B-Chat"
        self.init_models()

    def init_models(self):
        """Initialize both embedding and LLM models."""
        try:
            logging.info("Initializing models...")
            # Initialize Pinecone
            pc = Pinecone(api_key=self.api_key)
            self.index = pc.Index(self.index_name)

            # Initialize LLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logging.info("Models initialized successfully")
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            raise

    def generate_response(self, query, context_texts):
        """Generate a response using the LLM with retrieved contexts."""
        try:
            # Prepare prompt with context and specific formatting
            prompt = f"""Below is information about smoking cessation and health. Please provide a comprehensive answer about {query}.

Context:
{' '.join(context_texts)}

Question: {query}

Answer: Let me explain this based on the available information."""

            # Generate response with proper attention mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )
            
            # Generate with better parameters
            outputs = self.llm.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Process response
            response = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up response by removing the prompt
            response = response.replace(prompt, "").strip()
            
            # Format response in markdown
            formatted_response = f"""# Answer
{response}

## Sources
The above information is based on:
"""
            return formatted_response
            
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            raise

    def search_and_respond(self, query, top_k=3):
        """Perform semantic search and generate LLM response."""
        try:
            logging.info(f"Processing query: {query}")
            
            # Get embeddings and search
            from upload_vector_db import VectorManager
            vector_manager = VectorManager()
            vector = vector_manager.get_embedding(query)
            
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True
            )

            # Extract unique context texts
            context_texts = []
            seen_texts = set()
            for match in results['matches']:
                text = match['metadata'].get('text', '')
                if text and text not in seen_texts:
                    context_texts.append(text)
                    seen_texts.add(text)

            # Generate enhanced response
            if context_texts:
                response = self.generate_response(query, context_texts)
                return {
                    'answer': response,
                    'sources': [match['metadata'].get('source', 'Unknown') for match in results['matches']]
                }
            else:
                return {'answer': "No relevant information found.", 'sources': []}

        except Exception as e:
            logging.error(f"Search and respond failed: {e}")
            raise

def main():
    searcher = EnhancedSearch()
    
    while True:
        query = input("\nEnter your question about smoking cessation (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        try:
            result = searcher.search_and_respond(query)
            print("\n" + "=" * 50)
            print(result['answer'])
            print("\nReferences:")
            for source in result['sources']:
                print(f"* {source}")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()