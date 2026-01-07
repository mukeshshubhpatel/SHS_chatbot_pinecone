# download_model_local.py
import os
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.join(current_dir, "all-MiniLM-L6-v2")

    print(f"=== Downloading model '{model_name}' into: {local_dir} ===")

    # Force download into local_dir instead of Hugging Face cache
    snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)

    # Load model from the local folder
    model = SentenceTransformer(local_dir)

    print("=== Model loaded successfully! ===")
    print("Embedding dimension:", model.get_sentence_embedding_dimension())

    # Quick test
    sentences = [
        "This is a test sentence.",
        "Converting PDF text into embeddings is useful for vector databases."
    ]
    embeddings = model.encode(sentences)
    print("Embeddings shape:", embeddings.shape)   # e.g. (2, 384)

if __name__ == "__main__":
    main()
