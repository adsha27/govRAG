from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the Embedder with a specified SentenceTransformer model.
        
        Args:
            model_name (str): The model name to load from SentenceTransformers.
        """
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: list) -> list:
        """
        Generates embeddings for a list of text strings.
        
        Args:
            texts (list): List of text strings.
        
        Returns:
            list: List of embeddings.
        """
        return self.model.encode(texts, convert_to_tensor=False)

if __name__ == "__main__":
    embedder = Embedder()
    sample_texts = [
        "This is a test sentence.",
        "Here is another sentence for embedding generation."
    ]
    embeddings = embedder.encode(sample_texts)
    print("Embeddings:")
    for i, emb in enumerate(embeddings):
        print(f"Sample {i+1}: {emb[:10]}...")  # Print first 10 elements as a preview