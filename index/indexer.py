import faiss
import numpy as np

class VectorIndexer:
    def __init__(self, embedding_dim: int):
        """
        Initializes the vector indexer with the specified embedding dimension.
        
        Args:
            embedding_dim (int): Dimension of the semantic embeddings.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # Using L2 distance for similarity search
        self.metadata = []  # Stores metadata (e.g., the text chunks) corresponding to each embedding

    def add(self, embedding: list, meta: str):
        """
        Adds a single embedding and its metadata to the index.
        
        Args:
            embedding (list): The embedding vector.
            meta (str): The corresponding metadata (e.g., text chunk).
        """
        embedding_np = np.array(embedding, dtype='float32').reshape(1, -1)
        self.index.add(embedding_np)
        self.metadata.append(meta)

    def add_bulk(self, embeddings: list, metadatas: list):
        """
        Adds multiple embeddings and their metadata to the index.
        
        Args:
            embeddings (list): List of embedding vectors.
            metadatas (list): List of corresponding metadata strings.
        """
        embeddings_np = np.array(embeddings, dtype='float32')
        self.index.add(embeddings_np)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: list, k: int = 5):
        """
        Searches for the top-k most similar embeddings in the index.
        
        Args:
            query_embedding (list): The query embedding vector.
            k (int): The number of nearest neighbors to retrieve.
        
        Returns:
            tuple: A tuple containing the distances and the corresponding metadata results.
        """
        query_np = np.array(query_embedding, dtype='float32').reshape(1, -1)
        distances, indices = self.index.search(query_np, k)
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return distances, results

if __name__ == "__main__":
    # Example testing of the VectorIndexer with dummy embeddings
    indexer = VectorIndexer(embedding_dim=384)
    dummy_embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    dummy_texts = ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]
    indexer.add_bulk(dummy_embeddings, dummy_texts)
    
    query = [0.15] * 384
    distances, results = indexer.search(query, k=2)
    print("Distances:", distances)
    print("Results:", results)