import numpy as np

class ExactRetriever:
    """
    Deterministic Exact Nearest Neighbor Search Index.
    Replaces HNSW to guarantee identical results during compression and decompression,
    avoiding the fatal non-determinism trap for Arithmetic Coding.
    """
    def __init__(self, dim=64):
        self.dim = dim
        self.latents = []
        self.num_elements = 0
        
    def add(self, latents: np.ndarray):
        """
        Add batch of latents to the index.
        latents: [N, dim]
        """
        self.latents.append(latents)
        self.num_elements += latents.shape[0]
        
    def search(self, query: np.ndarray, k=1):
        """
        Search for k nearest neighbors using Exact L2 Distance.
        query: [N, dim]
        Returns: labels [N, k], distances [N, k]
        """
        if self.num_elements == 0:
            return None, None
            
        k = min(k, self.num_elements)
        
        # Concatenate all stored latents
        database = np.concatenate(self.latents, axis=0) # [Total_N, dim]
        
        # Calculate Exact L2 distance
        # dist = (q - db)^2 = q^2 + db^2 - 2*q*db
        q_sq = np.sum(query**2, axis=1, keepdims=True)
        db_sq = np.sum(database**2, axis=1)
        dot = np.dot(query, database.T)
        
        distances = q_sq + db_sq - 2 * dot
        
        # Find indices of top-k smallest distances
        # Using np.argsort for determinism
        sorted_indices = np.argsort(distances, axis=1)
        top_k_indices = sorted_indices[:, :k]
        
        top_k_distances = np.take_along_axis(distances, top_k_indices, axis=1)
        
        return top_k_indices, top_k_distances
        
    def get_items(self, labels: np.ndarray):
        """
        Retrieve actual latent vectors by their labels.
        labels: 1D array of indices
        """
        database = np.concatenate(self.latents, axis=0)
        return database[labels]
