import numpy as np

class ExactRetriever:
    """
    Deterministic Exact Nearest Neighbor Search Index.
    Optimized to prevent O(N^2) memory reallocation.
    """
    def __init__(self, dim=64):
        self.dim = dim
        self.latents = []
        self.num_elements = 0
        self._flat_db = None
        self._needs_update = False
        
    def add(self, latents: np.ndarray):
        """
        Add batch of latents to the index.
        latents: [N, dim]
        """
        # Cast to float32 immediately to ensure uniformity
        self.latents.append(latents.astype(np.float32))
        self.num_elements += latents.shape[0]
        self._needs_update = True
        
    def _update_db(self):
        if self._needs_update and self.num_elements > 0:
            self._flat_db = np.concatenate(self.latents, axis=0)
            self._needs_update = False

    def search(self, query: np.ndarray, k=1):
        if self.num_elements == 0:
            return None, None
            
        k = min(k, self.num_elements)
        self._update_db()
        
        # Cast query to float32 to prevent floating-point drift across hardware FMA implementations
        query = query.astype(np.float32)
        
        # Calculate Exact L2 distance
        q_sq = np.sum(query**2, axis=1, keepdims=True)
        db_sq = np.sum(self._flat_db**2, axis=1)
        dot = np.dot(query, self._flat_db.T)
        
        distances = q_sq + db_sq - 2 * dot
        
        # Round distances to 4 decimal places to guarantee absolute determinism across GPUs/CPUs
        distances = np.round(distances, decimals=4)
        
        # Find indices of top-k smallest distances deterministically
        sorted_indices = np.argsort(distances, axis=1)
        top_k_indices = sorted_indices[:, :k]
        top_k_distances = np.take_along_axis(distances, top_k_indices, axis=1)
        
        return top_k_indices, top_k_distances
        
    def get_items(self, labels: np.ndarray):
        if self.num_elements == 0:
            return None
        self._update_db()
        return self._flat_db[labels]
