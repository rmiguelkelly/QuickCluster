
import numpy as np
import quickcluster._C as _binding
from random import randint

class KMeans:

    def __init__(self, k: int, max_iterations: int = 150, random_state: int | None = None) -> None:

        # Ensure that the random state is never None
        r_state = randint(0, 10000) if random_state == None else random_state

        self._handle = _binding.kmeans_init(k, max_iterations, r_state)
        print(self._handle)
    

    def fit(self, X: np.array):
        _binding.kmeans_fit(self._handle, np.array(X, dtype=np.float32))
        return None


    def predict(self, X: np.array):
        _binding.kmeans_predict()
        pass

    
    def centroid_centers(self) -> np.array: 
        return _binding.kmeans_centroids(self._handle)
