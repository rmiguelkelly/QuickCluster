
from quickcluster._device import retrieve_device_handle
import quickcluster._C as _binding

import numpy as np
from random import randint

class KMeans:

    def __init__(self, k: int, max_iterations: int = 150, random_state: int | None = None) -> None:
        """Inititation for the KMeans algorithm"""


        # Ensure that the random state is never None
        random_state = randint(0, 10000) if random_state == None else random_state

        device_capsule = retrieve_device_handle()

        # Create the KMeans handle
        self._handle = _binding.kmeans_init(k, max_iterations, random_state, device_capsule)
    

    def fit(self, X: np.ndarray):
        """Attempts to cluster data expressed as a 2D matrix"""

        _binding.kmeans_fit(self._handle, np.array(X, dtype=np.float32))
        return None


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Labels the data with the most logic cluster id"""

        _binding.kmeans_predict()
        pass

    
    def centroid_centers(self) -> np.ndarray:
        """The centers of the converged clusters"""
        
        return _binding.kmeans_centroids(self._handle)
