
import numpy as np
import quickclusterinternal
from random import randint

class KMeans:

    def __init__(self, k: int, max_iterations: int = 150, random_state: int | None = None) -> None:

        # Ensure that the random state is never None
        r_state = randint(0, 10000) if random_state == None else random_state

        self._handle = quickclusterinternal.kmeans_init(k, max_iterations, r_state)
        print(self._handle)
    

    def fit(self, X: np.array):
        quickclusterinternal.kmeans_fit(self._handle, np.array(X, dtype=np.float32))
        return None


    def predict(self, X: np.array):
        quickclusterinternal.kmeans_predict()
        pass

    
    def centroid_centers(self) -> np.array: 
        return quickclusterinternal.kmeans_centroids(self._handle)
