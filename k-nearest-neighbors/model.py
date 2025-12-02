from numpy import ndarray, sort as numpy_sort, argpartition, mean
from numpy.linalg import norm as euclidean_distance
from math import floor


class KNearestNeighbors:
    """
        K-nearest neighbors – a lazy learning
        algorithm using Euclidean distance

        Parameters
        ----------

        n_neighbors: int
            n is the number of data points that can be used to
            make a prediction, taking them as close neighbors.

            if not given, the odd form of the square root of n_sample is taken
    """

    def __init__(self, n_neighbors: int | None = None, /):

        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X: ndarray, y: ndarray, /):

        self.X, self.y = X, y
        if self.n_neighbors is None:
            self.n_neighbors = floor(X.shape[0] ** (1 / 2))
            self.n_neighbors = self.n_neighbors if self.n_neighbors % 2 == 1 else (self.n_neighbors - 1)

    def predict(self, X: ndarray, /) -> int:
        """Return class label"""

        distance = euclidean_distance(self.X - X, axis=1)
        indexes = argpartition(distance, self.n_neighbors)[:self.n_neighbors]
        class_label = mean(self.y[indexes])
        return class_label