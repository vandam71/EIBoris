import numpy as np
from cs.tl.induction import DecisionForest


class TopLevel(object):
    def __init__(self, min_samples_split: int, max_depth: int) -> None:
        # Initializes a DecisionForest instance with the provided min_samples_split and max_depth
        self.forest = DecisionForest(min_samples_split=min_samples_split, max_depth=max_depth)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Fits the DecisionForest to the given dataset X and labels y
        self.forest.fit(X, y)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        # Calls the DecisionForest on the input data X and returns the predicted labels
        return self.forest(X)
