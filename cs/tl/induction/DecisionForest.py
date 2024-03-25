from typing import Any, Tuple
import numpy as np
import multiprocessing as mp
from .DecisionTree import DecisionTree


class DecisionForest:
    def __init__(self, min_samples_split: int, max_depth: int) -> None:
        """Create a DecisionForest, the parameters are used in the decision tree building
        Args:
            min_samples_split (int, optional): Minumum number of samples per split. Defaults to 10.
            max_depth (int, optional): maximum depth the Decision Trees can reach, None mean no cap. Defaults to None.
        """
        self.max_depth = max_depth  # maximum depth the Decision Trees can reach
        self.min_samples_split = min_samples_split  # minimuym number of samples to split
        self.dts: list[DecisionTree] = []  # list to hold the decision trees

    def fit(self, X_data: np.ndarray, y_data: np.ndarray):
        """Fit method for the decision forest, to create decision trees
        Args:
            X_data (np.ndarray), y_data (np.ndarray)
        """
        assert X_data.shape == y_data.shape  # verifies that the data has the same number of instances and features
        num_processes = mp.cpu_count()  # number of processes the cpu is capable of running simultaneously
        pool = mp.Pool(num_processes)
        self.dts = pool.map(
            self._create_dt,
            [(i, X_data, column, self.min_samples_split, self.max_depth) for i, column in enumerate(y_data.T)],
        )  # splits the data into columns, each column generating a process to create a decision tree
        pool.close()  # indicates that no more tasks will be added to the pool
        pool.join()  # waits for all the processes in the pool to finish their execution

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Forward method for the decision forest
        Args:
            X (np.ndarray): feature array
        Returns:
            np.ndarray: array of predicted labels
        """
        num_processes = mp.cpu_count()
        pool = mp.Pool(num_processes)
        preds = pool.map(self._predict_dt, [(dt, X) for dt in self.dts])  # predicts for each of the decision tree
        pool.close()  # waits for the process to finish
        pool.join()
        return np.array(preds).T  # returns the predictions array, in a list wise shape

    @staticmethod
    def _create_dt(args: Tuple[int, np.ndarray, np.ndarray, int, int]) -> DecisionTree:
        """Auxiliary function to create a new decision tree
        Args:
            args (Tuple[np.ndarray, np.ndarray, int, int]): Collection of arguments
        Returns:
            DecisionTree: generated decision tree
        """
        i, X, column, min_samples_split, max_depth = args  # retrieves the arguments
        print(f"Building DecisionTree on class {i}")
        # Creates a DecisionTree instance from the arguments
        dt = DecisionTree(min_samples_split=min_samples_split, max_depth=max_depth)
        dt.fit(X, column)  # Fit the decision tree
        return dt  # Returns the constructed decision tree

    @staticmethod
    def _predict_dt(args: Tuple[DecisionTree, np.ndarray]) -> np.ndarray:
        """Auxiliary function to predict from a decision tree
        Args:
            args (Tuple[DecisionTree, np.ndarray]): Collection of arguments
        Returns:
            np.ndarray: predicted from the decision tree
        """
        dt, X = args  # gets the decision tree argument and the feature array
        return dt(X)  # call predict, same as forward methods for neural networks
