import numpy as np
from typing import Union
from utils import ignore_warning
from .Nodes import DecisionNode, LeafNode, SplitNode


def entropy(labels: np.ndarray):
    """Calculate the entropy for a given subset of labels
    Args:
        labels (np.ndarray): subset of labels
    Returns:
        np.ndarray: entropy of the given labels
    """
    _, counts = np.unique(labels, return_counts=True)  # calculate total count of each class
    norm_counts = counts / len(labels)  # calculate the probability of each class
    return -np.sum(norm_counts * np.log2(norm_counts))  # calculate the probability of each class


def info_gain(labels: np.ndarray, left_labels: np.ndarray, right_labels: np.ndarray) -> float:
    """Info gain for splitting the labels into two branches
    Args:
        labels (np.ndarray): parent label array
        left_labels (np.ndarray): left generated branch from thresholding a feature
        right_labels (np.ndarray): right generated branch from thresholding a feature
    Returns:
        float: information gain for the split
    """
    return entropy(labels) - ((len(left_labels) / len(labels)) * entropy(left_labels)) - ((len(right_labels) / len(labels)) * entropy(right_labels))


def split_info(labels: np.ndarray, left_labels: np.ndarray, right_labels: np.ndarray) -> float:
    """Split infotmation for splitting the labels into two branches
    Args:
        labels (np.ndarray): parent label array
        left_labels (np.ndarray): left generated branch from thresholding a feature
        right_labels (np.ndarray): right generated branch from thresholding a feature
    Returns:
        float: split information
    """
    epsilon = 1e-8
    return -((len(left_labels) / len(labels)) * np.log2(len(left_labels) / len(labels) + epsilon)) + ((len(right_labels) / len(labels)) * np.log2(len(right_labels) / len(labels) + epsilon))


def info_gain_ratio(labels: np.ndarray, left_labels: np.ndarray, right_labels: np.ndarray) -> float:
    """Information Gain ratio for splitting the labels into two branches
    Args:
        labels (np.ndarray): parent label array
        left_labels (np.ndarray): left generated branch from thresholding a feature
        right_labels (np.ndarray): right generated branch from thresholding a feature
    Returns:
        float: information gain ratio
    """
    return abs(np.nan_to_num(info_gain(labels, left_labels, right_labels) / split_info(labels, left_labels, right_labels)))


class DecisionTree:
    METRIC = {"info_gain": info_gain, "info_gain_ratio": info_gain_ratio}

    def __init__(self, min_samples_split: int, max_depth: int) -> None:
        """Initializes a decision tree with the specified parameters.
        Args:
            min_samples_split (int): The minimum number of samples required to split a node.
            max_depth (int): The maximum depth of the decision tree.
        """
        self.root: Union[LeafNode, DecisionNode] = None  # root node of the tree, initialy a None since no tree was built yet
        self.lowest_depth = 0  # lowest depth achieved by the tree during the construction process
        # Stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the decision tree to the given data set
        Args:
            X (np.ndarray): X values of the dataset
            y (np.ndarray): y values of the dataset (labels)
        """
        self.n_features = X.shape[1]  # number of features in the dataset
        self.n_classes = len(np.unique(y))  # number of uniqe classes in the dataset
        self.root = self._build_tree(X, y)  # builds the decision tree recursively using the dataset

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Union[DecisionNode, LeafNode]:
        """Recursively build a decision tree
        Args:
            X (np.ndarray): subset of features
            y (np.ndarray): subset of labels
            depth (int, optional): current depth of the tree. Defaults to 0.
        Returns:
            Union[DecisionNode, LeafNode]: either a DecisionNode or a LeafNode, depending on the outcome
        """
        assert X.shape[0] == y.shape[0]  # verifies that the number of instances in X and y are equal
        self.lowest_depth = max(depth, self.lowest_depth)  # updates the lowest_depth attribute with the maximum value between depth and the current lowest_depth
        num_samples = len(y)  # calculates the number of samples in the dataset
        # return a leaf when:
        # - the max deapth is reached
        # - the current data is too short to be split
        # - every label on the data is the same
        if (self.max_depth and depth >= self.max_depth) or (num_samples <= self.min_samples_split) or (np.all(y == y[0])):
            return LeafNode(np.bincount(y).argmax(), num_samples)
        best_split: SplitNode = self._get_best_split(X, y)
        if best_split["metric_gain"] > 0:  # if enough improvement from splitting the data
            left_subtree = self._build_tree(
                X=X[X[:, best_split["feature_index"]] <= best_split["threshold"]],
                y=y[X[:, best_split["feature_index"]] <= best_split["threshold"]],
                depth=depth + 1,
            )  # build left subtree
            right_subtree = self._build_tree(
                X=X[X[:, best_split["feature_index"]] > best_split["threshold"]],
                y=y[X[:, best_split["feature_index"]] > best_split["threshold"]],
                depth=depth + 1,
            )  # build right subtree
            # If both subtrees come as a leaf node with the same label, make them into one
            if isinstance(left_subtree, LeafNode) and isinstance(right_subtree, LeafNode):
                if left_subtree.label == right_subtree.label:
                    return LeafNode(np.bincount(y).argmax(), num_samples)
            return DecisionNode(best_split, left_subtree, right_subtree, num_samples)
        return LeafNode(np.bincount(y).argmax(), num_samples)  # elsewise return a leaf

    def _get_best_split(self, X: np.ndarray, y: np.ndarray, metric: str = "info_gain") -> SplitNode:
        """Computes the best split for the given labels
        Args:
            X (np.ndarray): feature array
            y (np.ndarray): label array
            metric (str, optional): metric to be used to calculate the best split ("info_gain", "info_gain_ratio"). Defaults to "info_gain".
        Returns:
            SplitNode: object with the data for selecting the best split
        """
        best_split = SplitNode(metric_gain=-float("inf"))
        for feature in range(self.n_features):  # for each feature
            for threshold in np.unique(X[:, feature]):  # for each unique feature value
                left_y = y[X[:, feature] <= threshold]  # left subtree with features bellow or equal to threshold
                right_y = y[X[:, feature] > threshold]  # right subtree with features above threshold
                if len(left_y) == 0 or len(right_y) == 0:  # if one of the branches is empty is invalid
                    continue
                with ignore_warning(RuntimeWarning):  # throws a warning when a branch is pure
                    mg = self.METRIC[metric](y, left_y, right_y)
                if mg > best_split["metric_gain"]:  # if the metric for the split is better, replace the best split
                    best_split["metric_gain"] = mg
                    best_split["threshold"] = threshold
                    best_split["feature_index"] = feature
        return best_split

    def _make_prediction(self, X: np.ndarray, tree: Union[DecisionNode, LeafNode]) -> int:
        """Recursively iterates over the tree untill a leaf node is found
        Args:
            X (np.ndarray): feature array
            tree (Union[DecisionNode, LeafNode]): current node to be tested
        Returns:
            int: label for the given features
        """
        if isinstance(tree, LeafNode):
            return tree.label  # Return the label when at a leaf node
        
        # Ensure tree.split_data is properly defined and accessed
        if 'feature_index' in tree.split_data and 'threshold' in tree.split_data:
            feature_index = tree.split_data["feature_index"]
            threshold = tree.split_data["threshold"]
        else:
            raise ValueError("Tree node split_data is not properly defined.")

        # Recursive descent based on the split condition
        if X[int(feature_index)] <= threshold:
            return self._make_prediction(X, tree.left)
        else:
            return self._make_prediction(X, tree.right)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.array(self._make_prediction(X, self.root))  # returns the predicted values
