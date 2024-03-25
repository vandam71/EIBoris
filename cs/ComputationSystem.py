import torch
import numpy as np
from .tl import TopLevel
from .bl import BottomLevel
from typing import Tuple, List
from torch.utils.data import DataLoader


class ComputationSystem(object):
    def __init__(self, num_classes: int, attention: bool, net_type: str, use_segmentation: bool, min_samples_split: int, max_depth: int) -> None:
        """Initializes the ComputationSystem.
        Args:
            num_classes (int): Number of classes for classification.
            attention (bool): Flag indicating whether attention mechanism is enabled.
            net_type (str): Type of network architecture.
            use_segmentation (bool): Flag indicating whether segmentation is used as a pre-processor for classification.
            min_samples_split (int): Minimum number of samples required to split a node in the decision tree.
            max_depth (int): Maximum depth of the decision tree.
        """
        # Create the BottomLevel and TopLevel components
        self._bl = BottomLevel(num_classes, attention, net_type, use_segmentation)
        self._tl = TopLevel(min_samples_split, max_depth)

    def fit(self, segmentation: Tuple[DataLoader, int] = None, classification: Tuple[DataLoader, int] = None) -> None:
        """Fits the ComputationSystem to the given datasets.
        Args:
            segmentation (Tuple[DataLoader, int], optional): Segmentation dataset and number of epochs for training. Defaults to None.
            classification (Tuple[DataLoader, int], optional): Classification dataset and number of epochs for training. Defaults to None.
        """
        self._bl.train(True)  # sets the bottom level to training mode
        self._bl.fit(segmentation, classification)  # fits the bottom level to the datasets given
        with torch.no_grad():  # no need to compute gradients
            X, y = self._bl(classification[0])  # forwards the bottom-level with the training data
        X = X.cpu().numpy()  # converts the tensors to arrays
        y = y.cpu().numpy().astype(int)
        self._tl.fit(X, y)  # fits the top-level with the training data from the bottom-level
        self._bl.train(False)  # sets the bottom level to inference mode

    def __call__(self, X: torch.Tensor) -> Tuple[List[int], List[int]]:
        """Performs the inference on the given input.
        Args:
            X (torch.Tensor): Input tensor.
        Returns:
            Tuple[List[int], List[int]]: Tuple containing the probabilities and labels.
        """
        with torch.no_grad():  # no need to compute gradients
            probability: torch.Tensor = self._bl(X)  # forwards the bottom-level
        probability = probability.cpu().tolist()[0]  # converts the probability tensor to a list
        labels: np.ndarray = self._tl(np.array(probability))  # forwards the probability in the top-level
        labels = labels.tolist()[0]  # converts the labels numpy array to a list
        return (probability, labels)
