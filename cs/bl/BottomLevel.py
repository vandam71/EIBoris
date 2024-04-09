import torch
import warnings
from tqdm import tqdm
from typing import Any, Tuple
from torch.utils.data import DataLoader
from functools import singledispatchmethod
from .segmentation import SegmentationNetwork
from .classification import ClassificationNetwork


class BottomLevel(object):
    def __init__(self, num_classes: int, attention: bool, net_type: str, use_segmentation: bool) -> None:
        """Bottom level interface constructor
        Args:
            num_classes (int): number of classification classes
            attention (bool, optional): if the segmentation will use attention. Defaults to True.
            net_type (str, optional): name of the network. Defaults to "resnet50".
            use_segmentation (bool, optional): if the classifier will use segmentation. Defaults to True.
        """
        # Initialize the classifier network
        self.classifier = ClassificationNetwork(num_classes=num_classes, net_type=net_type)
        self.segmenter = SegmentationNetwork(attention=attention) if use_segmentation else None  # Initialize the segmentation network
        self.classifier.segmenter = self.segmenter

    @property
    def use_segmentation(self) -> bool:
        # Getter property for the 'use_segmentation' attribute.
        return True if self.segmenter is not None else False

    @use_segmentation.setter
    def use_segmentation(self, value: bool, attention: bool = True) -> None:
        # Setter property for the 'use_segmentation' attribute.
        assert isinstance(value, bool)  # Ensure that the value is a boolean.
        if value is True:
            # If use_segmentation is True, set the classifier's segmenter to the current segmenter.
            self.segmenter = SegmentationNetwork(attention=attention)
        else:
            # If use_segmentation is False, set the classifier's segmenter to None.
            self.segmenter = None

    def train(self, enable: bool):
        # Controls the training mode of the segmenter and classifier models.
        if self.segmenter is not None:
            self.segmenter.train(enable)  # Set the training mode of the segmenter.
        self.classifier.train(enable)  # Set the training mode of the classifier.

    def fit(self, segmentation: Tuple[DataLoader, int] = None, classification: Tuple[DataLoader, int] = None):
        """Trains the BottomLevel model. Can be used for both segmentation and classification.
        The training can be done for both segmentation and classification.
        Args:
            segmentation (Tuple[DataLoader, int], optional): segmentation tuple with the DataLoader and the number of epochs for training. Defaults to None.
            classification (Tuple[DataLoader, int], optional): classification tuple with the DataLoader and the number of epochs for training. Defaults to None.
            use_segmentation (bool, optional): enables if the classification network will use segmentation as a pre-processor for it's data. Defaults to True.
        """
        if segmentation and self.segmenter is not None:
            segmentation_data, seg_epochs = segmentation
            assert isinstance(segmentation_data, DataLoader) and isinstance(seg_epochs, int)
        if classification:
            classification_data, class_epochs = classification
            assert isinstance(classification_data, DataLoader) and isinstance(class_epochs, int)
        # Perform segmentation training if specified
        if segmentation and self.segmenter is not None:
            self.segmenter.fit(segmentation_data, epochs=seg_epochs)
        # Check if segmentation is enabled but no previous training is found
        if self.segmenter is None or self.segmenter.prev_training is None:
            warnings.warn("Segmentation is enabled but no previous training was found.")
        # Perform classification training if specified
        if classification:
            self.classifier.fit(classification_data, epochs=class_epochs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # Call the 'forward' method with the provided arguments and keyword arguments.
        return self.forward(*args, **kwds)

    @singledispatchmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Check if segmentation is enabled but no previous training was found
        if self.segmenter is None or self.segmenter.prev_training is None:
            warnings.warn("Segmentation is enabled but no previous training was found.")
        # Check if classifier training is available
        if self.classifier.prev_training is None:
            warnings.warn("No Classifier training is available.")
        X = self.classifier(X)  # Apply classification to the input tensor
        X = torch.softmax(X, dim=1)  # Convert output scores to probabilities using softmax
        return X  # Return the processed tensor

    @forward.register
    def _(self, X: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        # Iterate over batches and compile probabilities
        X_probs, y_labels = zip(*[(self(X_data), y_true) for X_data, y_true in tqdm(X, desc="Compiling probabilities...")])
        # Concatenate the collected probabilities and labels
        X_probs = torch.cat(X_probs, dim=0)
        y_labels = torch.cat(y_labels, dim=0)
        return X_probs, y_labels  # Return the compiled probabilities and labels
