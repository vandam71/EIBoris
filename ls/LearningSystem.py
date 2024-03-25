from __future__ import annotations

# This module will have the learning methods and functions to be used by the computation system for learning.
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cs.bl.classification import ClassificationNetwork
    from cs.bl.segmentation import SegmentationNetwork

EPS = 1e-8  # small value to avoid division by zero


# TP = [i, i]                   # true positives have the same true label and predicted label
# FP = [i, :] - TP              # false positives are the ones that are in the same collumn (were predicted but are not true)
# FN = [:, i] - TP              # false negatives are the ones that are in the same row (are from that class but were predicted wrong)
# TN = total - TP - FP - FN     # true negatives are everything that is not in the same row collumn


class LearningSystem(object):
    def __init__(self, classifier: ClassificationNetwork, segmenter: SegmentationNetwork) -> None:
        """Initializes a LearningSystem object with a classifier and segmenter.
        Args:
            classifier (ClassificationNetwork): The classifier network.
            segmenter (SegmentationNetwork): The segmenter network.
        """
        self.classifier_learn = {}  # Dictionary to store classifier learning-related information
        self.segmenter_learn = {}  # Dictionary to store segmenter learning-related information
        # Configures optimizer and scheduler for the classifier
        classifier.optim = self.classifier_learn["optimiser"] = torch.optim.SGD(classifier.model.parameters(), lr=1e-4, momentum=0.9)
        classifier.scheduler = self.classifier_learn["scheduler"] = torch.optim.lr_scheduler.StepLR(self.classifier_learn["optimiser"], step_size=8, gamma=0.1, verbose=True)
        # Configures optimizer and scheduler for the segmenter
        segmenter.optim = self.segmenter_learn["optimiser"] = torch.optim.Adam(segmenter.model.parameters(), lr=1e-4)
        segmenter.scheduler = self.segmenter_learn["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.segmenter_learn["optimiser"], "min", patience=2, verbose=True)

    @staticmethod
    def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the confusion matrix for the given predictions and targets.
        Args:
            preds (torch.Tensor): 1D tensor of predicted labels.
            targets (torch.Tensor): 1D tensor of true labels.
        Returns:
            torch.Tensor: 2D tensor with the confusion matrix.
        """
        # Creates a tensor with the shape (2, 2) for binary classification
        confusion = torch.zeros(2, 2, dtype=int)
        # Iterates over the prediction/target pairs and increments to the coordinates of these pairs
        # When the prediction is the same as the true label, the value is incremented in the diagonal meaning that the prediction is correct
        # The difference between the prediction and the true label means either a false positive or a false negative.
        preds = preds.flatten()
        targets = targets.flatten()
        for p, t in zip(preds, targets):
            confusion[p, t] += 1
        return confusion

    @staticmethod
    def accuracy(conf_matrix: torch.Tensor) -> torch.Tensor:
        """Gives the overall accuracy of the given confusion matrix.
        Args:
            conf_matrix (torch.Tensor): 2D tensor with the confusion matrix.
        Returns:
            torch.Tensor: One element tensor with the overall accuracy.
        """
        return conf_matrix.diagonal().sum() / (conf_matrix.sum() + EPS)

    @staticmethod
    def class_accuracy(conf_matrix: torch.Tensor) -> torch.Tensor:
        """Gives the accuracy of the given confusion matrix for each class.
        Args:
            conf_matrix (torch.Tensor): 2D tensor with the confusion matrix.
        Returns:
            torch.Tensor: 1D tensor with the accuracy for each class.
        """
        TP = torch.diag(conf_matrix)
        FP = conf_matrix.sum(dim=0) - TP
        FN = conf_matrix.sum(dim=1) - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        return (TP + TN) / ((TP + TN + FP + FN) + EPS)

    @staticmethod
    def precision(conf_matrix: torch.Tensor) -> torch.Tensor:
        """Gives the precision for each class of the given confusion matrix.
        Args:
            conf_matrix (torch.Tensor): 2D tensor with the confusion matrix.
        Returns:
            torch.Tensor: 1D tensor with the precision for each class.
        """
        TP = torch.diag(conf_matrix)
        FP = conf_matrix.sum(dim=0) - TP
        return TP / (TP + FP + EPS)

    @staticmethod
    def recall(conf_matrix: torch.Tensor) -> torch.Tensor:
        """Gives the recall for each class of the given confusion matrix.
        Args:
            conf_matrix (torch.Tensor): 2D tensor with the confusion matrix.
        Returns:
            torch.Tensor: 1D tensor with the recall for each class.
        """
        TP = torch.diag(conf_matrix)
        FN = conf_matrix.sum(dim=1) - TP
        return TP / (TP + FN + EPS)

    @staticmethod
    def f1_score(conf_matrix: torch.Tensor) -> torch.Tensor:
        """Gives the F1 Score for each class of the given confusion matrix.
        It uses the precision and recall for each class.
        Args:
            conf_matrix (torch.Tensor): 2D tensor with the confusion matrix.
        Returns:
            torch.Tensor: 1D tensor with the F1 Score for each class.
        """
        precisions = LearningSystem.precision(conf_matrix)
        recalls = LearningSystem.recall(conf_matrix)
        return 2 * ((precisions * recalls) / (precisions + recalls + EPS))

    @staticmethod
    def mcc(conf_matrix: torch.Tensor) -> torch.Tensor:
        """Calculates the Matthews Correlation Coefficient for each class of the given confusion matrix.
        Args:
            conf_matrix (torch.Tensor): 2D tensor with the confusion matrix.
        Returns:
            torch.Tensor: 1D tensor with the MCC for each class.
        """
        TP = torch.diag(conf_matrix)
        FP = conf_matrix.sum(dim=0) - TP
        FN = conf_matrix.sum(dim=1) - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        return ((TP * TN) - (FP * FN)) / (torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + EPS)

    @staticmethod
    def mask_accuracy(preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Calculates the accuracy of the given predictions and masks.
        Args:
            preds (torch.Tensor): 2D tensor with the mask predictions.
            masks (torch.Tensor): 2D tensor with the true masks.
        Returns:
            torch.Tensor: One element tensor with the mask accuracy.
        """
        return (preds == masks).sum() / torch.numel(preds)

    @staticmethod
    def dice_score(preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Calculates the Dice Score of the given predictions and masks.
        Args:
            preds (torch.Tensor): 2D tensor with the mask predictions.
            masks (torch.Tensor): 2D tensor with the true masks.
        Returns:
            torch.Tensor: One element tensor with the Dice Score.
        """
        return (2 * (preds * masks).sum()) / ((preds + masks).sum() + EPS)

    @staticmethod
    def iou_score(preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Calculates the IoU Score of the given predictions and masks.
        Args:
            preds (torch.Tensor): 2D tensor with the mask predictions.
            masks (torch.Tensor): 2D tensor with the true masks.
        Returns:
            torch.Tensor: One element tensor with the IoU Score.
        """
        return (preds * masks).sum() / ((preds + masks).sum() - ((preds * masks).sum()) + EPS)
