import time
import torch
from tqdm import tqdm
import torch.nn as nn
from ls import LearningSystem
from .resnet import ResNet
from utils import ignore_warning
from torch.utils.data import DataLoader
from typing import Union


class ClassificationNetwork(nn.Module):
    network_types = {
        "resnet50": lambda num_classes: ResNet(ResNet.ResNet50_LAYERS, num_classes=num_classes),
        "resnet101": lambda num_classes: ResNet(ResNet.ResNet101_LAYERS, num_classes=num_classes),
        "resnet152": lambda num_classes: ResNet(ResNet.ResNet152_LAYERS, num_classes=num_classes),
    }

    def __init__(
        self,
        num_classes: int,
        net_type: str,
        segmenter: nn.Module = None,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        """Classification Network model.
        Args:
            num_classes (int): Number of output classes.
            net_type (str, optional): Network type. Defaults to "resnet50".
            device (str, optional): Device to be used (cuda or cpu). Defaults to cuda:0 if available, otherwise cpu.
            segmenter (nn.Module, optional): Previously trained segmentation model. Defaults to None.
        """
        super(ClassificationNetwork, self).__init__()
        self.device = device
        # Checks if the network string is a valid implemented ones
        if net_type not in ClassificationNetwork.network_types.keys():
            raise RuntimeError("Invalid network type")
        self.model = ClassificationNetwork.network_types[net_type](num_classes)  # Instantiates the chosen network type
        self.loss_fn = nn.CrossEntropyLoss()  # Binary cross-entropy loss function
        # optimizer and scheduler are started in the Learning System
        self.optim: torch.optim.SGD = None  # Placeholder for the optimizer
        self.scheduler: torch.optim.lr_scheduler.StepLR = None  # Placeholder for the learning rate scheduler
        # scaler used when CUDA is available
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None  # GradScaler for mixed precision training
        self.prev_training: torch.Tensor = None  # tensor to hold previous training metrics
        # The classification network makes use of a previously trained segmentation model
        self.segmenter = segmenter  # Segmentation model used for pre-processing
        self.num_classes = num_classes
        self.to(self.device)  # Sends the module to CUDA or keeps it on the CPU

    def fit(self, dataloader: DataLoader, epochs: int = 8, pos_weight: Union[torch.tensor, None] = None):
        """Fit method for the Classification Network Model.
        Args:
            dataloader (DataLoader): Dataloader containing image and mask data for training.
            epochs (int, optional): Number of epochs to train the model. Defaults to 8.
        Returns:
            torch.Tensor: Training metrics for the current training.
        """

        if pos_weight is not None:
            self.loss_fn = self.loss_fn.__class__(weight=pos_weight).to(self.device)

        # Verifies if the model is correctly initialized, as in, if a Learning System is assigned to the Network
        assert self.optim is not None and self.scheduler is not None, "No Learning System assigned to the model"
        results = []  # List to store the results of each epoch
        since = time.time()  # Time at the start of training
        for epoch in range(epochs):
            self.commulative_matrix = torch.zeros((2, 2), dtype=torch.int64, device=self.device)
            print(f"\nEpoch {epoch+1}/{epochs}: Learning rate is {self.scheduler.get_last_lr()[0]}")
            epoch_results = self.epoch_train(e_data=dataloader)  # Trains the network on the current epoch
            with ignore_warning(UserWarning):  # A UserWarning is sometimes triggered because of desynchronization, it is not relavant so this context is used to avoid it
                self.scheduler.step(epoch + 1)  # Adjusts the learning rate based on the current epoch
            # Print training metrics for the current epoch
            # Print epoch metrics
            print(f"Loss: {epoch_results[:, 0].mean():.4f}")
            print(f"Accuracy: {epoch_results[:, 1].mean():.2f}")
            print(f"Precision: {epoch_results[:, 2].mean():.4f}")
            print(f"Recall: {epoch_results[:, 3].mean():.4f}")
            print(f"F1_Score: {epoch_results[:, 4].mean():.4f}")
            print(f"MCC: {epoch_results[:, 5].mean():.4f}")
            print(self.commulative_matrix)

            # Print class accuracies
            for i in range(self.num_classes):
                print(f"Class {i} Accuracy: {epoch_results[:, 6 + i].mean():.4f}")

            results.append(epoch_results)
        time_elapsed = time.time() - since  # Total training time
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        self.prev_training = torch.cat(results, dim=0)  # Concatenates the epoch results into a single tensor
        return self.prev_training  # Returns the training metrics for the current training

    def epoch_train(self, e_data: DataLoader) -> torch.Tensor:
        """Trains in a single epoch of data
        Args:
            e_data (DataLoader): holds the training DataLoader, it contains the training images and masks
        Returns:
            torch.Tensor: returns the epoch_metrics for training
        """
        epoch_metrics = torch.zeros(0, 6 + self.num_classes)  # Initializes an empty tensor to store the metrics

        pbar = tqdm(e_data, unit=" batch")
        for images, labels in pbar:  # Iterates over the batches of training images and labels
            batch_metrics = self.batch_train(b_data_x=images, b_data_y=labels)  # Trains on a single batch of training images and labels
            epoch_metrics = torch.cat([epoch_metrics, batch_metrics], dim=0)  # Concatenates the batch metrics to the epoch metrics tensor
            pbar.set_postfix({
                'Loss': batch_metrics[0][0].item(),
                'Accuracy': batch_metrics[0][1].item(),
                #'Precision': batch_metrics[0][2].item(),
                #'Recall': batch_metrics[0][3].item(),
                #'F1_Score': batch_metrics[0][4].item(),
                #'MCC': batch_metrics[0][5].item()
            })
        
        return epoch_metrics

    def batch_train(self, b_data_x: torch.Tensor, b_data_y: torch.Tensor) -> torch.Tensor:
        b_data_x = b_data_x.to(self.device, non_blocking=True)
        b_data_y = b_data_y.to(self.device, non_blocking=True)
        if self.segmenter:  # forwards the segmentation model if available
            with torch.no_grad():  # avoid the computation of gradients
                b_data_x = self.segmenter(b_data_x)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):  # autocasts to the specific device type
            output: torch.Tensor = self.model(b_data_x)  # Forward pass: computes the model's output
            # print(b_data_y.shape, output.shape)
            # exit()
            loss: torch.Tensor = self.loss_fn(output, b_data_y)  # Computes the loss between the output and target labels
        predictions: torch.Tensor = (output >= 0.5).long()  # Applies a threshold to the output to obtain binary predictions
        # Detaches the loss tensor from the computation graph and calculates metrics
        batch_metrics = [loss.detach(), *self._calculate_metrics(predictions, b_data_y.long())]
        # from pprint import pprint
        # pprint(batch_metrics)
        self.optim.zero_grad(set_to_none=True)  # Clears the gradients of the optimizer
        if self.scaler:
            self.scaler.scale(loss).backward()  # Backward pass: computes the gradients using automatic mixed precision (if enabled)
            self.scaler.step(self.optim)  # Updates the model's parameters using the optimizer (scaled gradients)
            self.scaler.update()  # Updates the scale for automatic mixed precision
        else:
            loss.backward()  # Backward pass: computes the gradients
            self.optim.step()  # Updates the model's parameters using the optimizer (un-scaled gradients)
        return torch.tensor(batch_metrics).unsqueeze(0)  # Returns the batch training metrics

    def _calculate_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
        """Calculates various performance metrics based on the predicted values and the ground truth labels.
        Args:
            preds (torch.Tensor): Predicted values.
            labels (torch.Tensor): Ground truth labels.
        Returns:
            list[torch.Tensor]: List of performance metrics, including class accuracy, precision, recall, F1 score, and Matthews correlation coefficient (MCC).
        """
        _conf_matrix, _class_conf_matrix = LearningSystem.confusion_matrix(preds, labels, self.num_classes)  # Compute confusion matrix

        _conf_matrix: torch.Tensor
        _class_conf_matrix: torch.Tensor

        self.commulative_matrix += _conf_matrix.to(self.device)
        # print(self.commulative_matrix)

        # print(_class_conf_matrix)

        return [
            LearningSystem.class_accuracy(_conf_matrix).mean(),  # Calculate mean class accuracy
            LearningSystem.precision(_conf_matrix).mean(),  # Calculate mean precision
            LearningSystem.recall(_conf_matrix).mean(),  # Calculate mean recall
            LearningSystem.f1_score(_conf_matrix).mean(),  # Calculate mean F1 score
            LearningSystem.mcc(_conf_matrix).mean(),  # Calculate mean Matthews correlation coefficient (MCC),
            *[LearningSystem.class_accuracy(_class_conf_matrix[i]).mean() for i in range(self.num_classes)]  # Class accuracies
        ]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Classification network forward method
        Args:
            X (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        X = X.to(self.device)  # Move input tensor to the specified device
        # Check if a segmenter is available and apply segmentation
        if self.segmenter:
            X = self.segmenter(X)
        return self.model(X)  # Pass the segmented tensor through the model and return the result
