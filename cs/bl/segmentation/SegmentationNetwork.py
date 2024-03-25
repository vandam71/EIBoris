import time
import torch
from tqdm import tqdm
import torch.nn as nn
from ls import LearningSystem
from utils import ignore_warning
from torch.utils.data import DataLoader
from cs.bl.segmentation.unet import UNet, AttentionUNet


class SegmentationNetwork(nn.Module):
    def __init__(
        self,
        attention: bool,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        """Segmentation Network constructor
        Args:
            attention (bool, optional): if attention or not. Defaults to True.
            device (_type_, optional): Defaults to torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """
        super(SegmentationNetwork, self).__init__()
        self.device = device
        # conditional usage of the attention network
        self.model = AttentionUNet() if attention else UNet()
        self.loss_fn = nn.BCEWithLogitsLoss()  # loss function to evaluate the model
        # optimizer and scheduler are started in the Learning System
        self.optim: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None
        # scaler used when CUDA is available
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.prev_training: torch.Tensor = None  # tensor to hold previous training metrics
        self.to(self.device)  # sends the module to CUDA or keeps it in the CPU

    def fit(self, dataloader: DataLoader, epochs: int = 8) -> torch.Tensor:
        """Fit method for the Segmentation Network Model.
        Args:
            dataloader (DataLoader): Dataloader containing image and mask data for training.
            epochs (int, optional): Number of epochs to train the model. Defaults to 8.
        Returns:
            torch.Tensor: Training metrics for the current training.
        """
        # Verifies if the model is correctly initialized, as in, if a Learning System is assigned to the Network
        assert self.optim is not None and self.scheduler is not None, "No Learning System assigned to the model"
        results = []  # List to store the results of each epoch
        since = time.time()  # Time at the start of training
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_results = self.epoch_train(e_data=dataloader)  # Trains the network on the current epoch
            with ignore_warning(UserWarning):  # A UserWarning is sometimes triggered because of desynchronization, it is not relavant so this context is used to avoid it
                self.scheduler.step(sum(epoch_results[:, 0]) / len(dataloader))  # Adjusts the learning rate based on the average loss
            # Print training metrics for the current epoch
            print(f"Loss: {sum(epoch_results[:, 0])/len(dataloader)}; Dice score: {sum(epoch_results[:, 1])/len(dataloader)}; Accuracy: {sum(epoch_results[:, 2])/len(dataloader)*100:.2f}%; IoU: {sum(epoch_results[:, 3])/len(dataloader)}\n")
            results.append(epoch_results)  # Stores the epoch results
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
        epoch_metrics = torch.zeros(0, 4)  # Initializes an empty tensor to store the metrics
        for images, labels in tqdm(e_data, unit=" batch"):  # Iterates over the batches of training images and labels
            batch_metrics = self.batch_train(b_data_x=images, b_data_y=labels)  # Trains on a single batch of training images and labels
            epoch_metrics = torch.cat([epoch_metrics, batch_metrics], dim=0)  # Concatenates the batch metrics to the epoch metrics tensor
        return epoch_metrics

    def batch_train(self, b_data_x: torch.Tensor, b_data_y: torch.Tensor) -> torch.Tensor:
        """Trains on a single batch of data
        Args:
            b_data_x (torch.Tensor): Image data; b_data_y (torch.Tensor): Mask data
        Returns:
            torch.Tensor: Batch training metrics
        """
        # Moves the input image data to the device (GPU if available)
        b_data_x = b_data_x.to(self.device, non_blocking=True)
        b_data_y = b_data_y.to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):  # autocasts to the specific device type
            output: torch.Tensor = self.model(b_data_x)  # Forward pass: computes the model's output
            loss: torch.Tensor = self.loss_fn(output, b_data_y)  # Computes the loss between the output and target mask
        predictions = (torch.sigmoid(output) > 0.5).float()  # Applies a threshold to the output to obtain binary predictions
        # Stacks the metrics into a single tensor and unsqueezes it to have a batch dimension
        # Detaches the loss tensor from the computation graph and calculates metrics
        batch_metrics = torch.stack([loss.detach(), *self._calculate_metrics(predictions, b_data_y)], dim=0).unsqueeze(0).cpu()
        self.optim.zero_grad(set_to_none=True)  # Clears the gradients of the optimizer
        if self.scaler:
            self.scaler.scale(loss).backward()  # Backward pass: computes the gradients using automatic mixed precision (if enabled)
            self.scaler.step(self.optim)  # Updates the model's parameters using the optimizer (scaled gradients)
            self.scaler.update()  # Updates the scale for automatic mixed precision
        else:
            loss.backward()  # Backward pass: computes the gradients
            self.optim.step()  # Updates the model's parameters using the optimizer (un-scaled gradients)
        return batch_metrics  # Returns the batch training metrics

    def _calculate_metrics(self, preds: torch.Tensor, masks: torch.Tensor) -> list[torch.Tensor]:
        """Calculates various metrics based on the predicted masks and ground truth masks.
        Args:
            preds (torch.Tensor): Predicted masks.
            masks (torch.Tensor): Ground truth masks.
        Returns:
            list[torch.Tensor]: List of metrics including dice score, mask accuracy, and intersection over union (IoU) score.
        """
        return [
            LearningSystem.dice_score(preds, masks),  # Calculate the dice score metric
            LearningSystem.mask_accuracy(preds, masks),  # Calculate the mask accuracy metric
            LearningSystem.iou_score(preds, masks),  # Calculate the intersection over union (IoU) score metric
        ]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model
        Args:
            X (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        X = X.to(self.device)  # Moves the input tensor to the device (GPU if available)
        X_skip = X.clone()  # Creates a clone of the input tensor for later multiplication
        X = self.model(X)  # Forward pass: computes the model's output
        X = torch.clamp(X, min=0, max=1).round()  # Applies element-wise clamping and rounding to ensure values are between 0 and 1
        return X * X_skip  # Element-wise multiplication of the output tensor and the cloned input tensor
