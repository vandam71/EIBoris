from __future__ import annotations
import torch
import torch.nn as nn
from ss import StorageSystem, MemoryNode


class ResBlock(nn.Module):
    # For a ResNet architecture, the expansion (relation between the number of input
    # channels in a block vs the number of output channels of that block is '4')
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, i_downsample: ResBlock = None, stride: int = 1) -> None:
        """ResNet block definition
        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            i_downsample (_type_, optional): donwsampling block. Defaults to None.
            stride (int, optional): stride size. Defaults to 1.
        """
        super(ResBlock, self).__init__()
        # Define the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=(out_channels * self.expansion), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=(out_channels * self.expansion)),
        )
        self.i_downsample = i_downsample  # Store the value of i_downsample for downsampling
        self.relu = nn.ReLU(inplace=True)  # ReLU activation for the residual connection

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward method for the ResNet block
        Args:
            X (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        X_skip = X.clone()  # Create a copy of the input tensor for the skip connection
        X = self.conv(X)  # Pass the input tensor through the convolutional
        # Apply downsampling to the skip connection if i_downsample is not None
        if self.i_downsample is not None:
            X_skip = self.i_downsample(X_skip)
        X = X + X_skip  # Add the convolved tensor and the skip connection tensor
        return self.relu(X)  # Apply ReLU activation to the combined tensor and return the result


class ResNet(nn.Module):
    ResNet50_LAYERS = [3, 4, 6, 3]
    ResNet101_LAYERS = [3, 4, 23, 3]
    ResNet152_LAYERS = [3, 8, 36, 3]

    def __init__(self, res_layers: list, num_classes: int, num_channels: int = 3) -> None:
        """ResNet initialization
        Args:
            res_layers (list): list with the number of layers for each filter size in the network
            num_classes (int): number of output classes
            num_channels (int, optional): channels of the image (3 if 'RGB', 1 if 'Grayscale'). Defaults to 1.
        """
        super(ResNet, self).__init__()
        self._num_channels = num_channels
        self._num_classes = num_classes
        self.in_channels: int = 64  # Initial number of channels
        # Input layer: Convolution, BatchNorm, ReLU, and MaxPool
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # List of residual layers
        self.network = nn.ModuleList(
            [
                self._make_layer(res_layers[0], 64),
                self._make_layer(res_layers[1], 128, stride=2),
                self._make_layer(res_layers[2], 256, stride=2),
                self._make_layer(res_layers[3], 512, stride=2),
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling layer
        self.fc = nn.Linear(in_features=(512 * ResBlock.expansion), out_features=num_classes)  # Fully connected layer for classification

    @property
    def num_channels(self) -> int:
        """Get the number of channels.
        Returns:
            int: The number of channels.
        """
        return self._num_channels

    @num_channels.setter
    def num_channels(self, num_channels: int) -> None:
        """Set the number of channels.
        Args:
            num_channels (int): The number of channels.
        """
        self._num_channels = num_channels  # Set the number of channels
        self.input[0] = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)  # Replace the input layer with the given number of channels

    @property
    def num_classes(self) -> int:
        """Get the number of classes.
        Returns:
            int: The number of classes.
        """
        return self._num_classes

    @num_classes.setter
    def num_classes(self, num_classes: int) -> None:
        """Set the number of classes.
        Args:
            num_classes (int): The number of classes.
        """
        self._num_classes = num_classes  # Set the number of classes
        self.fc = nn.Linear(in_features=(512 * ResBlock.expansion), out_features=num_classes).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))  # Update the fully connected layer with the new number of classes

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.
        Args:
            X (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        X = self.input(X)  # Pass the input through the initial layers
        for i, layer in enumerate(self.network):
            X: torch.Tensor = layer(X)  # Pass the input through each layer in the network
            if self.training:
                continue
            if i == 0:
                _low_level_features = X.clone()  # Store the low-level features at the first layer
            elif i == len(self.network) - 1:
                _high_level_features = X.clone()  # Store the high-level features at the last layer
        if not self.training:
            # Create a MemoryNode with the stored low-level and high-level features and assign it to the most recent node in the StorageSystem
            StorageSystem().most_recent_node = MemoryNode((_low_level_features, _high_level_features))
        X = self.avgpool(X)  # Apply adaptive average pooling
        X = X.reshape(X.shape[0], -1)  # Reshape the tensor to a 2D shape for the fully connected layer
        # Pass the reshaped tensor through the fully connected layer for classification
        return self.fc(X)

    def _make_layer(self, res_blocks: int, filters: int, stride: int = 1) -> nn.Sequential:
        """Makes a feature layer
        Args:
            res_blocks (int): number of total blocks in the layer (1 full block and res_blocks-1 of residual blocks)
            filters (int): number of filters for the blocks in the layer
            stride (int, optional): stride size. Defaults to 1.
        Returns:
            nn.Sequential: specified layer
        """
        ii_downsample: ResBlock = None
        layers = nn.ModuleList()
        # Check if downsampling is needed
        if stride != 1 or self.in_channels != filters * ResBlock.expansion:
            # Create downsampling layer using 1x1 convolution and BatchNorm
            ii_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=(filters * ResBlock.expansion), kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=(filters * ResBlock.expansion)),
            )
        # Create the first residual block and add it to the layers
        layers.append(ResBlock(self.in_channels, out_channels=filters, i_downsample=ii_downsample, stride=stride))
        self.in_channels = filters * ResBlock.expansion  # Update the number of input channels for the remaining blocks
        # Adds the remaining residual blocks to the network
        layers.extend([ResBlock(self.in_channels, filters) for _ in range(res_blocks - 1)])
        return nn.Sequential(*layers)  # Return the sequential module containing all the layers
