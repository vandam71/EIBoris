import torch
import torch.nn as nn


class UNet_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Convolution Module, two consecutive convolution with Normalization and ReLU activation
        Args:
            in_channels (int): number of input channels (either last layer output or number of channels in the image (RGB-3, Gray-1))
            out_channels (int): number of output channels
        """
        super(UNet_Block, self).__init__()
        # Define the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.conv(X)


class AttentionGate(nn.Module):
    def __init__(self, features: int, n_coefficients: int) -> None:
        """Attention Gate
        Args:
            features (int): number of features to
            n_coefficients (int): number of transitory coefficients
        """
        super(AttentionGate, self).__init__()
        # Define the W_gate module with a convolutional layer and batch normalization
        self.W_gate = nn.Sequential(nn.Conv2d(in_channels=features, out_channels=n_coefficients, kernel_size=1), nn.BatchNorm2d(n_coefficients))
        # Define the W_x module with a convolutional layer and batch normalization
        self.W_x = nn.Sequential(nn.Conv2d(in_channels=features, out_channels=n_coefficients, kernel_size=1), nn.BatchNorm2d(n_coefficients))
        # Define the psi module with a convolutional layer, batch normalization, and sigmoid activation
        self.psi = nn.Sequential(nn.Conv2d(in_channels=n_coefficients, out_channels=1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)  # ReLU activation

    def forward(self, gate: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """Forward function for the attention gate
        Args:
            gate (torch.Tensor): gate tensor
            skip_connection (torch.Tensor): tensor of the skip connection
        Returns:
            torch.Tensor: weighted attention gate tensor
        """
        g1 = self.W_gate(gate)  # Pass the gate tensor through the W_gate module
        x1 = self.W_x(skip_connection)  # Pass the skip connection tensor through the W_x module
        psi = self.relu(g1 + x1)  # Element-wise addition of g1 and x1 followed by ReLU activation
        psi = self.psi(psi)  # Pass the combined tensor through the psi module for attention weighting
        return skip_connection * psi  # Multiply the skip connection tensor with the attention weights


class UNet(nn.Module):
    # Encoder and decoder blocks are pretty much the same, but the decoder has a
    # Transposed Convolution that receives the skip connection from the previous encoder
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = [64, 128, 256, 512]) -> None:
        """UNet Structure definition
        Args:
            in_channels (_type_): number of input channels (channels of the image)
            out_channels (_type_): number of output channels (usually one image, so 1 output channel)
            features (list, optional): number of features for each layer in the network. Defaults to [64, 128, 256, 512].
        """
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer for downsampling
        # Encoder and Decoder modules
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # Encoder ladder
        for feature in features:
            self.encoder.append(UNet_Block(in_channels=in_channels, out_channels=feature))
            in_channels = feature
        # Decoder ladder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2))
            self.decoder.append(UNet_Block(feature * 2, feature))
        # Connection between encoder and decoder, the bottom of the U in the network
        self.bottleneck = UNet_Block(features[-1], features[-1] * 2)
        # Output Layer
        self.final_layer = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # connections between an encoder block and the assigned decoder block
        skip_connections = []
        # Encoding
        for block in self.encoder:
            X = block(X)
            skip_connections.append(X)
            # Max Pooling after each block
            X = self.pool(X)
        # Final encoding layer, with no skip connection assotiated
        X = self.bottleneck(X)
        # Reverse the skip connections
        skip_connections = skip_connections[::-1]
        # Decoding (maybe this part could be less messy, but that way the encoder/decoder variables
        # couldnt be nn.ModuleList() types)
        for idx in range(0, len(self.decoder), 2):
            X = self.decoder[idx](X)
            skip_connection = skip_connections[idx // 2]
            # Concatenate the skip connection with the output from previous layer
            concat_skip = torch.cat((skip_connection, X), dim=1)
            X = self.decoder[idx + 1](concat_skip)
        return self.final_layer(X)


class AttentionUNet(nn.Module):
    # Encoder and decoder blocks are pretty much the same, but the decoder has a Transposed Convolution that receives the skip connection from the previous encoder
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list[int] = [64, 128, 256, 512]) -> None:
        """UNet Structure definition but with Attention gates
        Args:
            in_channels (_type_): number of input channels (channels of the image)
            out_channels (_type_): number of output channels (usually one image, so 1 output channel)
            features (list, optional): number of features for each layer in the network. Defaults to [64, 128, 256, 512].
        """
        super(AttentionUNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer for downsampling
        self._num_channels = in_channels
        # Encoder and Decoder modules
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # Encoder ladder
        for feature in features:
            self.encoder.append(UNet_Block(in_channels=in_channels, out_channels=feature))
            in_channels = feature
        # Decoder ladder with the attention gate blocks
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2))
            self.decoder.append(AttentionGate(feature, feature // 2))
            self.decoder.append(UNet_Block(feature * 2, feature))
        # Connection between encoder and decoder, the bottom of the U in the network
        self.bottleneck = UNet_Block(features[-1], features[-1] * 2)
        # Output Layer
        self.final_layer = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    @property
    def num_channels(self) -> int:
        """Get the number of channels.
        Returns:
            int: The number of channels.
        """
        return self._num_channels

    @num_channels.setter
    def num_channels(self, num_channels: int, feature: int = 64) -> None:
        """Set the number of channels.
        Args:
            num_channels (int): The number of channels.
            feature (int, optional): The feature size. Defaults to 64.
        """
        self._num_channels = num_channels  # Set the number of channels
        # Update the first encoder block with the new number of channels
        self.encoder[0] = UNet_Block(in_channels=num_channels, feature=feature)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward method for the AttentionUnet
        Args:
            X (torch.Tensor): input tensor
        Returns:
            torch.Tensor: segmented output tensor
        """
        # Connections between an encoder block and the assigned decoder block
        skip_connections = []  # List to store skip connections from the encoder
        # Encoding
        for block in self.encoder:  # Iterate through the encoder blocks
            X = block(X)  # Pass input through the current encoder block
            skip_connections.append(X)  # Store the skip connection
            X = self.pool(X)  # Max Pooling after each block to downsample
        X = self.bottleneck(X)  # Final encoding layer, with no skip connection assotiated
        skip_connections = skip_connections[::-1]  # Reverse the skip connections
        # Decoding
        for idx in range(0, len(self.decoder), 3):  # Iterate through the decoder blocks
            X = self.decoder[idx](X)  # Transposed Convolution (Upsampling)
            skip_connection = skip_connections[idx // 3]  # Retrieve the corresponding skip connection
            X = self.decoder[idx + 1](X, skip_connection)  # Attention module combining skip connection and current output
            concat_skip = torch.cat((X, skip_connection), dim=1)  # Concatenate skip connection and output
            X = self.decoder[idx + 2](concat_skip)  # Double convolution to refine the concatenated features
        return self.final_layer(X)  # Output layer
