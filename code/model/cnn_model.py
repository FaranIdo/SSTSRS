import torch
import torch.nn as nn


class CNNNDVIModel(nn.Module):

    def __init__(self, sequence_length: int, num_channels: int, kernel_size: int, num_layers: int):
        super(CNNNDVIModel, self).__init__()
        self.sequence_length = sequence_length

        layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=kernel_size, padding=kernel_size // 2))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())

        self.cnn_layers = nn.Sequential(*layers)

        # Add a final layer to reduce to a single channel
        self.final_conv = nn.Conv1d(in_channels=num_channels, out_channels=1, kernel_size=1)

        # Add a linear layer for the final prediction
        self.linear = nn.Linear(sequence_length, 1)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        # Add a channel dimension to the input
        ndvi = ndvi.unsqueeze(1)

        # Pass through CNN layers
        output = self.cnn_layers(ndvi)

        # Reduce to a single channel
        output = self.final_conv(output)

        # Remove the channel dimension
        output = output.squeeze(1)

        # Predict the next value
        prediction = self.linear(output)

        # Remove the extra dimension to have a 1D output of NDVI values
        return prediction.squeeze(1)
