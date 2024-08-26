import torch
import torch.nn as nn


class FullyConnectedNDVIModel(nn.Module):
    def __init__(self, num_features: int, sequence_length: int, hidden_size: int, num_layers: int):
        super(FullyConnectedNDVIModel, self).__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length

        layers = []
        layers.append(nn.Linear(num_features * sequence_length, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 1))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = ndvi.shape

        # Flatten the input features
        x = ndvi.reshape(batch_size, -1)

        # Pass through fully connected layers
        output = self.fc_layers(x)

        return output
