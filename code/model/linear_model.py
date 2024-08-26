import torch
import torch.nn as nn


class LinearNDVIModel(nn.Module):
    def __init__(self, num_features: int, sequence_length: int):
        super(LinearNDVIModel, self).__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length

        self.linear = nn.Linear(num_features * sequence_length, 1)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = ndvi.shape

        # Flatten the input features
        x = ndvi.reshape(batch_size, -1)

        # Pass through linear layer
        output = self.linear(x)

        return output
