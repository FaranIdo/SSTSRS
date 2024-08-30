import torch
import torch.nn as nn


class LSTMNDVIModel(nn.Module):

    def __init__(self, sequence_length: int, hidden_size: int, num_layers: int):
        super(LSTMNDVIModel, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(sequence_length, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, ndvi.size(0), self.hidden_size).to(ndvi.device)
        c0 = torch.zeros(self.num_layers, ndvi.size(0), self.hidden_size).to(ndvi.device)

        # Pass through LSTM layers
        out, _ = self.lstm(ndvi, (h0, c0))

        # Pass through fully connected layer
        out = self.fc(out[:, -1, :]).squeeze(-1)

        return out
