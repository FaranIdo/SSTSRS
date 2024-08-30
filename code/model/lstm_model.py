import torch
import torch.nn as nn


class LSTMNDVIModel(nn.Module):

    def __init__(self, sequence_length: int, hidden_size: int = 64, num_layers: int = 1):
        super(LSTMNDVIModel, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        # Reshape ndvi to (batch_size, sequence_length, 1)
        ndvi = ndvi.view(-1, self.sequence_length, 1)

        # Initialize hidden state and cell state
        batch_size = ndvi.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(ndvi.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(ndvi.device)

        # Forward propagate LSTM
        out, _ = self.lstm(ndvi, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out.squeeze()
