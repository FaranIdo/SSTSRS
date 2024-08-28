import torch
import torch.nn as nn


class FullyConnectedNDVIModel(nn.Module):

    def __init__(self, sequence_length: int, hidden_size: int, num_layers: int):
        super(FullyConnectedNDVIModel, self).__init__()
        self.sequence_length = sequence_length

        layers = []
        layers.append(nn.Linear(sequence_length, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 1))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        # Flatten the input features
        ndvi_flat = ndvi.view(-1, self.sequence_length)

        # Pass through fully connected layers
        output = self.fc_layers(ndvi_flat).squeeze(-1)

        return output


class TSFCEmbeddingNDVIModel(nn.Module):

    def __init__(self, sequence_length: int, ndvi_hidden_size: int, embed_size: int, hidden_sizes: list):
        """
        Initialize the TSFullyConnectedNDVIModel - predict multi-year NDVI.

        Args:
            sequence_length (int): Length of the input sequence - how many years of NDVI data we have.
            ndvi_hidden_size (int): Size of the hidden layer for NDVI processing.
            embed_size (int): Size of the embedding for both years and seasons.
            hidden_sizes (list): List of hidden sizes for fully connected layers.
        """
        super(TSFCEmbeddingNDVIModel, self).__init__()

        self.sequence_length = sequence_length

        # Embedding layers for years and seasons
        self.year_embedder = nn.Embedding(3000, embed_size)
        self.season_embedder = nn.Embedding(2, embed_size)

        # Linear layer for processing the NDVI features
        self.ndvi_processor = nn.Linear(1, ndvi_hidden_size)

        # Calculate input size for the first dense layer
        input_size = (ndvi_hidden_size + embed_size * 2) * sequence_length

        # Create fully connected layers
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        # Add final output layer to make prediction per value in batch
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        # Process NDVI features
        ndvi_processed = self.ndvi_processor(ndvi.unsqueeze(-1)).squeeze(-1)

        # Embed years and seasons for the entire sequence
        years_embedded = self.year_embedder(years)
        seasons_embedded = self.season_embedder(seasons)

        # Convert embedded tensors to float to concatenate with ndvi which is float
        years_embedded = years_embedded.float()
        seasons_embedded = seasons_embedded.float()

        # Combine NDVI with embedded years and seasons
        sequence_features = torch.cat([ndvi_processed, years_embedded, seasons_embedded], dim=-1)

        # Flatten the sequence features
        sequence_features = sequence_features.view(sequence_features.size(0), -1)

        # Pass through fully connected layers
        output = self.fc_layers(sequence_features).squeeze(-1)
        return output
