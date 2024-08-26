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


class TSFullyConnectedNDVIModel(nn.Module):
    def __init__(self, sequence_length: int, num_years: int, num_seasons: int, year_embed_size: int, season_embed_size: int, hidden_sizes: list):
        super(TSFullyConnectedNDVIModel, self).__init__()
        self.sequence_length = sequence_length

        # Embedding layers for years and seasons
        self.year_embedder = nn.Embedding(num_years, year_embed_size)
        self.season_embedder = nn.Embedding(num_seasons, season_embed_size)

        # Calculate input size for the first dense layer
        input_size = sequence_length * (1 + year_embed_size + season_embed_size) + year_embed_size + season_embed_size

        # Create fully connected layers
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        # Add final output layer
        layers.append(nn.Linear(input_size, 1))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor, target_year: torch.Tensor, target_season: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = ndvi.shape

        # Embed years and seasons for the entire sequence
        years_embedded = self.year_embedder(years)
        seasons_embedded = self.season_embedder(seasons)

        # Combine NDVI with embedded years and seasons
        sequence_features = torch.cat([ndvi.unsqueeze(-1), years_embedded, seasons_embedded], dim=-1)
        sequence_features = sequence_features.view(batch_size, -1)

        # Embed target year and season
        target_year_embedded = self.year_embedder(target_year)
        target_season_embedded = self.season_embedder(target_season)

        # Concatenate all inputs
        x = torch.cat([sequence_features, target_year_embedded, target_season_embedded], dim=1)

        # Pass through fully connected layers
        output = self.fc_layers(x)

        return output
