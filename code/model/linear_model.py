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


class TSLinearNDVIModel(nn.Module):
    def __init__(self, sequence_length: int, ndvi_hidden_size: int, year_embed_size: int, season_embed_size: int, num_years: int, num_seasons: int, combined_hidden_size: int):
        super(TSLinearNDVIModel, self).__init__()
        self.sequence_length = sequence_length

        # Embedding layers for years and seasons
        self.year_embedder = nn.Embedding(num_years, year_embed_size)
        self.season_embedder = nn.Embedding(num_seasons, season_embed_size)

        # Linear layer for processing the NDVI features
        self.ndvi_processor = nn.Linear(sequence_length, ndvi_hidden_size)

        # Linear layer for processing the combined NDVI and temporal features
        self.sequence_processor = nn.Linear(ndvi_hidden_size + year_embed_size + season_embed_size, ndvi_hidden_size)

        # Combine processed features
        self.combiner = nn.Linear(ndvi_hidden_size * sequence_length + year_embed_size + season_embed_size, combined_hidden_size)

        # Final prediction layer
        self.predictor = nn.Linear(combined_hidden_size, 1)

        # Activation function for intermediate layers
        self.relu = nn.ReLU()

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor, target_year: torch.Tensor, target_season: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = ndvi.shape

        # Flatten the input features
        ndvi = ndvi.reshape(batch_size, -1)

        # Process NDVI features
        ndvi_processed = self.relu(self.ndvi_processor(ndvi))

        # Embed years and seasons for the entire sequence
        years_embedded = self.year_embedder(years)
        seasons_embedded = self.season_embedder(seasons)

        # Combine NDVI with embedded years and seasons
        sequence_features = torch.cat([ndvi_processed, years_embedded, seasons_embedded], dim=-1)

        # Process the combined sequence
        processed_sequence = self.relu(self.sequence_processor(sequence_features))
        processed_sequence = processed_sequence.view(batch_size, -1)

        # Embed target year and season
        target_year_embedded = self.year_embedder(target_year)
        target_season_embedded = self.season_embedder(target_season)

        # Combine all features
        combined_features = torch.cat([processed_sequence, target_year_embedded, target_season_embedded], dim=1)
        combined_features = self.relu(self.combiner(combined_features))

        # Make prediction
        output = self.predictor(combined_features)

        return output
