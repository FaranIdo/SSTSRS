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
    def __init__(self, sequence_length: int, ndvi_hidden_size: int, year_embed_size: int, season_embed_size: int, num_years: int, num_seasons: int):
        super(TSLinearNDVIModel, self).__init__()
        self.sequence_length = sequence_length

        # Embedding layers for years and seasons
        self.year_embedder = nn.Embedding(num_years, year_embed_size)
        self.season_embedder = nn.Embedding(num_seasons, season_embed_size)

        # Linear layer for processing the NDVI features
        self.ndvi_processor = nn.Linear(1, ndvi_hidden_size)

        # Linear layer for processing the combined NDVI and temporal features
        self.sequence_processor = nn.Linear(ndvi_hidden_size + year_embed_size + season_embed_size, ndvi_hidden_size)

        # Final prediction layer
        self.predictor = nn.Linear(ndvi_hidden_size * sequence_length, 1)

        # Activation function for intermediate layers
        self.relu = nn.ReLU()

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = ndvi.shape

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

        # Make prediction
        output = self.predictor(processed_sequence)

        return output

if __name__ == "__main__":
    model = TSLinearNDVIModel(sequence_length=6, ndvi_hidden_size=5, year_embed_size=5, season_embed_size=5, num_years=41, num_seasons=2)
    print(model)

    # sample call to forward
    ndvi = torch.randn(1024, 6, 1)
    years = torch.randint(0, 41, (1024, 6))  # Assuming years range from 0 to 40
    seasons = torch.randint(0, 2, (1024, 6))  # Assuming 2 seasons (0 and 1)
    output = model(ndvi, years, seasons)
    print(output.shape)
