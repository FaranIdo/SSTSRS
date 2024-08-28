import torch
import torch.nn as nn


class LinearNDVIModel(nn.Module):

    def __init__(self, sequence_length: int):
        super(LinearNDVIModel, self).__init__()
        self.sequence_length = sequence_length
        self.linear = nn.Linear(sequence_length, 1)

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        # Flatten the input features
        ndvi_flat = ndvi.view(-1, self.sequence_length)

        # Pass through linear layer
        output = self.linear(ndvi_flat).squeeze(-1)

        return output


class TSLinearEmbeddingNDVIModel(nn.Module):
    def __init__(self, sequence_length: int, ndvi_hidden_size: int, embed_size: int):
        """
        Initialize the TSLinearNDVIMode - predict multi-year NDVI.

        Args:
            sequence_length (int): Length of the input sequence - how many years of NDVI data we have.
            ndvi_hidden_size (int): Size of the hidden layer for NDVI processing.
            embed_size (int): Size of the embedding for both years and seasons.
        """
        super(TSLinearEmbeddingNDVIModel, self).__init__()

        self.sequence_length = sequence_length

        # Embedding layers for years and seasons
        # years assume up to 2050 (from 0, because I don't "substruct" starting year)
        # seasons assume 2 seasons - 0 (winter) and 1 (summer)
        self.year_embedder = nn.Embedding(3000, embed_size)
        self.season_embedder = nn.Embedding(2, embed_size)

        # Linear layer for processing the NDVI features
        self.ndvi_processor = nn.Linear(1, ndvi_hidden_size)

        # Linear layer for processing the combined NDVI and temporal features
        self.sequence_processor = nn.Linear(ndvi_hidden_size + embed_size * 2, ndvi_hidden_size)

        # Final prediction layer
        self.predictor = nn.Linear(ndvi_hidden_size * sequence_length, 1)

        # Activation function for intermediate layers
        self.relu = nn.ReLU()

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        batch_size, _ = ndvi.shape

        # Process NDVI features
        ndvi_processed = self.relu(self.ndvi_processor(ndvi.unsqueeze(-1))).squeeze(-1)

        # Embed years and seasons for the entire sequence
        years_embedded = self.year_embedder(years)
        seasons_embedded = self.season_embedder(seasons)

        # Convert embedded tensors to float to conact with ndvi which is float
        years_embedded = years_embedded.float()
        seasons_embedded = seasons_embedded.float()

        # Combine NDVI with embedded years and seasons
        sequence_features = torch.cat([ndvi_processed, years_embedded, seasons_embedded], dim=-1)

        # Process the combined sequence
        processed_sequence = self.relu(self.sequence_processor(sequence_features))
        processed_sequence = processed_sequence.view(batch_size, -1)

        # Make prediction
        output = self.predictor(processed_sequence).squeeze(-1)

        return output


if __name__ == "__main__":
    model = TSLinearEmbeddingNDVIModel(sequence_length=6, ndvi_hidden_size=6, embed_size=2)
    print(model)

    # sample call to forward
    ndvi = torch.randn(1024, 6).float()
    years = torch.randint(0, 41, (1024, 6))  # Assuming years range from 0 to 40
    seasons = torch.randint(0, 2, (1024, 6))  # Assuming 2 seasons (0 and 1)
    output = model(ndvi, years, seasons)
    print(output.shape)

    # test also LinearNDVIModel
    linear_m = LinearNDVIModel(sequence_length=6)
    print(linear_m)
    output = linear_m(ndvi, years, seasons)
    print(output.shape)
