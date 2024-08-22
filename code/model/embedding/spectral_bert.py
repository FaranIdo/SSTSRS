import torch
import torch.nn as nn
from .position import PositionalEncoding


class SpectralBERTEmbedding(nn.Module):
    """
    Spectral BERT Embedding which is consisted with under features
        1. InputEmbedding : project the input to embedding size through a fully connected layer
        2. PositionalEncoding : adding positional information using sin, cos

        sum of both features are output of BERTEmbedding
    """

    def __init__(self, num_features, embedding_dim, dropout=0.1):
        """
        :param feature_num: number of input features
        :param embedding_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.input = nn.Linear(in_features=num_features, out_features=embedding_dim)
        self.position = PositionalEncoding(d_model=embedding_dim, max_len=366)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embedding_dim

    def forward(self, input_sequence, year_seq):
        batch_size = input_sequence.size(0)
        obs_embed = self.input(input_sequence)  # [batch_size, seq_length, embedding_dim]
        x = obs_embed.repeat(1, 1, 2)  # [batch_size, seq_length, embedding_dim*2]
        for i in range(batch_size):
            x[i, :, self.embed_size :] = self.position(year_seq[i, :])  # [seq_length, embedding_dim]

        return self.dropout(x)
