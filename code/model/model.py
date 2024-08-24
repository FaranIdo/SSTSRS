import torch.nn as nn
import torch
import math
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm


"""
Reference: https://github.com/codertimo/BERT-pytorch
Author: Junseong Kim
"""


class PositionalEncoding(nn.Module):
    # TODO - fix cause I have years 1980 to 2024
    def __init__(self, d_model, max_len=2025):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len + 1, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # [d_model/2,]

        # keep pe[0,:] to zeros
        pe[1:, 0::2] = torch.sin(position * div_term)  # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(position * div_term)  # broadcasting to [max_len, d_model/2]

        self.register_buffer("pe", pe)

    def forward(self, time):
        # convert time to int
        time = time.int()
        return self.pe[time, :]
        # # stack instead of concatenate in the origianl pytorch implementation
        # output = torch.stack([torch.index_select(self.pe, 0, time[i, :]) for i in range(time.shape[0])], dim=0)
        # return output  # [batch_size, seq_length, embed_dim]


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
        self.position = PositionalEncoding(d_model=embedding_dim, max_len=2500)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embedding_dim

    def forward(self, input_sequence, year_seq):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)
        obs_embed = self.input(input_sequence)  # [batch_size, seq_length, embedding_dim]
        x = obs_embed.repeat(1, 1, 2)  # [batch_size, seq_length, embedding_dim*2]
        for i in range(batch_size):
            x[i, :, self.embed_size :] = self.position(year_seq[i, :])  # [seq_length, embedding_dim]

        return self.dropout(x)


class BERT(nn.Module):
    def __init__(self, num_features, hidden, n_layers, attn_heads, dropout=0.1):
        """
        :param num_features: number of input features
        :param hidden: hidden size of the SITS-Former model
        :param n_layers: numbers of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        feed_forward_hidden = hidden * 4

        # Divide the hidden size by 2 since we are using concat on positional encoding + spectral features, and not sum
        self.embedding = SpectralBERTEmbedding(num_features, int(hidden / 2), dropout)
        encoder_layer = TransformerEncoderLayer(hidden, attn_heads, feed_forward_hidden, dropout)
        encoder_norm = LayerNorm(hidden)

        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)

    def forward(self, x, year_seq):
        x = self.embedding(input_sequence=x, year_seq=year_seq)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)

        return x


class BERTPrediction(nn.Module):
    def __init__(self, bert: BERT, num_features=1, seq_length=5):
        """
        :param bert: the BERT-Former model acting as a feature extractor
        :param num_features: number of features of an input pixel to be predicted
        """

        super().__init__()
        self.bert = bert
        self.pooling = nn.MaxPool1d(seq_length)
        self.linear = nn.Linear(self.bert.hidden, num_features)

    def forward(self, x, year_seq):
        x = self.bert(x, year_seq)
        # Use pooling to reduce the sequence length - since we want to predict a single future value for each sequence
        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        return self.linear(x)
