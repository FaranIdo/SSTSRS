"""
Reference: https://github.com/codertimo/BERT-pytorch
Author: Junseong Kim
"""
import torch.nn as nn
import torch
import math

# There is builtin

class PositionalEncoding(nn.Module):
    # TODO - fix cause I have years 1980 to 2024
    def __init__(self, d_model, max_len=2025):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len+1, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)         # [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  # [d_model/2,]

        # keep pe[0,:] to zeros
        pe[1:, 0::2] = torch.sin(position * div_term)   # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(position * div_term)   # broadcasting to [max_len, d_model/2]

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[x]


class SequencePositionalEncoding(nn.Module):
    # 100 - max sequence of years from example 1980 to 2080
    def __init__(self, d_model, max_len=100):
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

    def forward(self, x):
        return self.pe[x]
