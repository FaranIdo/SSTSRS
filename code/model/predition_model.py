import torch.nn as nn
from .bert import BERT


class BERTPrediction(nn.Module):
    def __init__(self, bert: BERT, num_features=10):
        """
        :param bert: the BERT-Former model acting as a feature extractor
        :param num_features: number of features of an input pixel to be predicted
        """

        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(self.bert.hidden, num_features)

    def forward(self, x, year_seq):
        x = self.bert(x, year_seq)
        return self.linear(x)
