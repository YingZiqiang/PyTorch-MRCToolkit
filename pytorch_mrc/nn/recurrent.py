# coding: utf-8
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    """
    further: choose whether batch first and return n timestep state
    """
    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.0):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(p=drop_prob)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=(0 if num_layers == 1 else drop_prob))

    def forward(self, x, lengths):
        orig_len = x.size(1)

        # pack
        lengths, sort_idx = lengths.sort(dim=0, descending=True)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # apply rnn
        x, _ = self.rnn(x)

        # unpack
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(dim=0)
        x = x[unsort_idx]

        # apply dropout
        x = self.dropout(x)

        return x, None
