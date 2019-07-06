import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ops import sequence_mask

VERY_NEGATIVE_NUMBER = -1e29


class Embedding(nn.Module):
    # TODO unk token need to train always
    def __init__(self, pretrained_embedding=None, embedding_shape=None, trainable=True, init_scale=0.02, dtype='float'):
        super(Embedding, self).__init__()
        if pretrained_embedding is None and embedding_shape is None:
            raise ValueError("At least one of pretrained_embedding and embedding_shape must be specified!")

        if pretrained_embedding is not None:
            if isinstance(pretrained_embedding, np.ndarray):
                pretrained_embedding = torch.from_numpy(pretrained_embedding)
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        else:
            self.embedding = nn.Embedding(embedding_shape[0], embedding_shape[1])
            nn.init.uniform_(self.embedding.weight, -init_scale, init_scale)

        if dtype == 'float':
            self.embedding = self.embedding.float()
        elif dtype == 'double':
            self.embedding = self.embedding.double()
        else:
            raise NotImplementedError('the dtype must be one of `float` and `double`.')

        self.embedding.weight.requires_grad = trainable

    def forward(self, indices):
        return self.embedding(indices)


class Highway(nn.Module):
    """
    Implements Highway Networks(https://arxiv.org/pdf/1505.00387.pdf)
    y = H(x, WH ) · T(x, WT) + x · (1 − T(x, WT))
    further: add hidden_size and layer_num.
    """

    def __init__(self,
                 input_units,
                 affine_activation=F.relu,
                 trans_gate_activation=torch.sigmoid,
                 drop_prob=0.0):
        super(Highway, self).__init__()

        self.affine_activation = affine_activation
        self.trans_gate_activation = trans_gate_activation

        self.affine_layer = nn.Linear(input_units, input_units)
        self.trans_gate_layer = nn.Linear(input_units, input_units)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        trans = self.dropout(self.affine_activation(self.affine_layer(x)))
        gate = self.trans_gate_activation(self.trans_gate_layer(x))
        return gate * trans + (1. - gate) * x


class Conv1DAndMaxPooling(nn.Module):
    """ Conv1D for 3D or 4D input tensor, the second-to-last dimension is regarded as timestep """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=F.relu):
        super(Conv1DAndMaxPooling, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.activation = activation

    def forward(self, x, seq_word_len=None):
        input_shape = x.size()
        if len(input_shape) == 4:
            batch_size, max_seq_len = input_shape[0], input_shape[1]
            x = x.reshape([-1, input_shape[-2], input_shape[-1]])
            x = self.activation(self.conv_layer(x.transpose(1, 2))).transpose(1, 2)

            if seq_word_len is not None:
                x = x.reshape([batch_size, max_seq_len, x.size(1), x.size(-1)])
                x = self._masked_max_pooling(x, seq_word_len)
            else:
                x = x.max(1).values
                x = x.reshape([batch_size, -1, x.size(-1)])
        elif len(input_shape) == 3:
            x = self.activation(self.conv_layer(x.transpose(1, 2))).transpose(1, 2)
            x = x.max(1).values
        else:
            raise ValueError('input tensor shape/size must be 3D or 4D')

        return x

    def _masked_max_pooling(self, input, seq_word_len=None):
        # TODO can be improved
        rank = len(input.size()) - 2
        if seq_word_len is not None:
            shape = input.size()
            mask = sequence_mask(seq_word_len.reshape([-1]), maxlen=shape[-2])
            mask = mask.reshape([shape[0], shape[1], shape[2], 1])
            input = input * mask + (1 - mask) * VERY_NEGATIVE_NUMBER
        return input.max(dim=rank).values
