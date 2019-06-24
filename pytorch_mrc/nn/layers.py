# coding: utf-8
# from pytorch_mrc.nn.ops import dropout, add_seq_mask
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VERY_NEGATIVE_NUMBER = -1e29


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


# class Conv1DAndMaxPooling(nn.Module):
#     """ Conv1D for 3D or 4D input tensor, the second-to-last dimension is regarded as timestep """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=F.relu):
#         super(Conv1DAndMaxPooling, self).__init__()
#         self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
#
#     def forward(self, x, seq_len=None):
#         input_shape = x.shape.as_list()
#         batch_size = None
#         if len(input_shape) == 4:
#             batch_size = tf.shape(x)[0]
#             seq_length = tf.shape(x)[1]
#             x = tf.reshape(x, (-1, tf.shape(x)[-2], input_shape[-1]))
#             x = self.conv_layer(x)
#             if seq_len is not None:
#                 hidden_units = x.shape.as_list()[-1]
#                 x = tf.reshape(x, (batch_size, seq_length, tf.shape(x)[1], hidden_units))
#                 x = self.max_pooling(x, seq_len)
#             else:
#                 x = tf.reduce_max(x, axis=1)
#                 x = tf.reshape(x, (batch_size, -1, x.shape.as_list()[-1]))
#         elif len(input_shape) == 3:
#             x = self.conv_layer(x)
#             x = tf.reduce_max(x, axis=1)
#         else:
#             raise ValueError()
#
#         return x
#
#     def max_pooling(self, inputs, seq_len=None):
#         rank = len(inputs.shape) - 2
#         if seq_len is not None:
#             shape = tf.shape(inputs)
#             mask = tf.sequence_mask(tf.reshape(seq_len, (-1,)), shape[-2])
#             mask = tf.cast(tf.reshape(mask, (shape[0], shape[1], shape[2], 1)), tf.float32)
#             inputs = inputs * mask + (1 - mask) * VERY_NEGATIVE_NUMBER
#         return tf.reduce_max(inputs, axis=rank)

class Embedding(nn.Module):
    # TODO unk token need to train always
    def __init__(self, pretrained_embedding=None, embedding_shape=None, trainable=True, init_scale=0.02):
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

        self.embedding = self.embedding.float()

        if not trainable:
            # do not update the weight
            self.embedding.weight.requires_grad = False

    def forward(self, indices):
        return self.embedding(indices)
