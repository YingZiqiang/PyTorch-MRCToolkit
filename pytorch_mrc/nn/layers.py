"""
This module implements most useful network layers in Machine Reading Comprehension(MRC) Field.
Such as Highway Network, Pointer Network and so on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attention import DotAttention
from .recurrent import BiGRU
from .dropout import VariationalDropout
from .util import sequence_mask, masked_softmax

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
    y = g * x + (1 - g) * f(A(x)) where `A` is a linear transformation, `f` is an element-wise
    non-linearity, `g` is an element-wise gate computed as `sigmoid(B(x))`.
    """

    def __init__(self,
                 input_dim,
                 num_layers=1,
                 activation=F.relu):
        super(Highway, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        for layer in self.layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


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


class Gate(nn.Module):
    def __init__(self, input_dim, drop_prob=0.0):
        super().__init__()
        self.gate = nn.Sequential(
            VariationalDropout(drop_prob, batch_first=True),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return inputs * self.gate(inputs)


class StaticPairEncoder(nn.Module):
    def __init__(self, input_dim, memory_dim, hidden_dim, drop_prob=0.0, batch_first=True):
        super(StaticPairEncoder, self).__init__()
        self.attention = DotAttention(input_dim, memory_dim, hidden_dim,
                                      drop_prob=drop_prob, batch_first=batch_first)
        self.gate = nn.Sequential(
            Gate(input_dim + memory_dim, drop_prob=drop_prob),
            VariationalDropout(drop_prob, batch_first=batch_first)
        )
        self.encoder = BiGRU(input_dim + memory_dim, hidden_dim, batch_first=batch_first)

    def forward(self, inputs, memory, inputs_len, memory_mask):
        new_inputs = self.gate(self.attention(inputs, memory, memory_mask))
        outputs, _ = self.encoder(new_inputs, inputs_len)
        return outputs


class StaticSelfMatchEncoder(StaticPairEncoder):
    """
    just same with `StaticPairEncoder`
    """
    pass


class PointerNetwork(nn.Module):
    """
    Implements the Pointer Network.
    """

    def __init__(self, context_dim, question_dim, hidden_dim,
                 cell_type='gru', drop_prob=0.0, batch_first=True):
        super(PointerNetwork, self).__init__()
        self.batch_first = batch_first
        self.cell_type = cell_type.lower()

        if self.cell_type == 'gru':
            cell_cls = nn.GRUCell
        elif self.cell_type == 'lstm':
            cell_cls = nn.LSTMCell
        elif self.cell_type == 'rnn':
            cell_cls = nn.RNNCell
        else:
            raise NotImplementedError('cell_type must be one of rnn/gru/lstm')
        self.cell = cell_cls(context_dim, question_dim)

        self.question_linear = nn.Sequential(
            VariationalDropout(drop_prob),
            nn.Linear(2 * question_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.context_linear = nn.Sequential(
            VariationalDropout(drop_prob),
            nn.Linear(question_dim + context_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.random_attn_vector = nn.Parameter(torch.randn(1, 1, question_dim))

    def forward(self, context_repr, question_repr, context_mask, question_mask):
        """
        Use Pointer Network to compute the probabilities of each position
        to be start and end of the answer
        Returns:
            the logits of evary position to be start and end of the answer
        """
        if not self.batch_first:
            context_repr = context_repr.transpose(0, 1)
            question_repr = question_repr.transpose(0, 1)
            context_mask = context_mask.transpose(0, 1)
            question_mask = question_mask.transpose(0, 1)

        state = self._question_pooling(question_repr, question_mask)  # B*QD
        cell_input, ans_start_logits = self._context_attention(context_repr, context_mask, state)  # B*CD, B*CL
        if self.cell_type == 'lstm':
            state, _ = self.cell(cell_input, hx=(state, state))  # B*QD
        else:
            state = self.cell(cell_input, hx=state)  # B*QD
        _, ans_end_logits = self._context_attention(context_repr, context_mask, state)  # _, B*CL

        return ans_start_logits, ans_end_logits

    def _question_pooling(self, question_repr, question_mask):
        """use attention-pooling to question and a random trainable vector"""
        expanded_att_vector = self.random_attn_vector.expand(question_repr.size(0), question_repr.size(1), -1)  # B*QL*QD
        logits = self.question_linear(torch.cat([question_repr, expanded_att_vector], dim=-1)).squeeze(-1)  # B*QL
        score = masked_softmax(logits, question_mask, dim=-1)  # B*QL
        state = torch.sum(score.unsqueeze(-1) * question_repr, dim=1)  # B*QD
        return state

    def _context_attention(self, context_repr, context_mask, state):
        expanded_state = state.unsqueeze(1).expand(-1, context_repr.size(1), -1)  # B*CL*QD
        logits = self.context_linear(torch.cat([context_repr, expanded_state], dim=-1)).squeeze(-1)  # B*CL
        score = masked_softmax(logits, context_mask, dim=-1)  # B*CL
        cell_input = torch.sum(score.unsqueeze(-1) * context_repr, dim=1)  # B*CD
        return cell_input, logits
