import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dropout import VariationalDropout
from .util import sequence_mask, masked_softmax

VERY_NEGATIVE_NUMBER = -1e29


class BiAttention(nn.Module):
    """ Bi-Directonal Attention from BIDAF Paper(https://arxiv.org/abs/1611.01603)"""

    def __init__(self, similarity_function):
        super(BiAttention, self).__init__()
        self.similarity_function = similarity_function

    def forward(self, context_repr, question_repr, context_mask, question_mask):
        """
        Args:
            context_repr: the 3D torch.Tensor, shape is `(batch_size, max_context_len, context_dim)`
            question_repr: the 3D torch.Tensor, shape is `(batch_size, max_question_len, question_dim)`
            context_mask: the 1D or 2D torch.Tensor, if 1D means context_len, we will use `sequence_mask`
                to generate mask, if 2D we just use the mask directly.
            question_mask: the 1D or 2D torch.Tensor. Similar to the `context_mask` usage
        Returns:
            context2query and query2context attention
        """
        sim_mat = self.similarity_function(context_repr, question_repr)

        if context_mask.dim() == 1:
            context_mask = sequence_mask(context_mask, maxlen=context_repr.size(1))
        if question_mask.dim() == 1:
            question_mask = sequence_mask(question_mask, maxlen=question_repr.size(1))
        mask = context_mask.unsqueeze(2) * question_mask.unsqueeze(1)  # B*CL*QL
        sim_mat = sim_mat + (1. - mask) * VERY_NEGATIVE_NUMBER

        # Context-to-query Attention in the paper
        context2query_prob = F.softmax(sim_mat, dim=-1)
        context2query_attention = torch.bmm(context2query_prob, question_repr)

        # Query-to-context Attention in the paper
        query2context_prob = F.softmax(sim_mat.max(-1).values, dim=-1)
        query2context_attention = torch.bmm(query2context_prob.unsqueeze(1), context_repr)
        query2context_attention = query2context_attention.repeat(1, context_repr.size(1), 1)

        return context2query_attention, query2context_attention


class DotAttention(nn.Module):
    """
    We do those in DotAttention:
    1. Use similarity to compute similarity score
    2. Use masked softmax to gain similarity between inputs and valid memory
    """
    def __init__(self, input_dim, memory_dim, hidden_dim, drop_prob=0.0, batch_first=True):
        super(DotAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.input_linear = nn.Sequential(
            VariationalDropout(drop_prob, batch_first=True),
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU()
        )

        self.memory_linear = nn.Sequential(
            VariationalDropout(drop_prob, batch_first=True),
            nn.Linear(memory_dim, hidden_dim, bias=False),
            nn.ReLU()
        )

    def forward(self, inputs, memory, memory_mask):
        if not self.batch_first:
            inputs = inputs.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_mask = memory_mask.transpose(0, 1)

        input_ = self.input_linear(inputs)  # B*L1*H
        memory_ = self.memory_linear(memory)  # B*L2*H

        logits = torch.bmm(input_, memory_.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # B*L1*L2
        memory_mask = memory_mask.unsqueeze(1).expand(-1, inputs.size(1), -1)  # B*L1*L2
        score = masked_softmax(logits, memory_mask, dim=-1)  # B*L1*L2

        context = torch.bmm(score, memory)  # B*L1*D_M
        new_input = torch.cat([context, inputs], dim=-1)  # B*L1*(D_IN+D_M)

        if not self.batch_first:
            return new_input.transpose(0, 1)
        return new_input


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, input_dim, units, attention_on_itself=True):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.input_dim = input_dim
        self.units = units
        self.attention_on_itself = attention_on_itself  # only workable when query==key
        self.dense_layers = nn.ModuleList([nn.Linear(input_dim, units) for _ in range(3)])

    def forward(self, query, key, value, key_mask=None):
        batch_size, max_query_len, max_key_len = query.size(0), query.size(1), key.size(1)
        wq = self.dense_layers[0](query).reshape(
            [batch_size, max_query_len, self.heads, self.units // self.heads]).permute(2, 0, 1, 3)  # Head*B*QL*(U/Head)
        wk = self.dense_layers[1](key).reshape(
            [batch_size, max_key_len, self.heads, self.units // self.heads]).permute(2, 0, 1, 3)  # Head*B*KL*(U/Head)
        wv = self.dense_layers[2](value).reshape(
            [batch_size, max_key_len, self.heads, self.units // self.heads]).permute(2, 0, 1, 3)  # Head*B*KL*(U/Head)

        attention_score = torch.matmul(wq, wk.transpose(2, 3)) / math.sqrt(float(self.units) / self.heads)  # Head*B*QL*KL
        if torch.equal(query, key) and not self.attention_on_itself:
            attention_score += torch.diag(wq.new_zeros(max_key_len) - VERY_NEGATIVE_NUMBER)
        if key_mask is not None:
            if key_mask.dim() == 1:
                key_mask = sequence_mask(key_mask, maxlen=max_key_len)
            attention_score += (1.0 - key_mask.unsqueeze(1).unsqueeze(0)) * VERY_NEGATIVE_NUMBER
        similarity = F.softmax(attention_score, dim=-1)  # Head*B*QL*KL
        return torch.matmul(similarity, wv).permute(1, 2, 0, 3).reshape([batch_size, max_query_len, self.units])  # B*QL*U
