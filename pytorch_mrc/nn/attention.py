import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_mrc.nn.util import sequence_mask, masked_softmax
from pytorch_mrc.nn.recurrent import GRU

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
        if len(context_mask.size()) == 1:
            context_mask = sequence_mask(context_mask, maxlen=context_repr.size(1))
        if len(question_mask.size()) == 1:
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


class CoAttention(nn.Module):
    """Gated attention RNNs in R-Net Paper. Here we call it Co-attention and we have made some changes."""

    def __init__(self, input_dim, hidden_size):
        self.context_dense = nn.Linear(input_dim, hidden_size)
        self.question_dense = nn.Linear(input_dim, hidden_size)
        self.score_dense = nn.Linear(hidden_size, 1)
        self.gate_dense = nn.Linear(2 * input_dim, 2 * input_dim)
        self.co_att_gru = GRU(2 * input_dim, hidden_size)

    def forward(self, context_repr, question_repr, context_len, question_mask):
        co_att_context = self.context_dense(context_repr).unsqueeze(2)  # B*CL*1*H
        co_att_question = self.question_dense(question_repr).unsqueeze(1)  # B*1*QL*H
        co_att_score = self.score_dense(F.tanh(co_att_context + co_att_question)).squeeze(-1)  # B*CL*QL
        co_att_sim = masked_softmax(co_att_score, question_mask.unsqueeze(1).repeat(1, co_att_score.size(1), 1))  # B*CL*QL
        co_att_rnn_input = torch.cat([context_repr, torch.bmm(co_att_sim, question_repr)], dim=-1)  # B*CL*(H*2*3*2)

        # Another Gate
        gate = torch.sigmoid(self.gate_dense(co_att_rnn_input))
        co_att_rnn_input = co_att_rnn_input * gate

        co_att_output, _ = self.co_att_gru(co_att_rnn_input, context_len)

        return co_att_output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, input_dim, hidden_dim, attention_on_itself=True):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_on_itself = attention_on_itself  # only workable when query==key
        self.dense_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(3)])

    def forward(self, query, key, value, mask=None):
        batch_size, max_query_len, max_key_len = query.size(0), query.size(1), key.size(1)
        wq = self.dense_layers[0](query).reshape(
            [batch_size, max_query_len, self.heads, self.units // self.heads]).permute(2, 0, 1, 3)  # Head*B*QL*(U/Head)
        wk = self.dense_layers[1](key).reshape(
            [batch_size, max_key_len, self.heads, self.units // self.heads]).permute(2, 0, 1, 3)  # Head*B*KL*(U/Head)
        wv = self.dense_layers[2](value).reshape(
            [batch_size, max_key_len, self.heads, self.units // self.heads]).permute(2, 0, 1, 3)  # Head*B*KL*(U/Head)

        attention_score = torch.matmul(wq, wk.transpose(2, 3)) / torch.sqrt(float(self.units) / self.heads)  # Head*B*QL*KL
        if query == key and not self.attention_on_itself:
            attention_score += torch.diag(wq.new_zeros(max_key_len) - VERY_NEGATIVE_NUMBER)
        if mask is not None:
            if len(mask.size()) == 1:
                mask = sequence_mask(mask, maxlen=max_key_len)
            attention_score += (1.0 - mask.unsqueeze(1).unsqueeze(0)) * VERY_NEGATIVE_NUMBER
        similarity = F.softmax(attention_score, dim=-1)  # Head*B*QL*KL
        return torch.matmul(similarity, wv).permute(1, 2, 0, 3).reshape([batch_size, max_query_len, self.units])  # B*QL*U
