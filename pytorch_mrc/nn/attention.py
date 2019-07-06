import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_mrc.nn.ops import sequence_mask

VERY_NEGATIVE_NUMBER = -1e29


class BiAttention(nn.Module):
    """ Bi-Directonal Attention from https://arxiv.org/abs/1611.01603 """

    def __init__(self, similarity_function):
        super(BiAttention, self).__init__()
        self.similarity_function = similarity_function

    def forward(self, context_repr, question_repr, context_len, question_len):
        sim_mat = self.similarity_function(context_repr, question_repr)

        # TODO mask operation, it maybe need to be improved better
        # mask shape is [batch_size, max_context_len, max_question_len]
        context_mask = sequence_mask(context_len, maxlen=context_repr.size(1))
        question_mask = sequence_mask(question_len, maxlen=question_repr.size(1))
        mask = context_mask.unsqueeze(2) * question_mask.unsqueeze(1)

        sim_mat = sim_mat + (1. - mask) * VERY_NEGATIVE_NUMBER

        # Context-to-query Attention in the paper
        context2query_prob = F.softmax(sim_mat, dim=-1)
        context2query_attention = torch.bmm(context2query_prob, question_repr)

        # Query-to-context Attention in the paper
        query2context_prob = F.softmax(sim_mat.max(-1).values, dim=-1)
        query2context_attention = torch.bmm(query2context_prob.unsqueeze(1), context_repr)
        query2context_attention = query2context_attention.repeat(1, context_repr.size(1), 1)

        return context2query_attention, query2context_attention
