"""
This module implements the two matrices similarity calculation.
Here we assume input shape is ``(batch_size, seq_len1, dim1)` and ``(batch_size, seq_len2, dim2)`, and
we will return output whose shape is ``(batch_size, seq_len1, seq_len2)``.
"""

import torch
import torch.nn as nn
import math


class CosineSimilarity(nn.Module):
    """
    This similarity function simply computes the cosine similarity between two matrixes.
    It has no parameters.
    """

    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return torch.bmm(normalized_tensor_1, normalized_tensor_2.transpose(1, 2))


class DotProductSimilarity(nn.Module):
    """
    This similarity function simply computes the dot product between two matrixes, with an
    optional scaling to reduce the variance of the output elements.
    """

    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        result = torch.bmm(tensor_1, tensor_2.transpose(1, 2))
        if self.scale_output:
            # TODO why allennlp do multiplication here ?
            result /= math.sqrt(tensor_1.size(-1))
        return result


class ProjectedDotProductSimilarity(nn.Module):
    """
    This similarity function does a projection and then computes the dot product between two matrices.
    It's computed as ``x^T W_1 (y^T W_2)^T + b(Optional)``. An activation function applied after the calculation.
    Default is no activation.
    """

    def __init__(self, tensor_1_dim, tensor_2_dim, projected_dim,
                 reuse_weight=False, bias=False, activation=None):
        super(ProjectedDotProductSimilarity, self).__init__()
        self.reuse_weight = reuse_weight
        self.projecting_weight_1 = nn.Parameter(torch.Tensor(tensor_1_dim, projected_dim))
        if self.reuse_weight:
            if tensor_1_dim != tensor_2_dim:
                raise ValueError('if reuse_weight=True, tensor_1_dim must equal tensor_2_dim')
        else:
            self.projecting_weight_2 = nn.Parameter(torch.Tensor(tensor_2_dim, projected_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projecting_weight_1)
        if not self.reuse_weight:
            nn.init.xavier_uniform_(self.projecting_weight_2)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.projecting_weight_1)
        if self.reuse_weight:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_1)
        else:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_2)
        result = torch.bmm(projected_tensor_1, projected_tensor_2.transpose(1, 2))
        if self.bias is not None:
            result += self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class BiLinearSimilarity(nn.Module):
    """
    This similarity function performs a bilinear transformation of the two input matrices. It's
    computed as ``x^T W y + b(Optional)``. An activation function applied after the calculation.
    Default is no activation.
    """

    def __init__(self, tensor_1_dim, tensor_2_dim, bias=False, activation=None):
        super(BiLinearSimilarity, self).__init__()
        self.weight_matrix = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_2_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        intermediate = torch.matmul(tensor_1, self.weight_matrix)
        result = torch.bmm(intermediate, tensor_2.transpose(1, 2))
        if self.bias is not None:
            result += self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class TriLinearSimilarity(nn.Module):
    """
    This similarity function performs a trilinear transformation of the two input matrices. It's
    computed as ``w^T [x; y; x*y] + b(Optional)``. An activation function applied after the calculation.
    Default is no activation.
    """

    def __init__(self, input_dim, bias=False, activation=None):
        super(TriLinearSimilarity, self).__init__()
        self.input_dim = input_dim
        self.weight_vector = nn.Parameter(torch.Tensor(3 * input_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self.weight_vector.size(0) + 1))
        self.weight_vector.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        w1, w2, w12 = self.weight_vector.chunk(3, dim=-1)
        tensor_1_score = torch.matmul(tensor_1, w1)  # B*L1
        tensor_2_score = torch.matmul(tensor_2, w2)  # B*L2
        combined_score = torch.bmm(tensor_1 * w12, tensor_2.transpose(1, 2))  # B*L1*L2
        result = combined_score + tensor_1_score.unsqueeze(2) + tensor_2_score.unsqueeze(1)
        if self.bias is not None:
            result += self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class MLPSimilarity(nn.Module):
    """
    This similarity function performs Multi-Layer Perception to compute similarity. It's
    computed as ``w^T f(linear(x) + linear(y)) + b(Optional)``. Notify we will use the
    activation(Default tanh) between two perception layers rather than output layer.
    """
    def __init__(self, tensor_1_dim, tensor_2_dim, hidden_dim, bias=False, activation=torch.tanh):
        super(MLPSimilarity, self).__init__()
        self.projecting_layers = nn.ModuleList([nn.Linear(tensor_1_dim, hidden_dim),
                                                nn.Linear(tensor_2_dim, hidden_dim)])
        self.score_weight = nn.Parameter(torch.Tensor(hidden_dim))
        self.score_bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self.score_weight.size(0) + 1))
        self.score_weight.data.uniform_(-std, std)
        if self.score_bias is not None:
            self.score_bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = self.projecting_layers[0](tensor_1)  # B*L1*H
        projected_tensor_2 = self.projecting_layers[1](tensor_2)  # B*L2*H
        combined_tensor = projected_tensor_1.unsqueeze(2) + projected_tensor_2.unsqueeze(1)  # B*L1*L2*H
        result = torch.matmul(self.activation(combined_tensor), self.score_weight)  # B*L1*L2
        if self.score_bias is not None:
            result += self.score_bias
        return result


# class SymmetricProject(nn.Module):
#     def __init__(self, tensor_1_dim, tensor_2_dim, hidden_dim, reuse_weight=True, activation=F.relu):
#         super(SymmetricProject, self).__init__()
#         self.reuse_weight = reuse_weight
#         with tf.variable_scope(self.name):
#             diagonal = tf.get_variable('diagonal_matrix', shape=[self.hidden_dim],initializer=tf.ones_initializer, dtype=tf.float32)
#         self.diagonal_matrix = tf.diag(diagonal)
#         self.projecting_layer = tf.keras.layers.Dense(hidden_dim, activation=activation,
#                                                       use_bias=False)
#         if not reuse_weight:
#             self.projecting_layer2 = tf.keras.layers.Dense(hidden_dim, activation=activation, use_bias=False)
#
#     def __call__(self, t0, t1):
#         trans_t0 = self.projecting_layer(t0)
#         trans_t1 = self.projecting_layer(t1)
#         return tf.matmul(tf.tensordot(trans_t0,self.diagonal_matrix,[[2],[0]]),trans_t1,transpose_b=True)
