# coding:utf-8
import torch
import torch.nn as nn
import math


class DotProduct(nn.Module):
    def __init__(self, scale=False):
        super(DotProduct, self).__init__()
        self.scale = scale

    def forward(self, t0, t1):
        dots = torch.bmm(t0, t1.transpose(1, 2))
        if self.scale:
            last_dims = t0.size()[-1]
            dots = dots / math.sqrt(last_dims)
        return dots


class ProjectedDotProduct(nn.Module):
    def __init__(self, t0_units, t1_units, hidden_units, activation=None, reuse_weight=False):
        super(ProjectedDotProduct, self).__init__()
        self.activation = activation
        self.reuse_weight = reuse_weight
        self.projecting_layer = nn.Linear(t0_units, hidden_units, bias=False)
        if reuse_weight:
            if t0_units != t1_units:
                raise Exception('if reuse_weight=True, t0_units must equal t1_units')
        else:
            self.projecting_layer2 = nn.Linear(t1_units, hidden_units, bias=False)

    def forward(self, t0, t1):
        t0 = self.projecting_layer(t0)
        if self.reuse_weight:
            t1 = self.projecting_layer(t1)
        else:
            t1 = self.projecting_layer2(t1)
        if self.activation is not None:
            t0 = self.activation(t0)
            t1 = self.activation(t1)

        return torch.bmm(t0, t1.transpose(1, 2))


class BiLinear(nn.Module):
    def __init__(self, t0_units, t1_units):
        super(BiLinear, self).__init__()
        self.projecting_layer = nn.Linear(t0_units, t1_units, bias=False)

    def forward(self, t0, t1):
        t0 = self.projecting_layer(t0)
        return torch.bmm(t0, t1.transpose(1, 2))


class TriLinear(nn.Module):
    def __init__(self, input_units, bias=False):
        super(TriLinear, self).__init__()
        # TODO why there has some device problems using [xx for _ in range(2)]
        # self.projecting_layers = [nn.Linear(input_units, 1, bias=False) for _ in range(2)]
        self.projecting_layer0 = nn.Linear(input_units, 1, bias=False)
        self.projecting_layer1 = nn.Linear(input_units, 1, bias=False)
        self.dot_w = nn.Parameter(torch.zeros(1, 1, input_units))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.bias = None

        # TODO it's ok?
        for weigth in (self.projecting_layer0.weight, self.projecting_layer1.weight, self.dot_w):
            nn.init.xavier_uniform_(weigth)

    def forward(self, t0, t1):
        t0_score = self.projecting_layer0(t0).squeeze(-1)
        t1_score = self.projecting_layer1(t1).squeeze(-1)

        t0_dot_w = t0 * self.dot_w
        t0_t1_score = torch.bmm(t0_dot_w, t1.transpose(1,  2))
        out = t0_t1_score + t0_score.unsqueeze(2) + t1_score.unsqueeze(1)

        if self.bias is not None:
            out += self.bias
        return out


# class MLP(Layer):
#     def __init__(self, hidden_units, activation=tf.nn.tanh, name="mlp"):
#         super(MLP, self).__init__(name)
#         self.activation = activation
#         self.projecting_layers = [tf.keras.layers.Dense(hidden_units, activation=None) for _ in range(2)]
#         self.score_layer = tf.keras.layers.Dense(1, activation=None, use_bias=False)
#
#     def __call__(self, t0, t1):
#         t0 = self.projecting_layers[0](t0)
#         t1 = self.projecting_layers[1](t1)
#         t0_t1 = tf.expand_dims(t0, axis=2) + tf.expand_dims(t1, axis=1)
#         return tf.squeeze(self.score_layer(self.activation(t0_t1)), axis=-1)
#
#
# class SymmetricProject(Layer):
#     def __init__(self, hidden_units, reuse_weight=True, activation=tf.nn.relu, name='symmetric_nolinear'):
#         super(SymmetricProject, self).__init__(name)
#         self.reuse_weight = reuse_weight
#         self.hidden_units = hidden_units
#         with tf.variable_scope(self.name):
#             diagonal = tf.get_variable('diagonal_matrix', shape=[self.hidden_units],initializer=tf.ones_initializer, dtype=tf.float32)
#         self.diagonal_matrix = tf.diag(diagonal)
#         self.projecting_layer = tf.keras.layers.Dense(hidden_units, activation=activation,
#                                                       use_bias=False)
#         if not reuse_weight:
#             self.projecting_layer2 = tf.keras.layers.Dense(hidden_units, activation=activation, use_bias=False)
#
#     def __call__(self, t0, t1):
#         trans_t0 = self.projecting_layer(t0)
#         trans_t1 = self.projecting_layer(t1)
#         return tf.matmul(tf.tensordot(trans_t0,self.diagonal_matrix,[[2],[0]]),trans_t1,transpose_b=True)
