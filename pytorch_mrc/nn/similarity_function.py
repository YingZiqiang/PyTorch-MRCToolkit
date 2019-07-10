import torch
import torch.nn as nn
import math


class CosineSimilarity(nn.Module):
    """
    This similarity function simply computes the cosine similarity between each pair of vectors. It has
    no parameters.
    """

    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


class DotProductSimilarity(nn.Module):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.
    """

    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self.scale_output:
            # TODO why allennlp do multiplication at here ?
            result /= math.sqrt(tensor_1.size(-1))
        return result


class ProjectedDotProductSimilarity(nn.Module):
    """
    This similarity function does a projection and then computes the dot product. It's computed
    as ``x^T W_1 (y^T W_2)^T``
    An activation function applied after the calculation. Default is no activation.
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
        result = (projected_tensor_1 * projected_tensor_2).sum(dim=-1)
        if self.bias is not None:
            result = result + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class BiLinearSimilarity(nn.Module):
    """
    This similarity function performs a bilinear transformation of the two input vectors. This
    function has a matrix of weights ``W`` and a bias ``b``, and the similarity between two vectors
    ``x`` and ``y`` is computed as ``x^T W y + b``.
    An activation function applied after the calculation. Default is no activation.
    """

    def __init__(self, tensor_1_dim, tensor_2_dim, activation=None):
        super(BiLinearSimilarity, self).__init__()
        self.weight_matrix = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_2_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        intermediate = torch.matmul(tensor_1, self.weight_matrix)
        result = (intermediate * tensor_2).sum(dim=-1) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class TriLinearSimilarity(nn.Module):
    """
    This similarity function performs a trilinear transformation of the two input vectors. It's
    computed as ``w^T [x; y; x*y] + b``.
    An activation function applied after the calculation. Default is no activation.
    """

    def __init__(self, input_dim, activation=None):
        super(TriLinearSimilarity, self).__init__()
        self.weight_vector = nn.Parameter(torch.Tensor(3 * input_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self.weight_vector.size(0) + 1))
        self.weight_vector.data.uniform_(-std, std)
        self.bias.data.fill_(0)

    def forward(self, tensor_1, tensor_2):
        combined_tensors = torch.cat([tensor_1, tensor_2, tensor_1 * tensor_2], dim=-1)
        result = torch.matmul(combined_tensors, self.weight_vector) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result


class MultiHeadedSimilarity(nn.Module):
    """
    This similarity function uses multiple "heads" to compute similarity.  That is, we take the
    input tensors and project them into a number of new tensors, and compute similarities on each
    of the projected tensors individually.  The result here has one more dimension than a typical
    similarity function.
    """

    def __init__(self,
                 num_heads,
                 tensor_1_dim,
                 tensor_1_projected_dim=None,
                 tensor_2_dim=None,
                 tensor_2_projected_dim=None,
                 internal_similarity=DotProductSimilarity()):
        super(MultiHeadedSimilarity, self).__init__()
        self.num_heads = num_heads
        self.internal_similarity = internal_similarity
        tensor_1_projected_dim = tensor_1_projected_dim or tensor_1_dim
        tensor_2_dim = tensor_2_dim or tensor_1_dim
        tensor_2_projected_dim = tensor_2_projected_dim or tensor_2_dim
        if tensor_1_projected_dim % num_heads != 0:
            raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
                             % (tensor_1_projected_dim, num_heads))
        if tensor_2_projected_dim % num_heads != 0:
            raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
                             % (tensor_2_projected_dim, num_heads))
        self.tensor_1_projection = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_1_projected_dim))
        self.tensor_2_projection = nn.Parameter(torch.Tensor(tensor_2_dim, tensor_2_projected_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.tensor_1_projection)
        torch.nn.init.xavier_uniform_(self.tensor_2_projection)

    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.tensor_1_projection)
        projected_tensor_2 = torch.matmul(tensor_2, self.tensor_2_projection)

        # Here we split the last dimension of the tensors from (..., projected_dim) to
        # (..., num_heads, projected_dim / num_heads), using tensor.view().
        last_dim_size = projected_tensor_1.size(-1) // self.num_heads
        new_shape = list(projected_tensor_1.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_1 = projected_tensor_1.view(*new_shape)
        last_dim_size = projected_tensor_2.size(-1) // self.num_heads
        new_shape = list(projected_tensor_2.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_2 = projected_tensor_2.view(*new_shape)

        # And then we pass this off to our internal similarity function.  Because the similarity
        # functions don't care what dimension their input has, and only look at the last dimension,
        # we don't need to do anything special here.  It will just compute similarity on the
        # projection dimension for each head, returning a tensor of shape (..., num_heads).
        return self.internal_similarity(split_tensor_1, split_tensor_2)


# class MLP(nn.Module):
#     """
#     This similarity function performs Multi-Layer Perception to compute similarity. It's
#     computed as ``w^T f(W_1 x + W_2 y)``
#     """
#     def __init__(self, hidden_units, activation=F.tanh):
#         super(MLP, self).__init__()
#         self.activation = activation
#         self.projecting_layers = [tf.keras.layers.Dense(hidden_units, activation=None) for _ in range(2)]
#         self.score_layer = tf.keras.layers.Dense(1, activation=None, use_bias=False)
#
#     def forward(self, t0, t1):
#         t0 = self.projecting_layers[0](t0)
#         t1 = self.projecting_layers[1](t1)
#         t0_t1 = tf.expand_dims(t0, axis=2) + tf.expand_dims(t1, axis=1)
#         return tf.squeeze(self.score_layer(self.activation(t0_t1)), axis=-1)


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
