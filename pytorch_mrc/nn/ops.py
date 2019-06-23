import torch
import torch.nn.functional as F

VERY_NEGATIVE_NUMBER = -1e29


def sequence_mask(lengths, maxlen=None, dtype=torch.float32):
    # TODO come from tf.sequence_mask. There should be better implementation
    if maxlen is None:
        maxlen = lengths.max().item()
    mask = torch.zeros(len(lengths), maxlen, dtype=dtype)
    for idx, real_len in enumerate(lengths):
        mask[idx, :real_len] = 1
    return mask


# def dropout(x, keep_prob, training, noise_shape=None):
#     if keep_prob >= 1.0:
#         return x
#     return tf.cond(training, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape), lambda: x)


def weighted_sum(seq, prob):
    return (seq * prob.unsqueeze(2)).sum(1)


def masked_softmax(logits, mask):
    if len(logits.size()) != len(mask.size()):
        mask = sequence_mask(mask, maxlen=logits.size(1), dtype=torch.float32)

    return F.softmax(logits + (1.0 - mask) * VERY_NEGATIVE_NUMBER, dim=-1)


# def mask_logits(logits, mask):
#     if len(logits.shape.as_list()) != len(mask.shape.as_list()):
#         mask = tf.sequence_mask(mask, tf.shape(logits)[1], dtype=tf.float32)
#
#     return logits + (1.0 - mask) * tf.float32.min

# def add_seq_mask(inputs, seq_len, mode='mul', max_len=None):
#     mask = tf.expand_dims(tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32), 2)
#     if mode == 'mul':
#         return inputs * mask
#     if mode == 'add':
#         mask = (1 - mask) * tf.float32.min
#         return inputs + mask
