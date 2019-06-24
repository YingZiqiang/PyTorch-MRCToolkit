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


def mask_logits(logits, mask):
    if len(logits.size()) != len(mask.size()):
        mask = sequence_mask(mask, maxlen=logits.size(1), dtype=torch.float32)
    return logits + (1.0 - mask) * VERY_NEGATIVE_NUMBER


def masked_softmax(logits, mask):
    return F.softmax(mask_logits(logits, mask), dim=-1)


def add_seq_mask(inputs, seq_len, mode='mul', max_len=None):
    mask = sequence_mask(seq_len, maxlen=max_len, dtype=torch.float32).unsqueeze(2)
    if mode == 'mul':
        return inputs * mask
    if mode == 'add':
        mask = (1 - mask) * VERY_NEGATIVE_NUMBER
        return inputs + mask
