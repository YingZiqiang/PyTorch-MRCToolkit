import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from pytorch_mrc.model.base_model import BaseModel
from pytorch_mrc.nn.layers import VariationalDropout, Embedding, PointerNetwork
from pytorch_mrc.nn.recurrent import BiGRU, MultiLayerBiGRU
from pytorch_mrc.nn.util import sequence_mask, masked_softmax
from pytorch_mrc.nn.attention import CoAttention, MultiHeadAttention


class RNET(BaseModel):
    def __init__(self, vocab, device,
                 pretrained_word_embedding=None,
                 word_embedding_trainable=False,
                 word_embedding_size=300,
                 char_embedding_size=100,
                 encoder_layers_num=3,
                 hidden_size=75,
                 dropout_prob=0.2):
        super(RNET, self).__init__(vocab, device)
        self.pretrained_word_embedding = pretrained_word_embedding
        self.word_embedding_trainable = word_embedding_trainable
        self.word_embedding_size = word_embedding_size
        self.char_embedding_size = char_embedding_size
        self.encoder_layers_num = encoder_layers_num
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # Embedding
        # TODO UNK token need to handle
        self.word_embedding = Embedding(pretrained_embedding=pretrained_word_embedding,
                                        embedding_shape=(len(vocab.get_word_vocab()), word_embedding_size),
                                        trainable=word_embedding_trainable)
        self.char_embedding = Embedding(embedding_shape=(len(vocab.get_char_vocab()), char_embedding_size),
                                        trainable=True, init_scale=0.2)
        self.char_bigru = BiGRU(char_embedding_size, hidden_size)

        # Encoder
        self.encoder_multi_bigru = MultiLayerBiGRU(word_embedding_size + 2 * hidden_size, hidden_size,
                                                   num_layers=encoder_layers_num, input_drop_prob=dropout_prob)

        # Gated attention RNNs in the paper, here we call it Co-attention
        self.co_attention_layer = CoAttention(2 * encoder_layers_num * hidden_size, hidden_size)

        # Self matching attention
        self.multi_head_att = MultiHeadAttention(3, 2 * hidden_size, hidden_size, attention_on_itself=False)
        self.gate_dense = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.self_att_bigru = BiGRU(2 * hidden_size, hidden_size)

        # Output Layer
        self.pointer_net = PointerNetwork()

        # RNN Dropout
        self.dropout = VariationalDropout(dropout_prob, batch_first=True)

    def forward(self, data):
        # Parsing data
        context_ids, context_len = data['context_ids'], data['context_len']
        question_ids, question_len = data['question_ids'], data['question_len']
        answer_start, answer_end = data['answer_start'], data['answer_end']
        context_char_ids, context_word_len = data['context_char_ids'], data['context_word_len']
        question_char_ids, question_word_len = data['question_char_ids'], data['question_word_len']

        # Record maximum length info; Generate mask matrix
        max_context_len = context_ids.size(1)
        max_context_word_len = context_char_ids.size(2)
        max_question_len = question_ids.size(1)
        max_question_word_len = question_char_ids.size(2)
        context_mask = sequence_mask(context_len, maxlen=max_context_len)
        question_mask = sequence_mask(question_len, maxlen=max_question_len)

        # 1. Context and Question Encoder
        # 1.1 Word and char embedding
        context_word_repr = self.word_embedding(context_ids)  # B*CL*WD
        context_char_embedding = self.dropout(self.char_embedding(context_char_ids).reshape(
            [-1, max_context_word_len, self.char_embedding_size]))  # (B*CL)*WL*CD
        question_word_repr = self.word_embedding(question_ids)  # B*QL*WD
        question_char_embedding = self.dropout(self.char_embedding(question_char_ids).reshape(
            [-1, max_question_word_len, self.char_embedding_size]))  # (B*QL)*WL*CD

        # 1.2 Char-level representation
        _, last_hidden_state = self.char_bigru(context_char_embedding, context_word_len.reshape([-1]))  # 2*(B*CL)*H
        context_char_repr = torch.cat([last_hidden_state[0].squeeze(0), last_hidden_state[1].squeeze(0)], dim=-1)  # (B*CL)*2H
        context_char_repr = context_char_repr.reshape([-1, max_context_len, 2 * self.hidden_size])  # B*CL*2H

        _, last_hidden_state = self.char_bigru(question_char_embedding, question_word_len.reshape([-1]))  # 2*(B*QL)*H
        question_char_repr = torch.cat([last_hidden_state[0].squeeze(0), last_hidden_state[1].squeeze(0)], dim=-1)  # (B*QL)*2H
        question_char_repr = question_char_repr.reshape([-1, max_question_len, 2 * self.hidden_size])  # B*QL*2H

        # 1.3 Concat word and char representation
        context_repr = torch.cat([context_word_repr, context_char_repr], dim=-1)  # B*CL*(WD+2H)
        question_repr = torch.cat([question_word_repr, question_char_repr], dim=-1)  # B*QL*(WD+2H)

        # 2. Encoder
        encoder_context, _ = self.encoder_multi_bigru(context_repr, context_len, concat_layers=True)  # B*CL*(H*2*num_layers)
        encoder_question, _ = self.encoder_multi_bigru(question_repr, question_len, concat_layers=True)  # B*QL*(H*2*num_layers)
        encoder_context = self.dropout(encoder_context)
        encoder_question = self.dropout(encoder_question)

        # 3. Gated attention RNNs in the paper, here we call it Co-attention
        co_att_output = self.dropout(self.co_attention_layer(
            encoder_context, encoder_question, context_len, question_mask))  # B*CL*(H*2)

        # 4. Self matching attention
        self_att_repr = self.dropout(self.multi_head_att(co_att_output, co_att_output, co_att_output, context_mask))
        self_att_rnn_input = torch.cat([co_att_output, self_att_repr], dim=-1)  # B*CL*(H*2)
        self_att_rnn_input = self_att_rnn_input * torch.sigmoid(self.gate_dense(self_att_rnn_input))
        self_att_output, _ = self.self_att_rnn(self_att_rnn_input, context_len)
        self_att_output = self.dropout(self_att_output)  # B*CL*(H*2)

        # 5. Pointer Network
