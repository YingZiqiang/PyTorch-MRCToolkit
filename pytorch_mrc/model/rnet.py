import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from pytorch_mrc.model.base_model import BaseModel
from pytorch_mrc.nn.layers import VariationalDropout, Embedding
from pytorch_mrc.nn.recurrent import BiGRU


class RNET(BaseModel):
    def __init__(self, vocab, device,
                 pretrained_word_embedding=None,
                 word_embedding_trainable=False,
                 word_embedding_size=300,
                 char_embedding_size=100,
                 hidden_size=75,
                 dropout_prob=0.2):
        super(RNET, self).__init__(vocab, device)
        self.pretrained_word_embedding = pretrained_word_embedding
        self.word_embedding_trainable = word_embedding_trainable
        self.word_embedding_size = word_embedding_size
        self.char_embedding_size = char_embedding_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # Embedding
        # TODO UNK token need to handle
        self.word_embedding = Embedding(pretrained_embedding=pretrained_word_embedding,
                                        embedding_shape=(len(vocab.get_word_vocab()), word_embedding_size),
                                        trainable=word_embedding_trainable)
        self.char_embedding = Embedding(embedding_shape=(len(vocab.get_char_vocab()), char_embedding_size),
                                        trainable=True, init_scale=0.2)
        self.char_gru = BiGRU(char_embedding_size, hidden_size)

        # Encoder
        self.encoder_bigru = BiGRU(word_embedding_size + 2 * hidden_size, hidden_size)

        # RNN Dropout
        self.dropout = VariationalDropout(dropout_prob, batch_first=True)

    def forward(self, data):
        # Parsing data
        context_ids, context_len = data['context_ids'], data['context_len']
        question_ids, question_len = data['question_ids'], data['question_len']
        answer_start, answer_end = data['answer_start'], data['answer_end']
        context_char_ids, context_word_len = data['context_char_ids'], data['context_word_len']
        question_char_ids, question_word_len = data['question_char_ids'], data['question_word_len']

        # Record Maximum Length Info
        max_context_len = context_ids.size(1)
        max_context_word_len = context_char_ids.size(2)
        max_question_len = question_ids.size(1)
        max_question_word_len = question_char_ids.size(2)

        # 1. Context and Question Encoder
        # 1.1 Word and char embedding
        context_word_repr = self.word_embedding(context_ids)  # B*CL*WD
        context_char_embedding = self.dropout(self.char_embedding(context_char_ids).reshape(
            [-1, max_context_word_len, self.char_embedding_size]))  # (B*CL)*WL*CD
        question_word_repr = self.word_embedding(question_ids)  # B*QL*WD
        question_char_embedding = self.dropout(self.char_embedding(question_char_ids).reshape(
            [-1, max_question_word_len, self.char_embedding_size]))  # (B*QL)*WL*CD

        # 1.2 Char-level representation
        _, last_hidden_state = self.char_gru(context_char_embedding, context_word_len.reshape([-1]))  # 2*(B*CL)*H
        context_char_repr = torch.cat([last_hidden_state[0].squeeze(0), last_hidden_state[1].squeeze(0)], dim=-1)  # (B*CL)*2H
        context_char_repr = context_char_repr.reshape([-1, max_context_len, 2 * self.hidden_size])  # B*CL*2H

        _, last_hidden_state = self.char_gru(question_char_embedding, question_word_len.reshape([-1]))  # 2*(B*QL)*H
        question_char_repr = torch.cat([last_hidden_state[0].squeeze(0), last_hidden_state[1].squeeze(0)], dim=-1)  # (B*QL)*2H
        question_char_repr = question_char_repr.reshape([-1, max_question_len, 2 * self.hidden_size])  # B*QL*2H

        # 1.3 Concat word and char representation
        context_repr = torch.cat([context_word_repr, context_char_repr], dim=-1)  # B*CL*(WD+2H)
        question_repr = torch.cat([question_word_repr, question_char_repr], dim=-1)  # B*QL*(WD+2H)

        # 2. Encoder
