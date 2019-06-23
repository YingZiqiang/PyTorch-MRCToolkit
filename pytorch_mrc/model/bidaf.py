# coding:utf-8
import os
import logging
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from pytorch_mrc.model.base_model import BaseModel
from pytorch_mrc.nn.layers import Conv1DAndMaxPooling, Dropout, Highway, Embedding, ElmoEmbedding
from pytorch_mrc.nn.recurrent import BiLSTM
from pytorch_mrc.nn.attention import BiAttention
from pytorch_mrc.nn.similarity_function import TriLinear
from pytorch_mrc.nn.ops import masked_softmax, weighted_sum, mask_logits


class BiDAF(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None, word_embedding_size=100, char_embedding_size=8,
                 char_conv_filters=100,
                 char_conv_kernel_size=5, rnn_hidden_size=100,
                 dropout_prob=0.2, max_answer_len=17, word_embedding_trainable=False,use_elmo=False,elmo_local_path=None,
                 enable_na_answer=False):
        super(BiDAF, self).__init__(vocab)
        self.rnn_hidden_size = rnn_hidden_size
        self.drop_prob = dropout_prob
        self.word_embedding_size = word_embedding_size
        self.pretrained_word_embedding = pretrained_word_embedding
        self.char_embedding_size = char_embedding_size
        self.char_conv_filters = char_conv_filters
        self.char_conv_kernel_size = char_conv_kernel_size
        self.max_answer_len = max_answer_len
        self.use_elmo = use_elmo
        self.elmo_local_path= elmo_local_path
        self.word_embedding_trainable = word_embedding_trainable
        self.enable_na_answer = enable_na_answer # for squad2.0

        # Embedding
        # TODO UNK token need to handle and char embed need to do
        word_embedding = Embedding(pretrained_embedding=pretrained_word_embedding,
                                   embedding_shape=(len(vocab.get_word_vocab()), word_embedding_size),
                                   trainable=word_embedding_trainable)
        # char_embedding = Embedding(embedding_shape=(len(self.vocab.get_char_vocab()), self.char_embedding_size),
        #                            trainable=True, init_scale=0.2)

        # TODO Encode Layer, maybe context and question need to encode separately
        self.encode_phrase_lstm = BiLSTM(100, rnn_hidden_size)

        # Attention Flow Layer
        self.bi_attention = BiAttention(TriLinear(input_units=2*rnn_hidden_size))

        # Modeling Layer
        self.modeling_lstm = BiLSTM(8*rnn_hidden_size, rnn_hidden_size)

        # Output Layer
        self.start_pred_layer = nn.Linear(10*rnn_hidden_size, 1, bias=False)
        self.end_lstm = BiLSTM(2*rnn_hidden_size, rnn_hidden_size)
        self.end_pred_layer = nn.Linear(10*rnn_hidden_size, 1, bias=False)

        # TODO Dropout
        self.dropout = nn.Dropout(p=self.drop_prob)


    def forward(self, data):
        # Parsing data
        context_ids, context_len = torch.tensor(data['context_ids']), torch.tensor(data['context_len'])
        question_ids, question_len = torch.tensor(data['question_ids']), torch.tensor(data['question_len'])
        start_positions, end_positions = torch.tensor(data['answer_start']), torch.tensor(data['answer_end'])

        # 1.1 Embedding
        context_word_repr = self.word_embedding(context_ids)
        # context_char_repr = self.char_embedding(context_char)
        question_word_repr = self.word_embedding(question_ids)
        # question_char_repr = self.char_embedding(question_char)

        # 1.2 Char convolution
        # TODO

        # elmo embedding
        # TODO

        #concat word and char
        # TODO

        # 1.3 Highway network
        # TODO

        # temp repr
        context_repr, question_repr = context_word_repr, question_word_repr

        # 2. Phrase encoding
        context_repr, _ = self.encode_phrase_lstm(context_repr, context_len)
        question_repr, _ = self.encode_phrase_lstm(question_repr, question_len)

        # 3. Bi-Attention
        c2q, q2c = self.bi_attention(context_repr, question_repr, context_len, question_len)

        # 4. Modeling layer
        final_merged_context = torch.cat([context_repr, c2q, context_repr*c2q, context_repr*q2c], dim=-1)
        modeled_context = self.modeling_lstm(final_merged_context, context_len)

        # 5. Start prediction
        start_logits = self.start_pred_layer(torch.cat([final_merged_context, modeled_context], dim=-1))
        start_logits = start_logits.squeeze(-1)
        self.start_prob = masked_softmax(start_logits, context_len)

        # 6. End prediction
        # TODO I just follow the BiDAF paper, maybe it can do better
        # start_repr = weighted_sum(modeled_context, self.start_prob)
        # tiled_start_repr = start_repr.unsqueeze(1).repeat(1, modeled_context.size(1), 1)
        encoded_end_repr = self.end_lstm(modeled_context, context_len)
        end_logits = self.end_pred_layer(torch.cat([final_merged_context, encoded_end_repr], dim=-1))
        end_logits = end_logits.squeeze(-1)
        self.end_prob = masked_softmax(end_logits, context_len)

        # 7. If train return loss, if eval return start_logits and end_logits
        # TODO for squad2.0
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


    def compile(self, initial_lr):
        self.optimizer = Adam(self.parameters(), lr=initial_lr)

    def get_best_answer(self, output, instances):
        na_prob = {}
        preds_dict = {}
        for i in range(len(instances)):
            instance = instances[i]
            max_prob, max_start, max_end = 0, 0, 0
            for end in range(output['end_prob'][i].shape[0]):
                for start in range(max(0, end - self.max_answer_len + 1), end + 1):
                    prob = output["start_prob"][i][start] * output["end_prob"][i][end]
                    if prob > max_prob:
                        max_start, max_end = start, end
                        max_prob = prob

            char_start_position = instance["context_token_spans"][max_start][0]
            char_end_position = instance["context_token_spans"][max_end][1]
            pred_answer = instance["context"][char_start_position:char_end_position]
            if not self.enable_na_answer:
                preds_dict[instance['qid']] = pred_answer
            else:
                preds_dict[instance['qid']] = pred_answer if max_prob > output['na_prob'][i] else ''
                na_prob[instance['qid']] = output['na_prob'][i]

        return preds_dict if not self.enable_na_answer else (preds_dict,na_prob)
