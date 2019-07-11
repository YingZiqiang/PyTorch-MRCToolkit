import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from pytorch_mrc.model.base_model import BaseModel
from pytorch_mrc.nn.layers import Embedding, Conv1DAndMaxPooling, Highway
from pytorch_mrc.nn.recurrent import BiLSTM
from pytorch_mrc.nn.attention import BiAttention
from pytorch_mrc.nn.similarity_function import TriLinear
from pytorch_mrc.nn.util import sequence_mask, masked_softmax, mask_logits, weighted_sum


class BiDAF(BaseModel):
    def __init__(self, vocab, device,
                 pretrained_word_embedding=None,
                 word_embedding_trainable=False,
                 word_embedding_size=100,
                 char_embedding_size=8,
                 char_conv_filters=100,
                 char_conv_kernel_size=5,
                 use_elmo=False,
                 elmo_local_path=None,
                 rnn_hidden_size=100,
                 dropout_prob=0.2,
                 max_answer_len=17,
                 enable_na_answer=False):
        super(BiDAF, self).__init__(vocab, device)
        self.use_elmo = use_elmo
        self.elmo_local_path = elmo_local_path
        self.max_answer_len = max_answer_len
        self.enable_na_answer = enable_na_answer  # for squad2.0

        # Embedding
        # TODO UNK token need to handle
        self.word_embedding = Embedding(pretrained_embedding=pretrained_word_embedding,
                                        embedding_shape=(len(vocab.get_word_vocab()), word_embedding_size),
                                        trainable=word_embedding_trainable)
        self.char_embedding = Embedding(embedding_shape=(len(vocab.get_char_vocab()), char_embedding_size),
                                        trainable=True, init_scale=0.2)
        embedding_dim = word_embedding_size + char_conv_filters
        self.conv1d = Conv1DAndMaxPooling(char_embedding_size, char_conv_filters, char_conv_kernel_size)
        if use_elmo:
            # TODO
            pass
            # embedding_dim += ??
        self.highway = Highway(input_dim=embedding_dim, num_layers=2)

        # TODO Encode Layer, maybe context and question need to encode separately
        self.encode_phrase_lstm = BiLSTM(embedding_dim, rnn_hidden_size)

        # Attention Flow Layer
        self.bi_attention = BiAttention(TriLinear(input_dim=2 * rnn_hidden_size))

        # Modeling Layer
        self.modeling_lstm1 = BiLSTM(8 * rnn_hidden_size, rnn_hidden_size)
        self.modeling_lstm2 = BiLSTM(2 * rnn_hidden_size, rnn_hidden_size)

        # Output Layer
        self.start_pred_layer = nn.Linear(10 * rnn_hidden_size, 1, bias=False)
        self.end_lstm = BiLSTM(14 * rnn_hidden_size, rnn_hidden_size)
        self.end_pred_layer = nn.Linear(10 * rnn_hidden_size, 1, bias=False)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, data):
        # Parsing data
        context_ids, context_len = data['context_ids'], data['context_len']
        question_ids, question_len = data['question_ids'], data['question_len']
        answer_start, answer_end = data['answer_start'], data['answer_end']
        context_char_ids, context_word_len = data['context_char_ids'], data['context_word_len']
        question_char_ids, question_word_len = data['question_char_ids'], data['question_word_len']

        # compute mask
        context_mask = sequence_mask(context_len, maxlen=context_ids.size(1))
        question_mask = sequence_mask(question_len, maxlen=question_ids.size(1))

        # 1.1 Embedding
        context_word_repr = self.word_embedding(context_ids)
        context_char_repr = self.char_embedding(context_char_ids)
        question_word_repr = self.word_embedding(question_ids)
        question_char_repr = self.char_embedding(question_char_ids)

        # 1.2 Char convolution
        context_char_repr = self.dropout(self.conv1d(context_char_repr, context_word_len))
        question_char_repr = self.dropout(self.conv1d(question_char_repr, question_word_len))

        # 1.3 Concat word and char
        context_repr = torch.cat([context_word_repr, context_char_repr], dim=-1)
        question_repr = torch.cat([question_word_repr, question_char_repr], dim=-1)

        # 1.4 ELMo embedding
        if self.use_elmo:
            # TODO
            pass

        # 1.5 Highway network
        context_repr = self.highway(context_repr)
        question_repr = self.highway(question_repr)

        # 2. Phrase encoding
        context_repr, _ = self.encode_phrase_lstm(self.dropout(context_repr), context_len)
        question_repr, _ = self.encode_phrase_lstm(self.dropout(question_repr), question_len)

        # 3. Bi-Attention
        c2q, q2c = self.bi_attention(context_repr, question_repr, context_mask, question_mask)

        # 4. Modeling layer
        final_merged_context = torch.cat([context_repr, c2q, context_repr * c2q, context_repr * q2c], dim=-1)
        modeled_context1, _ = self.modeling_lstm1(self.dropout(final_merged_context), context_len)
        modeled_context2, _ = self.modeling_lstm2(self.dropout(modeled_context1), context_len)
        modeled_context = modeled_context1 + modeled_context2

        # 5. Start prediction
        start_logits = self.start_pred_layer(self.dropout(torch.cat([final_merged_context, modeled_context], dim=-1)))
        start_logits = start_logits.squeeze(-1)
        self.start_prob = masked_softmax(start_logits, context_mask)

        # 6. End prediction
        start_repr = weighted_sum(modeled_context, self.start_prob)
        tiled_start_repr = start_repr.unsqueeze(1).repeat(1, modeled_context.size(1), 1)
        encoded_end_repr, _ = self.end_lstm(self.dropout(torch.cat(
            [final_merged_context, modeled_context, tiled_start_repr, modeled_context * tiled_start_repr], dim=-1)),
            context_len)
        end_logits = self.end_pred_layer(self.dropout(torch.cat([final_merged_context, encoded_end_repr], dim=-1)))
        end_logits = end_logits.squeeze(-1)
        self.end_prob = masked_softmax(end_logits, context_mask)

        # 7. Retured Things. If train return loss, if eval/inference return a dict
        # TODO for squad2.0 and for multi GPUs
        if answer_start is not None and answer_end is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            answer_start.clamp_(0, ignored_index)
            answer_end.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(mask_logits(start_logits, context_mask), answer_start)
            end_loss = loss_fct(mask_logits(end_logits, context_mask), answer_end)
            total_loss = (start_loss + end_loss) / 2

            if self.training:
                return total_loss
            else:
                output_dict = {
                    "start_prob": self.start_prob.cpu().numpy(),
                    "end_prob": self.end_prob.cpu().numpy()
                }
                return total_loss, output_dict
        else:
            output_dict = {
                "start_prob": self.start_prob.cpu().numpy(),
                "end_prob": self.end_prob.cpu().numpy()
            }
            return output_dict

    def compile(self, optimizer=torch.optim.Adam, initial_lr=0.001):
        self.optimizer = optimizer(self.parameters(), lr=initial_lr)

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

        return preds_dict if not self.enable_na_answer else (preds_dict, na_prob)
