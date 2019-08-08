from collections import deque
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from pytorch_mrc.model.base_model import BaseModel
from pytorch_mrc.nn.layers import VariationalDropout, Embedding, PointerNetwork
from pytorch_mrc.nn.recurrent import BiGRU, MultiLayerBiGRU
from pytorch_mrc.nn.util import sequence_mask, masked_softmax, mask_logits
from pytorch_mrc.nn.attention import CoAttention, MultiHeadAttention


class RNET(BaseModel):
    def __init__(self, vocab, device,
                 pretrained_word_embedding=None,
                 word_embedding_trainable=False,
                 word_embedding_size=300,
                 char_embedding_size=100,
                 heads=3,
                 encoder_layers_num=1,
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
        self.word_embedding = Embedding(pretrained_embedding=pretrained_word_embedding,
                                        embedding_shape=(len(vocab.get_word_vocab()), word_embedding_size),
                                        trainable=word_embedding_trainable)
        self.char_embedding = Embedding(embedding_shape=(len(vocab.get_char_vocab()), char_embedding_size),
                                        trainable=True, init_scale=0.2)
        self.char_bigru = BiGRU(char_embedding_size, hidden_size)

        # Encoder
        self.encoder_multi_bigru = MultiLayerBiGRU(word_embedding_size + 2 * hidden_size, hidden_size,
                                                   num_layers=encoder_layers_num,
                                                   input_drop_prob=dropout_prob)

        # Gated attention RNNs in the paper, here we use co-attention
        self.co_attention_layer = CoAttention(2 * hidden_size * encoder_layers_num,
                                              2 * hidden_size * encoder_layers_num,
                                              hidden_dim=hidden_size)

        # Self matching attention, here we use multi-head Attetion
        self.multi_head_att = MultiHeadAttention(heads, hidden_size, hidden_size, attention_on_itself=False)
        self.gate_dense = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.self_att_bigru = BiGRU(2 * hidden_size, hidden_size)

        # Output Layer
        self.pointer_net = PointerNetwork(2 * hidden_size, 2 * hidden_size, hidden_size)

        # RNN Dropout
        self.dropout = VariationalDropout(dropout_prob, batch_first=True)

    def forward(self, data):
        # Parsing data
        context_ids, context_len = data['context_ids'], data['context_len']
        question_ids, question_len = data['question_ids'], data['question_len']
        answer_start, answer_end = data['answer_start'], data['answer_end']
        context_char_ids, context_word_len = data['context_char_ids'], data['context_word_len']
        question_char_ids, question_word_len = data['question_char_ids'], data['question_word_len']

        # Record maximum length info and generate mask matrix
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
        _, last_hidden_state = self.char_bigru(context_char_embedding)  # 2*(B*CL)*H
        context_char_repr = torch.cat([last_hidden_state[0], last_hidden_state[1]], dim=-1)  # (B*CL)*2H
        context_char_repr = context_char_repr.reshape([-1, max_context_len, 2 * self.hidden_size])  # B*CL*2H

        _, last_hidden_state = self.char_bigru(question_char_embedding)  # 2*(B*QL)*H
        question_char_repr = torch.cat([last_hidden_state[0], last_hidden_state[1]], dim=-1)  # (B*QL)*2H
        question_char_repr = question_char_repr.reshape([-1, max_question_len, 2 * self.hidden_size])  # B*QL*2H

        # 1.3 Concat word and char representation
        context_repr = torch.cat([context_word_repr, context_char_repr], dim=-1)  # B*CL*(WD+2H)
        question_repr = torch.cat([question_word_repr, question_char_repr], dim=-1)  # B*QL*(WD+2H)

        # 2. Encoder
        encoder_context, _ = self.encoder_multi_bigru(context_repr, context_len)  # B*CL*(H*2*num_layers)
        encoder_question, _ = self.encoder_multi_bigru(question_repr, question_len)  # B*QL*(H*2*num_layers)
        encoder_context = self.dropout(encoder_context)
        encoder_question = self.dropout(encoder_question)

        # 3. Gated attention RNNs in the paper
        co_att_output = self.dropout(self.co_attention_layer(
            encoder_context, encoder_question, context_len, question_mask))  # B*CL*H

        # 4. Self matching attention
        self_att_repr = self.dropout(self.multi_head_att(co_att_output, co_att_output, co_att_output, context_mask))  # B*CL*H
        self_att_rnn_input = torch.cat([co_att_output, self_att_repr], dim=-1)  # B*CL*(H*2)
        self_att_rnn_input = self_att_rnn_input * torch.sigmoid(self.gate_dense(self_att_rnn_input))
        self_att_output, _ = self.self_att_bigru(self_att_rnn_input, context_len)  # B*CL*(H*2)
        self_att_output = self.dropout(self_att_output)  # B*CL*(H*2)

        # 5. Pointer Network
        start_logits, end_logits = self.pointer_net(self_att_output, encoder_question, context_mask, question_mask)
        self.start_prob = masked_softmax(start_logits, context_mask)
        self.end_prob = masked_softmax(end_logits, context_mask)

        # 6. Retured Things. If train return loss, if eval/inference return a dict
        # TODO for squad2.0 and for multi GPUs
        if answer_start is not None and answer_end is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            answer_start.clamp_(0, ignored_index)
            answer_end.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(mask_logits(start_logits, context_mask), answer_start)
            end_loss = loss_fct(mask_logits(end_logits, context_mask), answer_end)
            total_loss = start_loss + end_loss

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

    # def update(self, grad_clip=5.0):
    #     if not self.training:
    #         raise Exception("Only in the train mode, you can update the weights")
    #     if self.optimizer is None:
    #         raise Exception("The model need to compile!")
    #
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
    #     self.optimizer.step()

    def compile(self, optimizer=torch.optim.Adam, initial_lr=0.001):
        self.optimizer = optimizer(self.parameters(), lr=initial_lr)

    def get_best_answer(self, output, instances, max_len=15):
        answer_list = []
        for i in range(len(output['start_prob'])):
            instance = instances[i]
            max_prob = 0.0
            start_position = 0
            end_position = 0
            d = deque()
            start_prob, end_prob = output['start_prob'][i], output['end_prob'][i]
            for idx in range(len(start_prob)):
                while len(d) > 0 and idx - d[0] >= max_len:
                    d.popleft()
                while len(d) > 0 and start_prob[d[-1]] <= start_prob[idx]:
                    d.pop()
                d.append(idx)
                if start_prob[d[0]] * end_prob[idx] > max_prob:
                    start_position = d[0]
                    end_position = idx
                    max_prob = start_prob[d[0]] * end_prob[idx]
            char_start_position = instance["context_token_spans"][start_position][0]
            char_end_position = instance["context_token_spans"][end_position][1]
            pred_answer = instance["context"][char_start_position:char_end_position]
            answer_list.append(pred_answer)
        return answer_list
