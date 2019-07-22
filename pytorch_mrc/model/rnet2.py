from collections import deque
import torch
from torch.nn import CrossEntropyLoss

from pytorch_mrc.model.base_model import BaseModel
from pytorch_mrc.nn.dropout import VariationalDropout
from pytorch_mrc.nn.layers import Embedding, StaticPairEncoder, StaticSelfMatchEncoder, PointerNetwork
from pytorch_mrc.nn.recurrent import BiGRU, MultiLayerBiGRU
from pytorch_mrc.nn.util import sequence_mask, masked_softmax, mask_logits


class RNET(BaseModel):
    def __init__(self, vocab, device,
                 pretrained_word_embedding=None,
                 word_embedding_trainable=False,
                 word_embedding_size=300,
                 char_embedding_size=8,
                 char_hidden_size=100,
                 encoder_layers_num=3,
                 hidden_size=75,
                 dropout_prob=0.3):
        super(RNET, self).__init__(vocab, device)
        self.pretrained_word_embedding = pretrained_word_embedding
        self.word_embedding_trainable = word_embedding_trainable
        self.word_embedding_size = word_embedding_size
        self.char_embedding_size = char_embedding_size
        self.char_hidden_size = char_hidden_size
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
        self.char_bigru = BiGRU(char_embedding_size, char_hidden_size)

        # Encoder
        self.encoder_multi_bigru = MultiLayerBiGRU(word_embedding_size + 2 * char_hidden_size, hidden_size,
                                                   num_layers=encoder_layers_num, input_drop_prob=dropout_prob)

        # Gated attention RNNs
        self.gated_att_bigru = StaticPairEncoder(2 * encoder_layers_num * hidden_size,
                                                 2 * encoder_layers_num * hidden_size,
                                                 hidden_dim=hidden_size, drop_prob=dropout_prob)

        # Self matching attention
        self.self_match_att = StaticSelfMatchEncoder(2 * hidden_size, 2 * hidden_size,
                                                     hidden_dim=hidden_size, drop_prob=dropout_prob)

        # Output Layer
        self.pointer_net = PointerNetwork(2 * hidden_size, 2 * encoder_layers_num * hidden_size,
                                          hidden_dim=hidden_size, drop_prob=dropout_prob)

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
        _, last_hidden_state = self.char_bigru(context_char_embedding, context_word_len.reshape([-1]))  # 2*(B*CL)*CH
        context_char_repr = torch.cat([last_hidden_state[0], last_hidden_state[1]], dim=-1)  # (B*CL)*2CH
        context_char_repr = context_char_repr.reshape([-1, max_context_len, 2 * self.char_hidden_size])  # B*CL*2CH

        _, last_hidden_state = self.char_bigru(question_char_embedding, question_word_len.reshape([-1]))  # 2*(B*QL)*CH
        question_char_repr = torch.cat([last_hidden_state[0], last_hidden_state[1]], dim=-1)  # (B*QL)*2CH
        question_char_repr = question_char_repr.reshape([-1, max_question_len, 2 * self.char_hidden_size])  # B*QL*2CH

        # 1.3 Concat word and char representation
        context_repr = torch.cat([context_word_repr, context_char_repr], dim=-1)  # B*CL*(WD+2CH)
        question_repr = torch.cat([question_word_repr, question_char_repr], dim=-1)  # B*QL*(WD+2CH)

        # 2. Encoder
        encoder_context, _ = self.encoder_multi_bigru(context_repr, context_len, concat_layers=True)  # B*CL*(H*2*num_layers)
        encoder_question, _ = self.encoder_multi_bigru(question_repr, question_len, concat_layers=True)  # B*QL*(H*2*num_layers)

        # 3. Gated attention RNNs in the paper
        encoder_context = self.gated_att_bigru(encoder_context, encoder_question, context_len, question_mask)  # B*CL*2H

        # 4. Self matching attention
        encoder_context = self.self_match_att(encoder_context, encoder_context, context_len, context_mask)  # B*CL*2H

        # 5. Pointer Network
        start_logits, end_logits = self.pointer_net(encoder_context, encoder_question, context_mask, question_mask)
        start_prob = masked_softmax(start_logits, context_mask)
        end_prob = masked_softmax(end_logits, context_mask)

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
            total_loss = (start_loss + end_loss) / 2

            if self.training:
                return total_loss
            else:
                output_dict = {
                    "start_prob": start_prob.cpu().numpy(),
                    "end_prob": end_prob.cpu().numpy()
                }
                return total_loss, output_dict
        else:
            output_dict = {
                "start_prob": start_prob.cpu().numpy(),
                "end_prob": end_prob.cpu().numpy()
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
