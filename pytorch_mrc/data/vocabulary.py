import pickle
import logging

import numpy as np
from tqdm import tqdm
from collections import Counter


class Vocabulary(object):
    def __init__(self, do_lowercase=True, special_tokens=None):
        self.word_vocab = None
        self.char_vocab = None
        self.word2idx = None
        self.char2idx = None
        self.word_embedding_matrix = None
        self.special_tokens = special_tokens
        self.do_lowercase = do_lowercase  # only for word

        # Initial Tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.initial_tokens = (self.pad_token, self.unk_token)

    def build_vocab(self, instances, min_word_count=-1, min_char_count=-1):
        self.word_vocab = [token for token in self.initial_tokens]
        self.char_vocab = [token for token in self.initial_tokens]

        self.word_counter = Counter()
        char_counter = Counter()
        if self.special_tokens is not None and isinstance(self.special_tokens, list):
            self.word_vocab.extend(self.special_tokens)

        logging.info("Building vocabulary.")
        for instance in tqdm(instances):
            for token in instance['context_tokens']:
                for char in token:
                    char_counter[char] += 1
                token = token.lower() if self.do_lowercase else token
                self.word_counter[token] += 1
            for token in instance['question_tokens']:
                for char in token:
                    char_counter[char] += 1
                token = token.lower() if self.do_lowercase else token
                self.word_counter[token] += 1
        for w, v in self.word_counter.most_common():
            if v >= min_word_count:
                self.word_vocab.append(w)
        for c, v in char_counter.most_common():
            if v >= min_char_count:
                self.char_vocab.append(c)

        self._build_index_mapper()

    def set_vocab(self, word_vocab, char_vocab):
        self.word_vocab = [token for token in self.initial_tokens]
        self.char_vocab = [token for token in self.initial_tokens]
        if self.special_tokens is not None and isinstance(self.special_tokens, list):
            self.word_vocab.extend(self.special_tokens)

        self.word_vocab += word_vocab
        self.char_vocab += char_vocab

        self._build_index_mapper()

    def _build_index_mapper(self):
        self.word2idx = dict(zip(self.word_vocab, range(len(self.word_vocab))))
        self.char2idx = dict(zip(self.char_vocab, range(len(self.char_vocab))))

    def make_word_embedding(self, embedding_file, init_scale=0.02):
        if self.word_vocab is None or self.word2idx is None:
            raise ValueError("make_word_embedding must be called after build_vocab/set_vocab")

        # 1. Parse pretrained embedding
        embedding_dict = dict()
        with open(embedding_file) as f:
            for line in f:
                if len(line.rstrip().split(" ")) <= 2:
                    continue
                word, vector = line.rstrip().split(" ", 1)
                embedding_dict[word] = np.fromstring(vector, dtype=np.float, sep=" ")

        # 2. Update word vocab according to pretrained word embedding
        new_word_vocab = []
        special_tokens_set = set(self.special_tokens if self.special_tokens is not None else [])
        for word in self.word_vocab:
            if word in self.initial_tokens or word in special_tokens_set or word in embedding_dict:
                new_word_vocab.append(word)
        self.word_vocab = new_word_vocab
        self._build_index_mapper()

        # 3. Make word embedding matrix
        embedding_size = embedding_dict[list(embedding_dict.keys())[0]].shape[0]
        embedding_list = []
        for word in self.word_vocab:
            if word == self.pad_token:
                embedding_list.append(np.zeros([1, embedding_size], dtype=np.float))
            elif word == self.unk_token or word in special_tokens_set:
                embedding_list.append(np.random.uniform(-init_scale, init_scale, [1, embedding_size]))
            else:
                embedding_list.append(np.reshape(embedding_dict[word], [1, embedding_size]))

        self.word_embedding_matrix = np.concatenate(embedding_list, axis=0)

    def get_word_pad_idx(self):
        return self.word2idx[self.pad_token]

    def get_char_pad_idx(self):
        return self.char2idx[self.pad_token]

    def get_word_unk_idx(self):
        return self.word2idx[self.unk_token]

    def get_char_unk_idx(self):
        return self.char2idx[self.unk_token]

    def get_word_idx(self, token):
        token = token.lower() if self.do_lowercase else token
        if token in self.word_vocab:
            return self.word2idx[token]
        else:
            return self.get_word_unk_idx()

    def get_char_idx(self, token):
        if token in self.char_vocab:
            return self.char2idx[token]
        else:
            return self.get_char_unk_idx()

    def get_word_vocab(self):
        return self.word_vocab

    def get_char_vocab(self):
        return self.char_vocab

    def get_word_counter(self):
        return self.word_counter

    def get_word_embedding(self):
        if self.word_embedding_matrix is None:
            raise ValueError("get_word_embedding must be called after make_word_embedding")
        return self.word_embedding_matrix

    def save(self, file_path, include_word_embedding=True):
        logging.info("Saving vocabulary at {}".format(file_path))
        # if include_word_embedding is False, we will not save the word embedding matrix
        if not include_word_embedding:
            tmp_word_emb = self.word_embedding_matrix
            self.word_embedding_matrix = None
        with open(file_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        if not include_word_embedding:
            self.word_embedding_matrix = tmp_word_emb

    def load(self, file_path):
        logging.info("Loading vocabulary at {}".format(file_path))
        with open(file_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.__dict__.update(vocab_data)
