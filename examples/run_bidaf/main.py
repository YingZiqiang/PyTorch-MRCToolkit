# coding: utf-8

import sys
sys.path.append('../..')
from pytorch_mrc.data.vocabulary import Vocabulary
from pytorch_mrc.dataset.squad import SquadReader, SquadEvaluator
# from pytorch_mrc.model.bidaf import BiDAF
# import tensorflow as tf
import logging
# from pytorch_mrc.data.batch_generator import BatchGenerator

# tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
data_folder = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/'
embedding_folder = '/home/len/yingzq/nlp/mrc_dataset/word_embeddings/'
# train_file = data_folder + "dev-v1.1.json"
dev_file = data_folder + "dev-v1.1.json"
embedding_file = 'glove.6B.100d.txt'

reader = SquadReader()
# train_data = reader.read(train_file)
eval_data = reader.read(dev_file)
evaluator = SquadEvaluator(dev_file)

vocab = Vocabulary()
vocab.build_vocab(eval_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding(embedding_folder + embedding_file)
print(word_embedding.shape)
print('successful!')

# train_batch_generator = BatchGenerator(vocab, train_data, batch_size=60, training=True)
#
# eval_batch_generator = BatchGenerator(vocab, eval_data, batch_size=60)
#
# model = BiDAF(vocab, pretrained_word_embedding=word_embedding)
# model.compile(tf.train.AdamOptimizer, 0.001)
# model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=15, eposides=2)
