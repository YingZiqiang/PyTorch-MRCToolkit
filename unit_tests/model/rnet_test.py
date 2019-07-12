import sys
sys.path.append('../..')
import logging
import torch
from pytorch_mrc.data.vocabulary import Vocabulary
from pytorch_mrc.dataset.squad import SquadReader, SquadEvaluator
from pytorch_mrc.model.rnet1 import RNET
from pytorch_mrc.data.batch_generator import BatchGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
data_folder = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/'
embedding_folder = '/home/len/yingzq/nlp/mrc_dataset/word_embeddings/'
tiny_file = data_folder + "smaller_tiny-v1.1.json"
embedding_file = embedding_folder + 'glove.6B.100d.txt'

reader = SquadReader()
tiny_data = reader.read(tiny_file)
evaluator = SquadEvaluator(tiny_file)

logging.info('building vocab and making embedding...')
vocab = Vocabulary(do_lowercase=False)
vocab.build_vocab(tiny_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding(embedding_file)
logging.info('word vocab size: {}, word embedding shape: {}'.format(len(vocab.get_word_vocab()), word_embedding.shape))

train_batch_generator = BatchGenerator(vocab, tiny_data, batch_size=30, training=True)
eval_batch_generator = BatchGenerator(vocab, tiny_data, batch_size=30)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RNET(vocab, device, pretrained_word_embedding=word_embedding, word_embedding_size=100)
model.compile()
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=15, episodes=2, log_every_n_batch=10)