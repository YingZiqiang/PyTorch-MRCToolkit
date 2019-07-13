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
train_file = data_folder + "train-v1.1.json"
dev_file = data_folder + "dev-v1.1.json"
embedding_file = embedding_folder + 'glove.840B.300d.txt'

reader = SquadReader()
train_data = reader.read(train_file)
eval_data = reader.read(dev_file)
evaluator = SquadEvaluator(dev_file)

vocab = Vocabulary()
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding(embedding_file)

train_batch_generator = BatchGenerator(vocab, train_data, batch_size=32, training=True)
eval_batch_generator = BatchGenerator(vocab, eval_data, batch_size=32)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RNET(vocab, device, pretrained_word_embedding=word_embedding)
model.compile()
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=20, episodes=2)
