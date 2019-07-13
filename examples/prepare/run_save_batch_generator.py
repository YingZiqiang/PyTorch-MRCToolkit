import sys
sys.path.append('../..')
import logging
from pytorch_mrc.data.vocabulary import Vocabulary
from pytorch_mrc.dataset.squad import SquadReader
from pytorch_mrc.data.batch_generator import BatchGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

data_folder = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/'
embedding_folder = '/home/len/yingzq/nlp/mrc_dataset/word_embeddings/'
train_file = data_folder + "train-v1.1.json"
dev_file = data_folder + "dev-v1.1.json"
embedding_file = embedding_folder + 'glove.6B.100d.txt'
vocab_file = data_folder + 'vocab_data/' + 'vocab_100d.pkl'
bg_train_file = data_folder + 'batch_generator_data/' + 'bg_train_50b_100d.pkl'
bg_eval_file = data_folder + 'batch_generator_data/' + 'bg_eval_50b_100d.pkl'

reader = SquadReader()
train_data = reader.read(train_file)
eval_data = reader.read(dev_file)

vocab = Vocabulary()
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
vocab.make_word_embedding(embedding_file)
vocab.save(vocab_file)

logging.info("building train batch generator...")
train_batch_generator = BatchGenerator()
train_batch_generator.build(vocab, train_data, batch_size=50, training=True)
logging.info("building eval batch generator...")
eval_batch_generator = BatchGenerator()
eval_batch_generator.build(vocab, eval_data, batch_size=50)

train_batch_generator.save(bg_train_file)
eval_batch_generator.save(bg_eval_file)

print('done!')
