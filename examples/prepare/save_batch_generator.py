import sys
sys.path.append('../..')
import logging
from pytorch_mrc.data.batch_generator import BatchGenerator
from pytorch_mrc.dataset.squad import SquadReader
from pytorch_mrc.data.vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# define some variable
EMB_DIM = 300
BATCH_SIZE = 50
FINE_GRAINED = True
DO_LOWERCASE = True

# define data path
train_file = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/train-v1.1.json'
dev_file = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/dev-v1.1.json'
embedding_file = '/home/len/yingzq/nlp/mrc_dataset/word_embeddings/glove.840B.300d.txt'

# the path to save file
vocab_file = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/vocab_data/vocab_{}d_{}.pkl'.format(EMB_DIM, 'cased' if DO_LOWERCASE else 'uncased')
bg_train_file = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/bg_data/bg_train_{}b_{}d_{}.pkl'.format(
    BATCH_SIZE, EMB_DIM, 'cased' if DO_LOWERCASE else 'uncased')
bg_eval_file = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/bg_data/bg_eval_{}b_{}d_{}.pkl'.format(
    BATCH_SIZE, EMB_DIM, 'cased' if DO_LOWERCASE else 'uncased')

# read data
reader = SquadReader(fine_grained=FINE_GRAINED)
train_data = reader.read(train_file)
eval_data = reader.read(dev_file)

# build vocab and embedding
vocab = Vocabulary(do_lowercase=DO_LOWERCASE)
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
vocab.make_word_embedding(embedding_file)
vocab.save(vocab_file)

logging.info("building train batch generator...")
train_batch_generator = BatchGenerator()
train_batch_generator.build(vocab, train_data, batch_size=BATCH_SIZE, shuffle=True)

logging.info("building eval batch generator...")
eval_batch_generator = BatchGenerator()
eval_batch_generator.build(vocab, eval_data, batch_size=BATCH_SIZE)

train_batch_generator.save(bg_train_file)
eval_batch_generator.save(bg_eval_file)

print('done!')
