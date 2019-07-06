import sys
sys.path.append('../..')
from pprint import pprint
from pytorch_mrc.dataset.squad import SquadReader
from pytorch_mrc.data.vocabulary import Vocabulary
from pytorch_mrc.data.batch_generator import BatchGenerator

data_folder = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/'
embedding_folder = '/home/len/yingzq/nlp/mrc_dataset/word_embeddings/'
tiny_file = data_folder + "smaller_tiny-v1.1.json"
embedding_file = embedding_folder + 'glove.6B.100d.txt'

reader = SquadReader()
print('reading data from {} ...'.format(tiny_file))
tiny_data = reader.read(tiny_file)

vocab = Vocabulary()
print('building vocabulary...')
vocab.build_vocab(tiny_data, min_word_count=3, min_char_count=10)
print('making word embedding...')
word_embedding = vocab.make_word_embedding(embedding_file)
print('word vocab size: {}, word embedding shape: {}'.format(len(vocab.get_word_vocab()), word_embedding.shape))

print('***building batch generator***')
batch_generator = BatchGenerator(vocab, tiny_data, batch_size=60, training=False)
print('***dataset keys***: {}'.format(list(batch_generator.dataset[0].keys())))
print('***dataset sample***: \n{}'.format(batch_generator.dataset[0]))

batch_generator.init()
print('*****one batch data sample*****')
batch_sample = batch_generator.next()
# pprint(batch_sample)  # when batch_size is small you can do this.
for k, v in batch_sample.items():
    print('{} -> shape: {}'.format(k, list(v.size())))
print('*' * 10)

print('done!')
