import sys
sys.path.append('../..')
from pprint import pprint
from pytorch_mrc.dataset.squad import SquadReader
from pytorch_mrc.data.vocabulary import Vocabulary
from pytorch_mrc.data.batch_generator import BatchGenerator


# define print function to make sure something is right
def print_info(batch_generator):
    print('***dataset keys***: {}'.format(list(batch_generator.dataset[0].keys())))
    print('***dataset sample***: \n{}'.format(batch_generator.dataset[0]))

    batch_generator.init()
    print('*****one batch data sample*****')
    batch_sample = batch_generator.next()
    # pprint(batch_sample)  # when batch_size is small you can do this.
    for k, v in batch_sample.items():
        print('{} -> shape: {}'.format(k, list(v.size())))
    print('*' * 10)


# define data folder/file
data_folder = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/'
save_folder = data_folder + 'batch_generator_data/'
tiny_file = data_folder + "smaller_tiny-v1.1.json"
vocab_file = data_folder + 'vocab_data/' + 'vocab_tiny_100d.pkl'
save_file = save_folder + 'bg_tiny_32b_100d.pkl'

# read data
reader = SquadReader()
print('reading data from {} ...'.format(tiny_file))
tiny_data = reader.read(tiny_file)

# load vocabulary
vocab = Vocabulary()
print('***loading vocabulary...***')
vocab.load(vocab_file)
word_embedding = vocab.get_word_embedding()
print('word vocab size: {}, word embedding shape: {}'.format(len(vocab.get_word_vocab()), word_embedding.shape))

# build batch generator
print('***building batch generator***')
batch_generator = BatchGenerator()
batch_generator.build(vocab, tiny_data, batch_size=32, training=False)
print_info(batch_generator)

# save batch generator
print('***saving BatchGenerator...***')
batch_generator.save(save_file)
print('successful!')

# load batch generator
print('***loading BatchGenerator***')
del batch_generator
batch_generator = BatchGenerator()
batch_generator.load(save_file)
print_info(batch_generator)

print('done!')
