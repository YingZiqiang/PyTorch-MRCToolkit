import sys
sys.path.append('../..')
from pytorch_mrc.data.vocabulary import Vocabulary
from pytorch_mrc.dataset.squad import SquadReader

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
print('word pad token idx: {}, embedding is: \n{}'.format(
    vocab.get_word_pad_idx(), word_embedding[vocab.get_word_pad_idx()]))
print('word unk token idx: {}, embedding is: \n{}'.format(
    vocab.get_word_unk_idx(), word_embedding[vocab.get_word_unk_idx()]))
print('word `code` idx: {}, embedding is: \n{}'.format(
    vocab.get_word_idx('code'), word_embedding[vocab.get_word_idx('code')]))
print('word `randomrandom` idx: {}, embedding is: \n{}'.format(
    vocab.get_word_idx('randomrandom'), word_embedding[vocab.get_word_idx('randomrandom')]))
print('done!')
