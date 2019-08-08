from pytorch_mrc.data.vocabulary import Vocabulary
from pytorch_mrc.dataset.squad import SquadReader


# define print function to make sure something is right
def print_info(vocab, word_embedding):
    print('word vocab size: {}, word embedding shape: {}'.format(len(vocab.get_word_vocab()), word_embedding.shape))
    print('word pad token idx: {}, embedding is: \n{}'.format(
        vocab.get_word_pad_idx(), word_embedding[vocab.get_word_pad_idx()]))
    print('word unk token idx: {}, embedding is: \n{}'.format(
        vocab.get_word_unk_idx(), word_embedding[vocab.get_word_unk_idx()]))
    print('word `code` idx: {}, embedding is: \n{}'.format(
        vocab.get_word_idx('code'), word_embedding[vocab.get_word_idx('code')]))
    print('word `randomrandom` idx: {}, embedding is: \n{}'.format(
        vocab.get_word_idx('randomrandom'), word_embedding[vocab.get_word_idx('randomrandom')]))


# define data path
tiny_file = "/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/tiny-v1.1.json"
embedding_file = '/home/len/yingzq/nlp/mrc_dataset/word_embeddings/glove.6B.100d.txt'
vocab_save_file = '/home/len/yingzq/nlp/mrc_dataset/squad-v1.1/vocab_data/vocab_tiny_100d.pkl'  # where to save vocab data

# read data
reader = SquadReader()
print('reading data from {} ...'.format(tiny_file))
tiny_data = reader.read(tiny_file)

# build the vocabulary
vocab = Vocabulary()
print('building vocabulary...')
vocab.build_vocab(tiny_data, min_word_count=3, min_char_count=10)
print('making word embedding...')
vocab.make_word_embedding(embedding_file)
word_embedding = vocab.get_word_embedding()
print_info(vocab, word_embedding)

# save vocabulary
print('***saveing vocabulary...***')
vocab.save(vocab_save_file)
print('successful!')

# load vocabulary
print('***loading vocabulary...***')
vocab = Vocabulary()
vocab.load(vocab_save_file)
word_embedding = vocab.get_word_embedding()
print_info(vocab, word_embedding)

print('done!')
