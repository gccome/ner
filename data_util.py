from tensorflow import gfile
import logging
from collections import Counter
import pyprind


# shared global variables to be imported from model also
UNK = "_UNK"
NUM = "_NUM"
NONE = "O"


class Dataset(object):
    """
    Class that iterates over Dataset

        __iter__ method yields a tuple (words, tags)
            words: list of raw words
            tags: list of raw tags

        If processing_word and processing_tag are not None, optional preprocessing is applied

        Example:
            ```python
            data = Dataset(filename)
            for sentence, tags in data:
                pass
            ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None):
        """
        :param filename: path to the file
        :param processing_word: (optional) function that takes a word as input
        :param processing_tag: (optional) function that takes a tag as input
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.length = None

    def __iter__(self):
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith('-DOCSTART-'):
                    if len(words) != 0:
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words.append(word)
                    tags.append(tag)
            # yield last sentence
            yield words, tags

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length


def get_processing_word(vocab_words=None, vocab_chars=None, lowercase=False, use_chars=False):
    """
    Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.
    :param vocab_words: dict[word] = idx
    :param vocab_chars: dict[char] = idx
    :param lowercase:
    :param use_chars: if use char embedding
    :return: f("cat") = ([12, 4, 32], 12345)
                      = (list of char ids, word id)
    """
    def f(word):
        # 0. get chars ids of words
        if vocab_chars is not None and use_chars is True:
            char_ids = []
            for char in word:
                # ignore chars out of vocab
                if char in vocab_chars:
                    char_ids.append(vocab_chars[char])

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of words
        if vocab_words is not None:
            word = vocab_words.get(word, vocab_words[UNK])

        if vocab_chars is not None and use_chars is True:
            return char_ids, word
        else:
            return word

    return f


def create_vocabulary(datasets, vocab_words_path, tags_path, max_vocab_words_size=None):
    """
    Create vocabulary (e.g. word vocab, char vocab, tag vocab) from an iterable of datasets objects.
    Write into files.
    :param datasets: list. a list of dataset objects
    :param vocab_words_path:
    :param tags_path:
    :param max_vocab_words_size: int, limit on the size of the created vocabulary
    :return:
    """
    logger = logging.getLogger(__name__)

    if not gfile.Exists(vocab_words_path) or not gfile.GFile(tags_path):
        logger.info("Creating words and tags vocabulary.")
        word_counts = Counter()
        tags_counts = Counter()
        for dataset in datasets:
            # pbar = pyprind.ProgBar(len(dataset), title='Counting words and tags occurrences')
            for words, tags in dataset:
                # pbar.update()
                word_counts.update(words)
                tags_counts.update(tags)
        words_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        print(words_vocab)
        tags_vocab = sorted(tags_counts, key=tags_counts.get, reverse=True)

        # add NUM if not in vocab
        if NUM not in words_vocab:
            words_vocab = [NUM] + words_vocab
        # add UNK
        words_vocab = [UNK] + words_vocab

        if max_vocab_words_size is not None:
            words_vocab = words_vocab[:max_vocab_words_size]

        write_vocab(words_vocab, vocab_words_path)
        write_vocab(tags_vocab, tags_path)


def write_vocab(vocab, filename):
    """
    Write vocab to a file. Writes one word per line.
    :param vocab: iterable that yields word
    :param filename: path to vocab file
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info('Writing vocab...')
    with gfile.GFile(filename, mode='wb') as f:
        for word in vocab:
            f.write(word + '\n')
        logger.info("- done. {} tokens".format(len(vocab)))


if __name__ == '__main__':
    processing_word = get_processing_word(lowercase=True)
    dataset = Dataset('./data/test.txt', processing_word=processing_word)
    create_vocabulary([dataset], './data/words.txt', './data/tags.txt')



