from data_util import Dataset, create_vocabulary, get_processing_word
from logging_cfg import setup_logging
import logging


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    processing_word = get_processing_word(lowercase=True)
    dataset = Dataset('./data/test.txt', processing_word=processing_word)
    create_vocabulary([dataset], './data/words.txt', './data/tags.txt')


if __name__ == '__main__':
    main()
