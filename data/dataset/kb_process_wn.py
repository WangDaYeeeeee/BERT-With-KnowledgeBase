import os
import string

import nltk
from nltk.corpus import wordnet as wn


def process():
    print('*********************************')
    print(f'WordNet Processing')
    print('*********************************')

    # [concept_num]
    concept_ids = [line.strip().split()[0] for line in open('../kb/wn/wn_definitions.txt', 'r', encoding='utf-8')]

    stopwords = set(nltk.corpus.stopwords.words('english'))
    atis_vocab = create_vocab('atis')
    snips_vocab = create_vocab('snips')

    synset_files_process('atis', 'train', concept_ids, stopwords, atis_vocab, full=False),
    synset_files_process('atis', 'dev', concept_ids, stopwords, atis_vocab, full=False),
    synset_files_process('atis', 'test', concept_ids, stopwords, atis_vocab, full=False),
    synset_files_process('snips', 'train', concept_ids, stopwords, atis_vocab, full=False),
    synset_files_process('snips', 'dev', concept_ids, stopwords, snips_vocab, full=False),
    synset_files_process('snips', 'test', concept_ids, stopwords, snips_vocab, full=False)

    synset_files_process('atis', 'train', concept_ids, stopwords, atis_vocab, full=True),
    synset_files_process('atis', 'dev', concept_ids, stopwords, atis_vocab, full=True),
    synset_files_process('atis', 'test', concept_ids, stopwords, atis_vocab, full=True),
    synset_files_process('snips', 'train', concept_ids, stopwords, atis_vocab, full=True),
    synset_files_process('snips', 'dev', concept_ids, stopwords, snips_vocab, full=True),
    synset_files_process('snips', 'test', concept_ids, stopwords, snips_vocab, full=True)


def create_vocab(data_dir):
    """
    Args:
        data_dir: atis/snips

    Return:
        Vocabulary list.
    """

    vocab = set()
    with open(os.path.join(data_dir, "train", 'seq.in'), 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                if word.endswith(':'):
                    word = word[: -1]
                elif word.endswith("'s"):
                    word = word[: -2]
                vocab.add(word)
    return sorted(vocab)


def synset_files_process(data_dir, mode, concept_ids, stopwords, vocab, full=False):
    """
    Args:
        data_dir: atis/snips
        mode: train/test/dev
        concept_ids: list, id strings for WordNet concepts. [concept_num]
        stopwords: Those words which don't have synsets.
        vocab: Optional. Vocabulary for valid word.
        full: Optional.
    """

    print('*********************************')
    print(f'wn_synset_files_process, target = {data_dir}, {mode}')
    print('*********************************')

    f_r = os.path.join(data_dir, mode, 'seq.in')
    f_w = os.path.join(data_dir, mode, 'seq.synsets-full' if full else 'seq.synsets-partial')
    with open(f_r, 'r', encoding='utf-8') as f_r, open(f_w, 'w', encoding='utf-8') as f_w:
        total_ids = []  # [seq_num], content in single line: 'id1+id2+id3 id4+id5 _ id6' ('_' == none)

        for row, line in enumerate(f_r):
            line = line.strip()
            print(f'{row}  {line}')

            words = line.split()
            line_ids = []  # entity id for a line.
            for word in words:
                if word.endswith(':'):
                    word = word[: -1]
                elif word.endswith("'s"):
                    word = word[: -2]
                word_ids = []
                if (mode == 'train' or full or word not in vocab) \
                        and word not in set(string.punctuation) and word not in stopwords:
                    for synset in wn.synsets(word):
                        offset_str = str(synset.offset()).zfill(8)
                        if offset_str in concept_ids and offset_str not in word_ids:
                            word_ids.append(offset_str)

                if len(word_ids) == 0:
                    word_ids = '_'
                else:
                    word_ids = '+'.join(word_ids)

                line_ids.append(word_ids)

            total_ids.append(' '.join(line_ids))

        for line in total_ids:
            f_w.write(line + '\n')


if __name__ == "__main__":
    process()
