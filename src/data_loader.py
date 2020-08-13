import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from utils import get_intent_labels, get_slot_labels, get_wn_concept_id2name_dict, get_wn_concept_vector_names, \
    get_nell_concept_vector_names

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A training/test example for a single sequence(sentence).

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: string. The intent label of the example.
        slot_labels: list. The slot labels of the example.
        wn_synset_indexes: 2-dim list. The WordNest synset indexes of the example. [seq_len, synset_num]
        nell_entity_indexes: 2-dim list. The NELL entity indexes of the example. [seq_len, entity_num]
    """

    def __init__(self, guid, words, intent_label, slot_labels, wn_synset_indexes, nell_entity_indexes):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels
        self.wn_synset_indexes = wn_synset_indexes
        self.nell_entity_indexes = nell_entity_indexes

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        wn_synset_indexes: [seq_len, synset_num]
        wn_synset_lengths: [seq_len]
        nell_entity_indexes: [seq_len, entity_num]
        nell_entity_lengths: [seq_len]
    """

    def __init__(self, input_tokens, input_ids, attention_mask, token_type_ids, intent_label_id, slot_label_ids,
                 wn_synset_indexes, wn_synset_lengths, nell_entity_indexes, nell_entity_lengths):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_label_ids = slot_label_ids
        self.wn_synset_indexes = wn_synset_indexes
        self.wn_synset_lengths = wn_synset_lengths
        self.nell_entity_indexes = nell_entity_indexes
        self.nell_entity_lengths = nell_entity_lengths

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class ExampleLoader(object):

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)
        self.wn_concept_id2name_dict = get_wn_concept_id2name_dict(args)
        self.wn_concept_names = get_wn_concept_vector_names(args)
        self.nell_concept_names = get_nell_concept_vector_names(args)

        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq.out'
        self.wn_synsets_file = 'seq.synsets-full'  # if args.full else 'seq.synsets-partial'
        self.nell_entities_file = 'seq.entities-full'  # if args.full else 'seq.entities-partial'

    @classmethod
    def _read_file(cls, input_file):
        """
        Reads a tab separated value file.
        """

        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, wn_synsets, nell_entities, set_type):
        """
        Creates examples for the training and dev sets.

        Args:
            set_type: train, dev, test
        """

        examples = []
        for i, (text, intent, slot, wn_synset, nell_entity) in enumerate(
                zip(texts, intents, slots, wn_synsets, nell_entities)):
            logger.info(f'Creating example ({i + 1} / {len(texts)})')

            # guid.
            guid = "%s-%s" % (set_type, i)
            # words.
            words = text.split()  # Some are spaced twice
            # intent.
            intent_label = self.intent_labels.index(
                intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # slots.
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))
            # WordNet synsets.
            wn_synset_indexes = []
            for s in wn_synset.split():  # synset of a word.
                if s == '_':
                    wn_synset_indexes.append([])
                else:
                    concept_ids = s.split('+')
                    concept_names = [self.wn_concept_id2name_dict[concept_id] for concept_id in concept_ids]
                    concept_indexes = [self.wn_concept_names.index(concept_name) for concept_name in concept_names]
                    wn_synset_indexes.append(concept_indexes)
            # NELL entities.
            nell_entity_indexes = []
            for s in nell_entity.split():  # entity of a word.
                if s == '_':
                    nell_entity_indexes.append([])
                else:
                    concept_names = s.split('+')
                    concept_indexes = [self.nell_concept_names.index(concept_name) for concept_name in concept_names]
                    nell_entity_indexes.append(concept_indexes)

            assert len(words) == len(slot_labels) == len(wn_synset_indexes) == len(nell_entity_indexes)
            examples.append(
                InputExample(
                    guid=guid,
                    words=words,
                    intent_label=intent_label,
                    slot_labels=slot_labels,
                    wn_synset_indexes=wn_synset_indexes,
                    nell_entity_indexes=nell_entity_indexes
                )
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, 'dataset', self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            texts=self._read_file(os.path.join(data_path + '/', self.input_text_file)),
            intents=self._read_file(os.path.join(data_path + '/', self.intent_label_file)),
            slots=self._read_file(os.path.join(data_path + '/', self.slot_labels_file)),
            wn_synsets=self._read_file(os.path.join(data_path + '/', self.wn_synsets_file)),
            nell_entities=self._read_file(os.path.join(data_path + '/', self.nell_entities_file)),
            set_type=mode
        )


def convert_examples_to_features(examples, max_seq_len, max_wn_concepts_count, max_nell_concepts_count, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 pad_wn_concept_id=0,
                                 pad_nell_concept_id=0):
    # Setting based on the current models type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []  # [seq_num]
    for (ex_index, example) in enumerate(examples):
        logger.info(f'Converting example to feature ({ex_index + 1} / {len(examples)})')

        # Tokenize word by word.
        tokens = []  # [seq_len]
        intent_label_id = int(example.intent_label)
        slot_labels_ids = []  # [seq_len]
        wn_synset_indexes = []  # [seq_len, concepts_count]
        wn_synset_lengths = []  # [seq_len]
        nell_entity_indexes = []  # [seq_len, concepts_count]
        nell_entity_lengths = []  # [seq_len]

        for word, slot_label, wn_synset_index, nell_entity_index in zip(
                example.words, example.slot_labels, example.wn_synset_indexes, example.nell_entity_indexes):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))
            # Use the real concept index for the first token of the word
            wn_synset_indexes.extend([wn_synset_index] + [[]] * (len(word_tokens) - 1))
            wn_synset_lengths.extend([len(wn_synset_index)] + [0] * (len(word_tokens) - 1))
            nell_entity_indexes.extend([nell_entity_index] + [[]] * (len(word_tokens) - 1))
            nell_entity_lengths.extend([len(nell_entity_index)] + [0] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]
            wn_synset_indexes = wn_synset_indexes[:(max_seq_len - special_tokens_count)]
            wn_synset_lengths = wn_synset_lengths[:(max_seq_len - special_tokens_count)]
            nell_entity_indexes = nell_entity_indexes[:(max_seq_len - special_tokens_count)]
            nell_entity_lengths = nell_entity_lengths[:(max_seq_len - special_tokens_count)]

        # Add [CLS] & [SEP] token
        tokens = [cls_token] + tokens + [sep_token]
        token_type_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids + [pad_token_label_id]
        wn_synset_indexes = [[]] + wn_synset_indexes + [[]]
        wn_synset_lengths = [0] + wn_synset_lengths + [0]
        nell_entity_indexes = [[]] + nell_entity_indexes + [[]]
        nell_entity_lengths = [0] + nell_entity_lengths + [0]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens.
        # Only real tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + [pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        token_type_ids = token_type_ids + [pad_token_segment_id] * padding_length
        slot_labels_ids = slot_labels_ids + [pad_token_label_id] * padding_length
        wn_synset_indexes = wn_synset_indexes + [[]] * padding_length
        wn_synset_lengths = wn_synset_lengths + [0] * padding_length
        nell_entity_indexes = nell_entity_indexes + [[]] * padding_length
        nell_entity_lengths = nell_entity_lengths + [0] * padding_length

        # Zero-pad up to the max concept count.
        for token_index, (token_synset_indexes, token_entity_indexes) in enumerate(
                zip(wn_synset_indexes, nell_entity_indexes)):
            # Maximum concept count for each word.
            if len(token_synset_indexes) > max_wn_concepts_count:
                wn_synset_indexes[token_index] = token_synset_indexes[:max_wn_concepts_count]
                token_synset_indexes = wn_synset_indexes[token_index]
            if len(token_entity_indexes) > max_nell_concepts_count:
                nell_entity_indexes[token_index] = token_entity_indexes[:max_nell_concepts_count]
                token_entity_indexes = nell_entity_indexes[token_index]
            # Pad up.
            padding_length = max_wn_concepts_count - len(token_synset_indexes)
            token_synset_indexes.extend([pad_wn_concept_id] * padding_length)
            padding_length = max_nell_concepts_count - len(token_entity_indexes)
            token_entity_indexes.extend([pad_nell_concept_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length"
        assert len(attention_mask) == max_seq_len, "Error with attention mask length"
        assert len(token_type_ids) == max_seq_len, "Error with token type length"
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length"
        assert len(wn_synset_indexes) == max_seq_len, "Error with WordNet synset indexes length"
        assert len(wn_synset_lengths) == max_seq_len, "Error with WordNet synset lengths length"
        assert len(nell_entity_indexes) == max_seq_len, "Error with NELL entity indexes length"
        assert len(nell_entity_lengths) == max_seq_len, "Error with NELL entity lengths length"

        features.append(
            InputFeatures(
                input_tokens=tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                intent_label_id=intent_label_id,
                slot_label_ids=slot_labels_ids,
                wn_synset_indexes=wn_synset_indexes,
                wn_synset_lengths=wn_synset_lengths,
                nell_entity_indexes=nell_entity_indexes,
                nell_entity_lengths=nell_entity_lengths
            )
        )

    return features


def load_and_cache_dataset(args, tokenizer, mode):
    """
    Args:
        args:
        tokenizer: Bert tokenizer.
        mode: train/dev/test.
    """

    dataset_cache = os.path.join(
        args.data_dir + '/',
        'cache/'
        'cached_{}_{}_{}_{}_{}_{}'.format(
            args.task,  # atis, snips
            mode,  # dev, test, train
            list(filter(None, args.model_name_or_path.split("/"))).pop(),  # bert-base-uncased
            f'seq{args.max_seq_len}',
            f'kb{args.max_wn_concepts_count}+{args.max_nell_concepts_count}',
            'full'  # if args.full else 'partial'
        )
    )

    if os.path.exists(dataset_cache):
        # Load features from cache.
        logger.info("Loading features from cache: %s", dataset_cache)
        features = torch.load(dataset_cache)
    else:
        # Load examples from file, convert them to features and cache all features.
        logger.info("Creating features from file at %s", args.data_dir)
        examples = ExampleLoader(args).get_examples(mode)

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        features = convert_examples_to_features(examples,
                                                args.max_seq_len,
                                                args.max_wn_concepts_count,
                                                args.max_nell_concepts_count,
                                                tokenizer,
                                                pad_token_label_id=args.ignore_index)
        logger.info("Saving features into cached file %s", dataset_cache)
        torch.save(features, dataset_cache)

    # [num_examples, 9, ]
    return TensorDataset(
        # [CLS] [id_1, id_2] ... [id_t] [SEP] [PAD] [PAD] ... [PAD]
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        # 1     1      1     ... 1      1     0     0     ... 0
        torch.tensor([f.attention_mask for f in features], dtype=torch.long),
        # 0     0      0     ... 0      0     0     0     ... 0
        torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
        # intent_id
        torch.tensor([f.intent_label_id for f in features], dtype=torch.long),
        # -100  s_1    -100  ... s_t    -100  -100  -100  ... -100
        torch.tensor([f.slot_label_ids for f in features], dtype=torch.long),
        # []    [...]  []    ... [...]  []    []    []    ... []
        torch.tensor([f.wn_synset_indexes for f in features], dtype=torch.long),
        # 0     len    0     ... len    0     0     0     ... 0
        torch.tensor([f.wn_synset_lengths for f in features], dtype=torch.long),
        # []    [...]  []    ... [...]  []    []    []    ... []
        torch.tensor([f.nell_entity_indexes for f in features], dtype=torch.long),
        # 0     len    0     ... len    0     0     0     ... 0
        torch.tensor([f.nell_entity_lengths for f in features], dtype=torch.long)
    )
