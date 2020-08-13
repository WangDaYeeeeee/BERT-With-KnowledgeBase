import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizer


def get_intent_labels(args):
    f = os.path.join(args.data_dir, "dataset", args.task, 'intent_label.txt')
    return [label.strip() for label in open(f, 'r', encoding='utf-8')]


def get_slot_labels(args):
    f = os.path.join(args.data_dir, "dataset", args.task, 'slot_label.txt')
    return [label.strip() for label in open(f, 'r', encoding='utf-8')]


def get_wn_rdf_triples(args):
    """
    Return:
        list, contains tuples of WordNet RDF triple.
        [('00470084', '00224901', '_hypernym'), ('01160031', '01611067', '_also_see'), ...]
    """

    f = os.path.join(args.data_dir, "kb", "wn", "wn_rdf_triples.txt")
    return [tuple(line.strip().split()) for line in open(f, 'r', encoding='utf-8')]


def index_wn_rdf_triples(wn_rdf_triples, concept_ids, bidirectional=True):
    """
    Retrieve related WordNet RDF triples by synset entity id.

    Args:
        wn_rdf_triples: list, contains tuples of WordNet RDF triple.
        concept_ids: list, contains synset concept id.
        bidirectional: True if select RDF triples by 2 concept. Otherwise select RDF triples by the first concept.

    Return:
        list, contains index of WordNet RDF triples.
    """

    results = []
    for e_id in concept_ids:
        for t_index, t in enumerate(wn_rdf_triples):
            if (
                    (bidirectional and (t[0] == e_id or t[1] == e_id)) or (not bidirectional and t[0] == e_id)
            ) and t_index not in results:
                results.append(t_index)
    return results


def get_wn_rdf_dict(args):
    """
    Return:
        list, contains tuples of WordNet RDF dict.
        [(29170, 13422, 10), (34183, 18435, 5), ...]
    """

    f = os.path.join(args.data_dir, "kb", "wn", "wn_rdf_dict.txt")
    return [tuple([int(i) for i in line.strip().split()]) for line in open(f, 'r', encoding='utf-8')]


def get_wn_concept_vectors(args):
    """
    Return:
        [concept_num, dim]
    """

    f = os.path.join(args.data_dir, "kb", "wn", "wn_concept2vec.txt")
    return [[float(i) for i in line.strip().split()[1:]] for line in open(f, 'r', encoding='utf-8')]


def get_wn_concept_vector_names(args):
    """
    Return:
        [concept_num]
    """

    f = os.path.join(args.data_dir, "kb", "wn", "wn_concept2vec.txt")
    return [line.strip().split()[0] for line in open(f, 'r', encoding='utf-8')]


def get_wn_relations(args):
    f = os.path.join(args.data_dir, "kb", "wn", "wn_relation2id.txt")
    return [line.strip() for line in open(f, 'r', encoding='utf-8')]


def get_wn_concept_id2name_dict(args):
    id2name_dict = {}

    f = os.path.join(args.data_dir, "kb", "wn", "wn_definitions.txt")
    for line in open(f, 'r', encoding='utf-8'):
        line = line.strip().split()
        concept_id = line[0]
        concept_name = line[1]
        id2name_dict[concept_id] = concept_name

    return id2name_dict


def get_nell_concept_vectors(args):
    """
    Return:
        [concept_num, dim]
    """

    f = os.path.join(args.data_dir, "kb", "nell", "nell_concept2vec.txt")
    return [[float(i) for i in line.strip().split()[1:]] for line in open(f, 'r', encoding='utf-8')]


def get_nell_concept_vector_names(args):
    """
    Return:
        [concept_num]
    """

    f = os.path.join(args.data_dir, "kb", "nell", "nell_concept2vec.txt")
    return [line.strip().split()[0] for line in open(f, 'r', encoding='utf-8')]


def load_tokenizer(args):
    return BertTokenizer.from_pretrained(args.model_name_or_path)


def init_logger(args):
    if args.log_to_file:
        logging.basicConfig(filename=f'../records/{args.record_path}/log/console.log',
                            filemode='w',
                            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_embed(num_embed, embed_dim, padding_index=None):
    embed = np.empty([num_embed, embed_dim])
    scale = np.sqrt(3.0 / embed_dim)

    for index in range(num_embed):
        embed[index, :] = np.random.uniform(-scale, scale, [1, embed_dim])

    if padding_index is not None:
        embed[padding_index, :] = 0

    return embed


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    semantic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(semantic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    return {
        "intent_acc": (preds == labels).mean()
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """
    For the cases that intent and all the slots are correct (in one sentence)
    """

    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    return {
        "semantic_acc": np.multiply(intent_result, slot_result).mean()
    }
