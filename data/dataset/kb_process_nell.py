import os
import string

from pycorenlp import StanfordCoreNLP

SCORE_THRESHOLD = 0.9
STR_PUNCTUATION = set(string.punctuation)
STR_DIGITS = set(string.digits)


class NamedEntity(object):
    """
    Args:
        ner: Name of the entity. For example: O/MISC.
        entity_name: NELL entity name of the entity. For example: [* * * A A * *], token_seq = A_A
        from_index: Beginning index in sequence of entity. For example: [* * * A A * *], from = 3, to = 5
        to_index: End index(exclusive) in sequence of entity. For example: [* * * A A * *], from = 3, to = 5
    """

    def __init__(self, ner, entity_name, from_index, to_index):
        self.ner = ner
        self.entity_name = entity_name
        self.from_index = from_index
        self.to_index = to_index


def process():
    print('*********************************')
    print(f'NELL Processing')
    print('*********************************')

    # connecting Standford CoreNLP.
    # run the following command at the CoreNLP directory to start the CoreNLP server at first:
    # java -mx10g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9753 -timeout 20000
    nlp = StanfordCoreNLP('http://localhost:9753')

    # read valid NELL concept list.
    concept_set = set()
    for line in open('../kb/nell/nell_concept_list.txt'):
        concept_set.add(line.strip())

    # read NELL concepts from .csv file.
    # Get .csv file from the following url:
    # http://rtw.ml.cmu.edu/resources/results/08m/NELL.08m.1115.esv.csv.gz
    print('Begin to read NELL csv...')
    nell_entity_2_concept = {}
    header = True
    for line in open('../kb/nell/NELL.08m.1115.esv.csv', 'r', encoding='utf-8'):
        if header:
            header = False
            continue

        # For example:
        # root_concept_name     :entity_name            | generalizations | concept_name           | iter | confidence
        # concept:biotechcompany:aspect_medical_systems | generalizations |	concept:biotechcompany | 1103 | 0.9244426550775064
        items = line.strip().split('\t')
        if items[1] == 'generalizations' and float(items[4]) >= SCORE_THRESHOLD:
            entity_name = preprocess_nell_entity_name(items[0])
            concept_name = items[2]
            if entity_name not in nell_entity_2_concept:
                nell_entity_2_concept[entity_name] = set()
            nell_entity_2_concept[entity_name].add(concept_name)

    # tag tokens for every each dataset.
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('atis/train/seq.in'), 'atis/train/seq.entities',
        None, full=True)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('atis/dev/seq.in'), 'atis/dev/seq.entities',
        None, full=True)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('atis/test/seq.in'), 'atis/test/seq.entities',
        None, full=True)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('snips/train/seq.in'), 'snips/train/seq.entities',
        None, full=True)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('snips/dev/seq.in'), 'snips/dev/seq.entities',
        None, full=True)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('snips/test/seq.in'), 'snips/test/seq.entities',
        None, full=True)

    atis_vocab = create_vocab('atis')
    snips_vocab = create_vocab('snips')

    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('atis/train/seq.in'), 'atis/train/seq.entities',
        atis_vocab, full=False)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('atis/dev/seq.in'), 'atis/dev/seq.entities',
        atis_vocab, full=False)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('atis/test/seq.in'), 'atis/test/seq.entities',
        atis_vocab, full=False)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('snips/train/seq.in'), 'snips/train/seq.entities',
        snips_vocab, full=False)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('snips/dev/seq.in'), 'snips/dev/seq.entities',
        snips_vocab, full=False)
    tag(nlp, concept_set, nell_entity_2_concept, read_dataset('snips/test/seq.in'), 'snips/test/seq.entities',
        snips_vocab, full=False)


def preprocess_nell_entity_name(raw_name):
    """
    Remove category part of NELL entities, digit prefix 'n' and additional '_'.
    """

    entity_name = raw_name.split(':')[-1]
    if entity_name.startswith('n') and all([char in STR_DIGITS for char in entity_name.split('_')[0][1:]]):
        entity_name = entity_name[1:]

    return "_".join(
        filter(lambda x: len(x) > 0, entity_name.split('_'))
    )


def read_dataset(file):
    return [line.strip() for line in open(file, 'r', encoding='utf-8')]


def tag(nlp, concept_set, nell_entity_2_concept, dataset, f_out, vocab, full=False):
    f_out = f'{f_out}-full' if full else f'{f_out}-partial'
    print('*********************************')
    print(f'tag, output = {f_out}')
    print('*********************************')

    # NER for every sequences.
    named_entities = []  # List of NamedEntity. [num_seq, num_entity]
    for seq_index, seq in enumerate(dataset):
        tokens = seq.split()
        seq = " ".join([token.capitalize() for token in tokens])
        seq_entities = []

        print(f'{seq_index + 1}/{len(dataset)} - NER: {seq}')
        ner_result = nlp.annotate(seq, properties={
            'annotators': 'ner',
            'outputFormat': 'json'
        })

        if len(tokens) == len(ner_result['sentences'][0]['tokens']):
            # ignore those NER outputs have a mismatch token count.
            for entity in ner_result['sentences'][0]['entitymentions']:
                from_index = entity['tokenBegin']
                to_index = entity['tokenEnd']

                entity_name = tokens[from_index: to_index]
                entity_name = "_".join(filter(lambda x: x not in STR_PUNCTUATION, entity_name))

                seq_entities.append(
                    NamedEntity(entity['ner'], entity_name, from_index, to_index)
                )

        named_entities.append(seq_entities)

    # get concept list.
    dataset_concepts = []  # 3-dim list of NELL concept name. [num_seq, seq_len, num_concept]
    for (seq, seq_entities) in zip(dataset, named_entities):
        seq_concepts = [[]] * len(seq.split())
        for entity in seq_entities:
            if entity.entity_name not in nell_entity_2_concept:
                continue

            concepts = list(nell_entity_2_concept[entity.entity_name])
            for i, concept in enumerate(concepts):
                if concept not in concept_set or (not full and concept not in vocab):
                    concepts.pop(i)

            for i in range(entity.from_index, entity.to_index):
                seq_concepts[i] = concepts

        dataset_concepts.append(seq_concepts)

    # build concept dataset.
    concepts_out = []
    for seq_concepts in dataset_concepts:
        seq_concepts_out = []
        for token_concepts in seq_concepts:
            seq_concepts_out.append('+'.join(token_concepts) if len(token_concepts) != 0 else '_')
        concepts_out.append(" ".join(seq_concepts_out))

    # output.
    with open(f_out, 'w', encoding='utf-8') as f:
        for line in concepts_out:
            f.write(line + '\n')


def create_vocab(data_dir):
    """
    Args:
        data_dir: atis/snips

    Return:
        Vocabulary list.
    """

    vocab = set()
    with open(os.path.join(data_dir, "train", 'seq.entities-full'), 'r', encoding='utf-8') as f:
        for seq_concepts in f:
            for token_concepts in seq_concepts.strip().split():
                if token_concepts != '_':
                    vocab.update(token_concepts.split('+'))
    return sorted(vocab)


if __name__ == "__main__":
    process()
