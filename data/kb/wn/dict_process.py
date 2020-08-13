import os


def create_dict_file():
    f = os.path.join("wn_rdf_triples.txt")
    # [triple_num, (id, id, relation)]
    rdf_triples = [tuple(line.strip().split()) for line in open(f, 'r', encoding='utf-8')]

    f = os.path.join("wn_concept2vec.txt")
    concepts = [line.strip().split()[0] for line in open(f, 'r', encoding='utf-8')]  # [concept_num]

    f = os.path.join("wn_definitions.txt")
    # [triple_num, (id, name)]
    concept_id2names = [tuple(line.strip().split()[:2]) for line in open(f, 'r', encoding='utf-8')]

    f = os.path.join("wn_relation2id.txt")
    relations = [line.strip() for line in open(f, 'r', encoding='utf-8')]  # [relation_num]

    dicts = []
    for i, t in enumerate(rdf_triples):
        dicts.append(
            (
                str(index_concept(concepts, get_concept_name(concept_id2names, t[0]))),
                str(index_concept(concepts, get_concept_name(concept_id2names, t[1]))),
                str(index_relation(relations, t[2]))
            )
        )
        print(f'{i + 1} / {len(rdf_triples)} : {t}, {dicts[-1]}')

    with open(os.path.join("wn_rdf_dict.txt"), 'w', encoding='utf-8') as f:
        for d in dicts:
            f.write(' '.join(d) + '\n')
            print(f'Writing: {d}')


def get_concept_name(concept_id2names, concept_id):
    for item in concept_id2names:
        if item[0] == concept_id:
            return item[1]
    raise Exception("There is no concept has a id = {}".format(concept_id))


def index_concept(concepts, concept_name):
    for i, concept in enumerate(concepts):
        if concept == concept_name:
            return i
    raise Exception("There is no concept has a name = {}".format(concept_name))


def index_relation(relations, r):
    for i, relation in enumerate(relations):
        if relation == r:
            return i
    raise Exception("There is no relation has a name = {}".format(r))


if __name__ == "__main__":
    create_dict_file()
