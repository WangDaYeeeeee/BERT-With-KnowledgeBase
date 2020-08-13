import os


def kb_stats(data_dir, mode, full=False):
    num_seq = 0
    num_token = 0
    total_wn_concept = 0
    total_nell_concept = 0
    total_concept = 0
    max_wn_concept = 0
    max_nell_concept = 0
    max_concept = 0

    f_wn = os.path.join(data_dir, mode, 'seq.synsets-full' if full else 'seq.synsets-partial')
    f_nell = os.path.join(data_dir, mode, 'seq.entities-full' if full else 'seq.entities-partial')

    with open(f_wn, 'r', encoding='utf-8') as f_wn, open(f_nell, 'r', encoding='utf-8') as f_nell:
        for (seq_synsets, seq_entities) in zip(f_wn, f_nell):
            num_seq += 1
            token_synsets = seq_synsets.strip().split()
            token_entities = seq_entities.strip().split()
            for (synset, entity) in zip(token_synsets, token_entities):
                num_token += 1
                concept_count = 0
                if synset != '_':
                    synset_count = len(synset.split('+'))
                    concept_count += synset_count
                    total_concept += synset_count
                    total_wn_concept += synset_count
                    if max_wn_concept < synset_count:
                        max_wn_concept = synset_count
                if entity != '_':
                    entity_count = len(entity.split('+'))
                    concept_count += entity_count
                    total_concept += entity_count
                    total_nell_concept += entity_count
                    if max_nell_concept < entity_count:
                        max_nell_concept = entity_count

                if max_concept < concept_count:
                    max_concept = concept_count

    print(f'*********************************')
    print(f'kb_stats, {data_dir}, {mode}')
    print('')
    print(f'max wn concept = {max_wn_concept}')
    print(f'ave wn concept for each seq = {total_wn_concept / num_seq}')
    print(f'ave wn concept for each token = {total_wn_concept / num_token}')
    print('')
    print(f'max nell concept = {max_nell_concept}')
    print(f'ave nell concept for each seq = {total_nell_concept / num_seq}')
    print(f'ave nell concept for each token = {total_nell_concept / num_token}')
    print('')
    print(f'max concept = {max_concept}')
    print(f'ave concept for each seq = {total_concept / num_seq}')
    print(f'ave concept for each token = {total_concept / num_token}')
    print(f'*********************************')


if __name__ == "__main__":
    full = True
    kb_stats('atis', 'train', full=full)
    kb_stats('atis', 'dev', full=full)
    kb_stats('atis', 'test', full=full)
    kb_stats('snips', 'train', full=full)
    kb_stats('snips', 'dev', full=full)
    kb_stats('snips', 'test', full=full)
