import os


def vocab_stats(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')
    test_dir = os.path.join(data_dir, 'test')

    # intent
    intent_vocab = set()
    with open(os.path.join(train_dir, 'label'), 'r', encoding='utf-8') as f:
        for line in f:
            intent_vocab.add(line.strip())

    with open(os.path.join(dev_dir, 'label'), 'r', encoding='utf-8') as f:
        for line in f:
            intent_vocab.add(line.strip())

    with open(os.path.join(test_dir, 'label'), 'r', encoding='utf-8') as f:
        for line in f:
            intent_vocab.add(line.strip())

    with open(os.path.join(data_dir, 'total_intent_label.txt'), 'w', encoding='utf-8') as f:
        additional_tokens = ["UNK"]
        for token in additional_tokens:
            f.write(token + '\n')

        intent_vocab = sorted(list(intent_vocab))
        for intent in intent_vocab:
            f.write(intent + '\n')

    # slot
    slot_vocab = set()
    with open(os.path.join(train_dir, 'seq.out'), 'r', encoding='utf-8') as f:
        for line in f:
            for slot in line.strip().split():
                slot_vocab.add(slot)

    with open(os.path.join(dev_dir, 'seq.out'), 'r', encoding='utf-8') as f:
        for line in f:
            for slot in line.strip().split():
                slot_vocab.add(slot)

    with open(os.path.join(test_dir, 'seq.out'), 'r', encoding='utf-8') as f:
        for line in f:
            for slot in line.strip().split():
                slot_vocab.add(slot)

    with open(os.path.join(data_dir, 'total_slot_label.txt'), 'w', encoding='utf-8') as f:
        slot_vocab = sorted(list(slot_vocab), key=lambda x: (x[2:], x[:2]))

        # Write additional tokens
        additional_tokens = ["PAD", "UNK"]
        for token in additional_tokens:
            f.write(token + '\n')

        for slot in slot_vocab:
            f.write(slot + '\n')


if __name__ == "__main__":
    vocab_stats('atis/')
    vocab_stats('snips/')
