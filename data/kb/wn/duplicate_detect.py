import os


def detect():
    lines = []
    with open(os.path.join('wn_rdf_triples.txt'), 'r', encoding='utf-8') as f:
        for i, new in enumerate(f):
            new = new.strip()
            print(f'read line = {i}')
            exist = False
            for old in lines:
                if new == old:
                    exist = True
                    break
            if not exist:
                lines.append(new)
        print(f'valid = {len(lines)}')


if __name__ == "__main__":
    detect()
