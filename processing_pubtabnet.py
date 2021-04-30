import os
import json
from PIL import Image
from tqdm import tqdm


def preprocess_json(dataset_dir, output_dir):
    json_path = os.path.join(dataset_dir, 'PubTabNet_2.0.0.jsonl')

    dicts = {split: [] for split in ['train', 'val', 'test']}
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=509892):
            data = json.loads(line)
            
            data_new = dict()
            image_path = os.path.join(dataset_dir, data['split'], data['filename'])
            image = Image.open(image_path)
            data_new['image_path'] = image_path
            data_new['image_size'] =  image.size
            # start text
            nd = len([x for x in data['html']['structure']['tokens'] if x == '</td>']) 
            nc = len(data['html']['cells'])
            alert_msg = "The number of td (%d) is note equal to the number of cells (%d)." % (nd, nc)
            assert nd == nc, alert_msg

            data_new['text'] = ['<START>'] 
            cnt_cell = 0
            for struct_tok in data['html']['structure']['tokens']:
                if struct_tok == '</td>':
                    cell = data['html']['cells'][cnt_cell]
                    cnt_cell += 1
                    data_new['text'] += cell['tokens']
                data_new['text'].append(struct_tok)
            # end text
            data_new['num_tokens'] = len(data_new['text'])
            dicts[data['split']].append(data_new)

    for k, v in dicts.items():
        output_path = os.path.join(output_dir, k + ".json")
        with open(output_path, 'w', encoding='utf-8') as out:
            json.dump(v, out)


def generate_vocab(dataset_dir, output_dir):
    json_path = os.path.join(dataset_dir, "PubTabNet_2.0.0.jsonl")
    tokens = {key: set() for key in ['structure', 'cell']}
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=509892):
            data = json.loads(line)
            tokens['structure'].update(data['html']['structure']['tokens'])
            for cell in data['html']['cells']:
                tokens['cell'].update(cell['tokens'])

    print('\nsize of structure_tokens: ', len(tokens['structure']))
    print('size of cell_tokens: ', len(tokens['cell']))

    tokens['cell'] = tokens['cell'].difference(tokens['structure'])
    tokens_total = []
    for key, value in tokens.items():
        tokens_total.extend(sorted(list(value)))

    vocab_path = os.path.join(output_dir, 'vocab.txt')

    with open(vocab_path, 'w', encoding='utf-8') as out:
        out.write('<PAD>\n<START>\n<END>\n<CLS>\n')
        for token in tokens_total:
            out.write(token + '\n')


if __name__ == '__main__':
    dataset_dir = '/data/private/datasets/pubtabnet'
    output_dir = '/data/private/datasets/pubtabnet/annotations'
    generate_vocab(dataset_dir, output_dir)
    preprocess_json(dataset_dir, output_dir)
