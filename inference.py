import os
import math
import glob
import json
import yaml
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import utils
import commons
from models import TableRecognizer, get_positional_encoding
import beam_search

class HParams(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = HParams(value)
        return value

class ImageLoader(torch.utils.data.Dataset):
    """
    Load image
    """
    def __init__(self, dir_path, cfg, n_tot, n_idx):
        self.cfg = cfg
        self.n_tot = n_tot
        self.n_idx = n_idx
        self.image_paths, self.lengths, \
        self.image_heights, self.image_widths = self._build(dir_path)
        self.vocab = self._load_vocab()

    def _build(self, dir_path):
        image_paths = glob.glob(os.path.join(dir_path, "*.png"))

        tuple_list = []
        for image_path in image_paths:
            image = Image.open(image_path)
            w, h = [math.ceil(x / self.cfg.patch_length) for x in image.size]
            tuple_list.append((image_path, h*w, h, w))
        tuple_list.sort(key=lambda x: x[1], reverse=True)
        
        image_paths = []
        lengths = []
        image_heights = []
        image_widths = []
        for image_path, length, h, w in tuple_list[self.n_idx::self.n_tot]:
            image_paths.append(image_path)
            lengths.append(length)
            image_heights.append(h)
            image_widths.append(w)
        return image_paths, lengths, image_heights, image_widths

    def _load_vocab(self):
        with open(self.cfg.vocab_path) as f:
            words = [x.replace('\n', '') for x in f.readlines()]
        vocab = {word: idx for idx, word in enumerate(words)}
        return vocab

    def get_items(self, index):
        patch_length = self.cfg.patch_length
        h, w = self.image_heights[index], self.image_widths[index]
        c = 3

        image = Image.open(self.image_paths[index]).convert('RGB')
        image = (np.asarray(image, dtype=np.float32) / 255) * 2 - 1
        image = torch.from_numpy(image)
        image = torch.nn.functional.pad(image, [
            0, 0,
            0, (patch_length - (image.shape[1] % patch_length)) % patch_length,
            0, (patch_length - (image.shape[0] % patch_length)) % patch_length
        ])
        image = image.view([h, patch_length, w, patch_length, c])
        image = image.permute(0, 2, 4, 1, 3)
        image = image.reshape(h * w, c * (patch_length ** 2))

        length = self.lengths[index]
        image_height = h
        image_width = w
        return (image, length, image_height, image_width)

    def __getitem__(self, index):
        return self.get_items(index)

    def __len__(self):
        return len(self.image_paths)
    

class ImageCollate():
    """ Zero-pads model inputs
    """
    def __call__(self, batch):
        """Collate's training batch from image and text info
        Inputs:
        - batch: [img, t_tot, h_img, w_img]

        Outputs:
        - (img_padded, mask_img, pos_r, pos_c)
        """
        max_len = max(x[1] for x in batch)
        b = len(batch)
        c = batch[0][0].size(1) # image patch size

        img_padded = torch.FloatTensor(b, max_len, c)
        mask_img   = torch.FloatTensor(b, max_len, 1)
        pos_r      = torch.FloatTensor(b, max_len, 1)
        pos_c      = torch.FloatTensor(b, max_len, 1)
        
        img_padded.zero_()
        mask_img.zero_()
        pos_r.zero_()
        pos_c.zero_()
        for i in range(b):
            img, t_tot, h_img, w_img = batch[i]

            img_padded[i, :t_tot] = img
            mask_img[i, :t_tot] = 1
            pos_r[i, :t_tot] = torch.arange(h_img).unsqueeze(-1).repeat(1, w_img).view(-1, 1)
            pos_c[i, :t_tot] = torch.arange(w_img).repeat(h_img).view(-1, 1)
        return img_padded, mask_img, pos_r, pos_c

def load_checkpoints(dir_path, model, regex="model_*.pth", n=1):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    
    for i, fname in enumerate(f_list):
        idx_last = i
        if fname.find("_919000.pth") != -1:
            break
    f_list = f_list[:idx_last+1]
    
    f_list = f_list[-n:]
    
    saved_state_dict = {}
    for i, checkpoint_path in enumerate(f_list):
        print(checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        for k, v in checkpoint_dict['model'].items():
            if i == 0:
                saved_state_dict[k] = v / len(f_list)
            else:
                saved_state_dict[k] += v / len(f_list)
                
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict= {}
    for k, v in state_dict.items():
        new_state_dict[k] = saved_state_dict[k]
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

def inference(self, x_img, mask_img, pos_r, pos_c, idx_start=1, idx_end=2, max_decode_len=10000, beam_size=1, top_beams=1, alpha=1., n_toks=5000):
    from tqdm import tqdm
    with torch.no_grad():
        b = x_img.size(0)
        nh = self.n_heads
        d = self.hidden_channels // self.n_heads
        dtype = x_img.dtype
        device = x_img.device

        x_emb_img = self.emb_img(x_img, mask_img, pos_r, pos_c)
        cache = [{
            "kv": [],
            "k_cum": []
            } for _ in range(self.n_layers)
        ]
        n_split = max(n_toks // x_img.size(1), 1)
        n_iter = math.ceil(b / n_split)
        for i in range(n_iter):
            print("%05d" % i, end='\r')
            x_emb_img_iter = x_emb_img[i*n_split:(i+1)*n_split]
            mask_img_iter = mask_img[i*n_split:(i+1)*n_split]
            b_iter = x_emb_img_iter.size(0)
            
            cache_each = [{
                "kv": torch.zeros(b_iter, 1, nh, d, d).to(dtype=torch.float, device=device),
                "k_cum": torch.zeros(b_iter, 1, nh, d).to(dtype=torch.float, device=device)
                } for _ in range(self.n_layers)
            ]
            _ = self.enc(x_emb_img_iter, mask_img_iter, cache_each)
            for l in range(self.n_layers):
                cache[l]["kv"].append(cache_each[l]["kv"].clone())
                cache[l]["k_cum"].append(cache_each[l]["k_cum"].clone())
        for l in range(self.n_layers):
            cache[l]["kv"] = torch.cat(cache[l]["kv"], 0)
            cache[l]["k_cum"] = torch.cat(cache[l]["k_cum"], 0)

        pos_enc = get_positional_encoding(
            torch.arange(max_decode_len).view(1,-1,1).to(device=device), 
            self.hidden_channels
        )
        
        if beam_size == 1:
            finished = torch.BoolTensor(b,1).to(device=device).fill_(False)
            idx = torch.zeros(b,1).long().to(device=device) + idx_start
            ids = []
            for i in tqdm(range(max_decode_len)):
                x_emb_txt = self.emb_txt.emb(idx) + pos_enc[:,i:i+1]
                x = self.enc(x_emb_txt, None, cache)
                logit_txt = self.proj_txt(x)
                idx = torch.argmax(logit_txt, -1)
                ids.append(idx)
                finished |= torch.eq(idx, idx_end)
                if torch.all(finished):
                    break
            return ids
        else:
            def symbols_to_logits_fn(ids, i, cache):
                x_emb_txt = self.emb_txt.emb(ids[:,i:i+1]) + pos_enc[:,i:i+1]
                x = self.enc(x_emb_txt, None, cache)
                logit_txt = self.proj_txt(x)
                return logit_txt, cache
            initial_ids = torch.zeros(b).long().to(device=device) + idx_start
            decoded_ids, scores = beam_search.beam_search(
                symbols_to_logits_fn,
                initial_ids,
                beam_size,
                max_decode_len,
                self.n_vocab,
                alpha,
                states=cache,
                eos_id=idx_end,
                stop_early=(top_beams == 1))

            if top_beams == 1:
                decoded_ids = decoded_ids[:, 0, 1:]
                scores = scores[:, 0]
            else:
                decoded_ids = decoded_ids[:, :top_beams, 1:]
                scores = scores[:, :top_beams]
            return decoded_ids, scores


if __name__ == "__main__":
    # python inference.py -m "./outputs/base/" -i "/data/private/datasets/pubtabnet/val/" -o "./results/val1" -nt 16 -ni 0 -na 20
    # ...
    # python inference.py -m "./outputs/base/" -i "/data/private/datasets/pubtabnet/val/" -o "./results/val1" -nt 16 -ni 15 -na 20
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-m", type=str, help="model directory")
    parser.add_argument("--image_dir", "-i", type=str, help="image directory")
    parser.add_argument("--out_dir", "-o", type=str, help="output directory")
    parser.add_argument("--n_tot", "-nt", type=int, help="total number of processes")
    parser.add_argument("--n_idx", "-ni", type=int, help="index of current process")
    parser.add_argument("--n_avg", "-na", type=int, default=20, help="number of checkpoints to be averaged")
    args = parser.parse_args()


    with open(os.path.join(args.model_dir, ".hydra/config.yaml"), "r") as f:
        hps = HParams(yaml.full_load(f))

    dataset = ImageLoader(args.image_dir, hps.data, args.n_tot, args.n_idx)
    collate_fn = ImageCollate()
    loader = DataLoader(dataset, num_workers=8, shuffle=False, pin_memory=False,
                        collate_fn=collate_fn, batch_size=2**6)
    vocab_inv = {v: k for k, v in dataset.vocab.items()}

    model = TableRecognizer(
        len(dataset.vocab),
        3 * (hps.data.patch_length ** 2),
        **hps.model).cuda().eval()

    load_checkpoints(args.model_dir, model, "model_*.pth", args.n_avg)

    prefix = '<html><body><table>'
    postfix = '</table></body></html>'
    html_strings = []
    with torch.no_grad():
        for i, elms in enumerate(loader):
            print(i)
            (img, mask_img, pos_r, pos_c) = elms
            img = img.cuda()
            mask_img = mask_img.cuda()
            pos_r = pos_r.cuda()
            pos_c = pos_c.cuda()

            ret, _ = inference(model, img, mask_img, pos_r, pos_c, beam_size=32, alpha=0.6, max_decode_len=min(10000, math.ceil(4.5 * img.shape[1])))
            ret = ret.cpu().numpy()
            for j, r in enumerate(ret):
                try:
                    eos_pos = list(r).index(2)
                    r = r[:eos_pos]
                except:
                    pass
                html_string = prefix + "".join([vocab_inv[x] for x in r]) + postfix
                html_strings.append(html_string)

    image_names = [x.split("/")[-1] for x in dataset.image_paths]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.out_dir, "out_%d_of_%d.json" % (args.n_idx, args.n_tot)), 'w', encoding='utf-8') as f:
        json.dump({img: txt for img, txt in zip(image_names, html_strings)}, f)
