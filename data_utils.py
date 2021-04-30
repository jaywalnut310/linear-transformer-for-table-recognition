import time
import os
import json
from PIL import Image
import random
import math
import numpy as np
import torch
import torch.utils.data


class ImageTextLoader(torch.utils.data.Dataset):
    """
    Load image, text pairs
    """
    def __init__(self, file_path, cfg):
        self.cfg = cfg
        self.image_paths, self.texts, self.lengths, \
        self.image_heights, self.image_widths, self.text_lengths = self._build(file_path)
        self.vocab = self._load_vocab()

    def _build(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        image_paths = []
        texts = []
        lengths = []
        image_heights = []
        image_widths = []
        text_lengths = []
        for elm in data:
            image_paths.append(elm['image_path'])
            texts.append(elm['text'])
            w, h = [math.ceil(x / self.cfg.patch_length) for x in elm['image_size']]
            t = elm['num_tokens']

            lengths.append(h * w + t)
            image_heights.append(h)
            image_widths.append(w)
            text_lengths.append(t)
        return image_paths, texts, lengths, image_heights, image_widths, text_lengths

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

        text = torch.LongTensor([self.vocab[w] for w in self.texts[index]])
        length = self.lengths[index]
        image_height = h
        image_width = w
        text_length = self.text_lengths[index]
        return (image, text, length, image_height, image_width, text_length)

    def __getitem__(self, index):
        return self.get_items(index)

    def __len__(self):
        return len(self.image_paths)


class ImageTextCollate():
    """ Zero-pads model inputs
    """
    def __call__(self, batch):
        """Collate's training batch from image and text info
        Inputs:
        - batch: [img, txt, t_tot, h_img, w_img, t_txt]

        Outputs:
        - (img_padded, txt_padded, mask_img, mask_txt, pos_r, pos_c, pos_t)
        """
        max_len = max(x[2] for x in batch)
        b = len(batch)
        c = batch[0][0].size(1) # image patch size

        img_padded = torch.FloatTensor(b, max_len, c)
        txt_padded = torch.LongTensor(b, max_len)
        mask_img   = torch.FloatTensor(b, max_len, 1)
        mask_txt   = torch.FloatTensor(b, max_len, 1)
        pos_r      = torch.FloatTensor(b, max_len-1, 1) # for teacher forcing
        pos_c      = torch.FloatTensor(b, max_len-1, 1) # for teacher forcing
        pos_t      = torch.FloatTensor(b, max_len-1, 1) # for teacher forcing
        
        img_padded.zero_()
        txt_padded.zero_()
        mask_img.zero_()
        mask_txt.zero_()
        pos_r.zero_()
        pos_c.zero_()
        pos_t.zero_()
        for i in range(b):
            img, txt, t_tot, h_img, w_img, t_txt = batch[i]
            t_img = img.size(0)

            img_padded[i, :t_img] = img
            txt_padded[i, t_img:t_tot] = txt
            mask_img[i, :t_img] = 1
            mask_txt[i, t_img:t_tot] = 1
            pos_r[i, :t_img] = torch.arange(h_img).unsqueeze(-1).repeat(1, w_img).view(-1, 1)
            pos_c[i, :t_img] = torch.arange(w_img).repeat(h_img).view(-1, 1)
            pos_t[i, t_img:t_tot-1] = torch.arange(t_txt-1, dtype=torch.float).view(-1, 1)
        return img_padded, txt_padded, mask_img, mask_txt, pos_r, pos_c, pos_t


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar total token sizes in a batch.
    1) choose the minimum highly composite number among which is larger than given num_tokens.
    2) automatically set bucket boundaries and batch_sizes s.t. boundary * batch_size = the highly composite number.
    3) merge buckets that contain smaller number of elements than batch_sizes
    """
    def __init__(self, dataset, num_tokens=2**16, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        highly_composite_numbers = [
            1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
            2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
            83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
            720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
            7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
            36756720, 43243200, 61261200, 73513440, 110270160
        ]

        self.lengths = dataset.lengths
        self.num_tokens = min([i for i in highly_composite_numbers if i >= num_tokens])
        print("%s: num_tokens is changed from %d to %d." % (self.__class__.__name__, num_tokens, self.num_tokens))
  
        self.buckets, self.num_samples_per_bucket, self.batch_sizes = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        self.num_batches = sum([self.num_samples_per_bucket[i] // (self.batch_sizes[i] * self.num_replicas) for i in range(len(self.batch_sizes))])
  
    def _create_buckets(self):
        boundaries, batch_sizes = [], []
        for i in range(1, self.num_tokens + 1):
            q, r = divmod(self.num_tokens, i)
            if r == 0:
                boundaries.append(i)
                if i != 1:
                  batch_sizes.append(q)
        buckets = [[] for _ in range(len(boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length, boundaries)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                batch_sizes.pop(i)
  
        buckets_new = []
        batch_sizes_new = []
        bucket = []
        for i in range(len(buckets) - 1):
            bucket += buckets[i]
            if len(bucket) >= batch_sizes[i] * self.num_replicas:
                buckets_new.append(bucket)
                bucket = []
                batch_sizes_new.append(batch_sizes[i])
        buckets_new.append(bucket + buckets[-1])
        batch_sizes_new.append(batch_sizes[-1])
        buckets = buckets_new
        batch_sizes = batch_sizes_new
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * batch_sizes[i]
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket, batch_sizes
  
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
  
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
  
        batches = []
        for i in range(len(self.buckets)):
            batch_size = self.batch_sizes[i]
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
  
            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
            # batching
            for j in range(len(ids_bucket) // batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j*batch_size:(j+1)*batch_size]]
                batches.append(batch)
  
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
  
        assert sum([len(x) for x in self.batches]) == self.num_samples
        assert len(self.batches) == self.num_batches
        return iter(self.batches)
  
    def _bisect(self, x, boundaries, lo=0, hi=None):
      if hi is None:
          hi = len(boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if boundaries[mid] < x and x <= boundaries[mid+1]:
              return mid
          elif x <= boundaries[mid]:
              return self._bisect(x, boundaries, lo, mid)
          else:
              return self._bisect(x, boundaries, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_batches
