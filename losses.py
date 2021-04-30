import torch
from torch.nn import functional as F


def loss_fn_img(src, tgt, mask):
    src = src.float()
    tgt = tgt.float()
    c = src.size(-1)

    loss = (src - tgt) ** 2
    return (loss * mask).sum() / (c * mask.sum())


def loss_fn_txt(src, tgt, mask):
    src = src.transpose(1,2).float()
    loss = F.cross_entropy(src, tgt, reduction='none')
    return (loss * mask).sum() / (mask.sum())
