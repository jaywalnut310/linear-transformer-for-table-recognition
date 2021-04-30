import os
import math
import hydra
from omegaconf import OmegaConf
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import utils
import commons
from models import TableRecognizer
from data_utils import (
    ImageTextLoader,
    ImageTextCollate,
    DistributedBucketSampler
)
from losses import (
    loss_fn_img,
    loss_fn_txt
)


global_step = 0


@hydra.main(config_path='configs/', config_name='linear_transformer')
def main(hps):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    assert OmegaConf.select(hps, "model_dir") != ".hydra", "Please specify model_dir."
    print(OmegaConf.to_yaml(hps))
  
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80000'
  
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        writer = SummaryWriter(log_dir='./')
        writer_eval = SummaryWriter(log_dir='./eval')
  
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
  
    train_dataset = ImageTextLoader(hps.data.training_file_path, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.num_tokens,
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = ImageTextCollate()
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = ImageTextLoader(hps.data.validation_file_path, hps.data)
        eval_sampler = DistributedBucketSampler(
            eval_dataset,
            hps.train.num_tokens,
            num_replicas=1,
            rank=rank,
            shuffle=True)
        eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False, pin_memory=False,
            collate_fn=collate_fn, batch_sampler=eval_sampler)
  
    model = TableRecognizer(
        len(train_dataset.vocab),
        3 * (hps.data.patch_length ** 2),
        **hps.model).cuda(rank)
    optim = torch.optim.Adam(
        model.parameters(), 
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)
    model = DDP(model, device_ids=[rank])
  
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path('./', "model_*.pth"), model, optim)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0
  
    scaler = GradScaler(enabled=hps.train.fp16_run)
  
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank==0:
            train_and_evaluate(rank, epoch, hps, model, optim, scaler, [train_loader, eval_loader], [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, model, optim, scaler, [train_loader, None], None)


def train_and_evaluate(rank, epoch, hps, model, optim, scaler, loaders, writers):
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
  
    train_loader.batch_sampler.set_epoch(epoch)
    global global_step
  
    model.train()
    for batch_idx, (img, txt, mask_img, mask_txt, pos_r, pos_c, pos_t) in enumerate(train_loader):
        img = img.cuda(rank, non_blocking=True)
        txt = txt.cuda(rank, non_blocking=True)
        mask_img = mask_img.cuda(rank, non_blocking=True)
        mask_txt = mask_txt.cuda(rank, non_blocking=True)
        pos_r = pos_r.cuda(rank, non_blocking=True)
        pos_c = pos_c.cuda(rank, non_blocking=True)
        pos_t = pos_t.cuda(rank, non_blocking=True)

        img_i = img[:,:-1]
        img_o = img[:,1:]
        txt_i = txt[:,:-1]
        txt_o = txt[:,1:]
        mask_img_i = mask_img[:,:-1]
        mask_img_o = mask_img[:,1:]
        mask_txt_i = mask_txt[:,:-1]
        mask_txt_o = mask_txt[:,1:,0]

        with autocast(enabled=hps.train.fp16_run):
            logits_img, logits_txt = model(img_i, txt_i, mask_img_i, mask_txt_i, pos_r, pos_c, pos_t)
            with autocast(enabled=False):
                loss_img = loss_fn_img(logits_img, img_o, mask_img_o)
                loss_txt = loss_fn_txt(logits_txt, txt_o, mask_txt_o)
                loss_tot = loss_img * hps.train.lamb + loss_txt
        optim.zero_grad()
        scaler.scale(loss_tot).backward()
        scaler.unscale_(optim)
        grad_norm = commons.grad_norm(model.parameters())
        scaler.step(optim)
        scaler.update()
  
        if rank==0:
            num_tokens = mask_img.sum() + mask_txt.sum()
            if global_step % hps.train.log_interval == 0:
                lr = optim.param_groups[0]['lr']
                losses = [loss_tot, loss_img, loss_txt]
                print('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                print([x.item() for x in losses] + [global_step, lr])
          
                scalar_dict = {"loss/total": loss_tot, "loss/img": loss_img, "loss/txt": loss_txt}
                scalar_dict.update({"learning_rate": lr, "grad_norm": grad_norm, "num_tokens": num_tokens})
          
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict)
  
            if global_step % hps.train.eval_interval == 0:
                print("START: EVAL")
                eval_loader.batch_sampler.set_epoch(global_step)
                evaluate(hps, model, eval_loader, writer_eval)
                utils.save_checkpoint(model, optim, hps.train.learning_rate, epoch, 
                    "model_{}.pth".format(global_step)
                )
                print("END: EVAL")
        global_step += 1
  
    if rank == 0:
        print('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, model, eval_loader, writer_eval):
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, txt, mask_img, mask_txt, pos_r, pos_c, pos_t) in enumerate(eval_loader):
            img = img.cuda(0, non_blocking=True)
            txt = txt.cuda(0, non_blocking=True)
            mask_img = mask_img.cuda(0, non_blocking=True)
            mask_txt = mask_txt.cuda(0, non_blocking=True)
            pos_r = pos_r.cuda(0, non_blocking=True)
            pos_c = pos_c.cuda(0, non_blocking=True)
            pos_t = pos_t.cuda(0, non_blocking=True)

            img_i = img[:,:-1]
            img_o = img[:,1:]
            txt_i = txt[:,:-1]
            txt_o = txt[:,1:]
            mask_img_i = mask_img[:,:-1]
            mask_img_o = mask_img[:,1:]
            mask_txt_i = mask_txt[:,:-1]
            mask_txt_o = mask_txt[:,1:,0]

            with autocast(enabled=hps.train.fp16_run):
                logits_img, logits_txt = model(img_i, txt_i, mask_img_i, mask_txt_i, pos_r, pos_c, pos_t)
                with autocast(enabled=False):
                    loss_img = loss_fn_img(logits_img, img_o, mask_img_o)
                    loss_txt = loss_fn_txt(logits_txt, txt_o, mask_txt_o)
                    loss_tot = loss_img * hps.train.lamb + loss_txt
            break

    scalar_dict = {"loss/total": loss_tot.item(), "loss/img": loss_img.item(), "loss/txt": loss_txt.item()}

    utils.summarize(
        writer=writer_eval,
        global_step=global_step, 
        scalars=scalar_dict,
    )
    model.train()


if __name__ == "__main__":
  main()
