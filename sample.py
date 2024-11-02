# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner, download_ckpt
from dataset import dataset
from i2sb import ckpt_util

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx}, {end_idx}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log):
    val_dataset = dataset.Dataset(opt, log, mode='test')

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    recon_imgs_fn = RESULT_DIR / opt.ckpt / "samples_nfe{}{}_iter{}".format(
        nfe, "_clip" if opt.clip_denoise else "", opt.load_itr
    )
    os.makedirs(recon_imgs_fn, exist_ok=True)

    return recon_imgs_fn

def compute_batch(ckpt_opt, out):
    clean_img, corrupt_img = out
    x1 = corrupt_img.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)
 
    return clean_img, corrupt_img, x1, cond

def patchify(opt, x0, x1, x1_, patch_size, padding=None):
    batch_size, imgs_size = x0.size(0), x0.size(2)

    if padding is not None:
        padded_x0 = torch.zeros((x0.size(0), x0.size(1), x0.size(2) + padding * 2,
                                x0.size(3) + padding * 2), dtype=x0.dtype, device=opt.device)
        padded_x1 = torch.zeros((x1.size(0), x1.size(1), x1.size(2) + padding * 2,
                                x1.size(3) + padding * 2), dtype=x1.dtype, device=opt.device)
        padded_x1_ = torch.zeros((x1_.size(0), x1_.size(1), x1_.size(2) + padding * 2,
                                x1_.size(3) + padding * 2), dtype=x1_.dtype, device=opt.device)
        padded_x0[:, :, padding:-padding, padding:-padding] = x0
        padded_x1[:, :, padding:-padding, padding:-padding] = x1
        padded_x1_[:, :, padding:-padding, padding:-padding] = x1_
    else:
        padded_x0, padded_x1, padded_x1_ = x0, x1, x1_

    h, w = padded_x0.size(2), padded_x0.size(3)
    th, tw = patch_size, patch_size
    if w == tw and h == th:
        i = torch.zeros((batch_size,), device=opt.device).long()
        j = torch.zeros((batch_size,), device=opt.device).long()
    # elif w//4 == tw and h//4 == th: # For center crop
    #     i = torch.randint(64, h - th - 64 + 1, (batch_size,), device=opt.device)
    #     j = torch.randint(64, w - tw - 64 + 1, (batch_size,), device=opt.device)
    else:
        i = torch.randint(0, h - th + 1, (batch_size,), device=opt.device)
        j = torch.randint(0, w - tw + 1, (batch_size,), device=opt.device)

    rows = torch.arange(th, dtype=torch.long, device=opt.device) + i[:, None]
    columns = torch.arange(tw, dtype=torch.long, device=opt.device) + j[:, None]
    padded_x0, padded_x1, padded_x1_ = padded_x0.permute(1, 0, 2, 3), padded_x1.permute(1, 0, 2, 3), padded_x1_.permute(1, 0, 2, 3)
    padded_x0 = padded_x0[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]],
                columns[:, None]]
    padded_x1 = padded_x1[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]],
                columns[:, None]]
    padded_x1_ = padded_x1_[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]],
                columns[:, None]]
    padded_x0, padded_x1, padded_x1_ = padded_x0.permute(1, 0, 2, 3), padded_x1.permute(1, 0, 2, 3), padded_x1_.permute(1, 0, 2, 3)

    x_pos = torch.arange(tw, dtype=torch.long, device=opt.device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    y_pos = torch.arange(th, dtype=torch.long, device=opt.device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    x_pos = x_pos + j.view(-1, 1, 1, 1)
    y_pos = y_pos + i.view(-1, 1, 1, 1)
    x_pos = (x_pos / (imgs_size - 1) - 0.5) * 2.
    y_pos = (y_pos / (imgs_size - 1) - 0.5) * 2.
    imgs_pos = torch.cat((x_pos, y_pos), dim=1)

    assert padded_x0.shape == padded_x1.shape == padded_x1_.shape

    return padded_x0, padded_x1, padded_x1_, imgs_pos

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    nfe = opt.nfe or ckpt_opt.interval-1

    # build imagenet val dataset
    val_dataset = build_val_dataset(opt, log)
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder
    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    recon_imgs = []
    num = 0

    batch_mul_dict = {512: 1, 256: 2, 128: 4, 64: 16, 32: 32, 16: 64}
    # p_list = np.array([(1-ckpt_opt.real_p)*2/5, (1-ckpt_opt.real_p)*3/5, ckpt_opt.real_p]) # only want to see patchfy results
    p_list = np.array([0, 0, 1])
    patch_list = np.array([opt.image_size//4, opt.image_size//2, opt.image_size])

    for loader_itr, out in enumerate(val_loader):
        patch_size = int(np.random.choice(patch_list, p=p_list))
        batch_mul = batch_mul_dict[patch_size] // batch_mul_dict[opt.image_size]

        if ckpt_opt.run_patch:
            clean_img, corrupt_img, x1, cond = [], [], [], []
            for _ in range(batch_mul):
                clean_img_, corrupt_img_, x1_, cond_ = compute_batch(ckpt_opt, out)
                clean_img.append(clean_img_), corrupt_img.append(corrupt_img_), x1.append(x1_), cond.append(cond_)
            clean_img, corrupt_img, x1 = torch.cat(clean_img, dim=0), torch.cat(corrupt_img, dim=0), torch.cat(x1, dim=0)
            clean_img, corrupt_img, x1, x_pos = patchify(opt, clean_img, corrupt_img, x1, patch_size)
            if cond.count("None") == 0:
                cond = x1.detach()
            else:
                cond = None
            del clean_img_, corrupt_img_, x1_, cond_
        else:
            clean_img, corrupt_img, x1, cond = compute_batch(ckpt_opt, out)
            x_pos = None

        xs, _ = runner.ddpm_sampling(
            ckpt_opt, x1, x_pos=x_pos, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe, verbose=opt.n_gpu_per_node==1
        )
        recon_img = xs[:, 0, ...].to(opt.device)

        assert recon_img.shape == corrupt_img.shape == clean_img.shape

        tu.save_image((corrupt_img+1)/2, recon_imgs_fn / f"{loader_itr:05}_source.png", value_range=(0, 1))
        tu.save_image((clean_img+1)/2, recon_imgs_fn / f"{loader_itr:05}_target.png", value_range=(0, 1))
        tu.save_image((recon_img+1)/2, recon_imgs_fn / f"{loader_itr:05}_target_recon.png", value_range=(0, 1))
        # if ckpt_opt.cond_x1:
        #     tu.save_image((cond[i:i+1, ...]+1)/2, RESULT_DIR / opt.ckpt / f"output_nfe{nfe}/{loader_itr:05}_{i}_cond_x1.png")            
        log.info("Saved output images!")

        # [-1,1]
        gathered_recon_img = collect_all_subset(recon_img, log)
        recon_imgs.append(gathered_recon_img)

        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")
        dist.barrier()

    del runner

    arr = torch.cat(recon_imgs, axis=0)[:n_samples]

    if opt.global_rank == 0:
        torch.save({"arr": arr}, recon_imgs_fn / "recon.pt")
        log.info(f"Save at {recon_imgs_fn}")
    dist.barrier()

    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--master-port",    type=str,  default='6020',      help="port for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=512)
    parser.add_argument("--dataset-dir",    type=Path, default="/data",  help="path to dataset")
    parser.add_argument("--src",            type=str,  default='src',       help="source folder name")
    parser.add_argument("--trg",            type=str,  default='trg',       help="target folder name")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--load-itr",       type=int,  default=15000)
    parser.add_argument("--batch-size",     type=int,  default=1)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
