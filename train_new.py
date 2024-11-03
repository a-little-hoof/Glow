import argparse
import logging
import ast
import sys
import os
import time
import datetime

import torch
import torch.optim
import numpy as np

from glow_new import Glow
import dataset
import utils

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

def sample_img(post_model, num, save_folder, temp):
    samples = post_model.sample(16, temp=temp)
    utils.save_images(samples, f'{save_folder}/fakes_{(num):06}_temp{temp}.png', imrange=[-0.5,0.5])

def train(args):    
    device = f'cuda:{args.gpu}'
    torch.manual_seed(0)
    
    args.data_args['input_shape'] = args.input_shape
    train_dataset = getattr(dataset, args.data_type)(train=True,  **args.data_args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    post_model = Glow(3, 32, 4)
    post_model = post_model.to(device)
    # print(post_model)
    # exit()
    print(f"post_model parameters:{sum(p.numel() for p in post_model.parameters() if p.requires_grad)}")
    
    post_optimizer = torch.optim.Adam(post_model.parameters(), lr=args.lr_post)

    save_folder = utils.setup_saver(args.results_dir,  f'-lr{args.lr_post}' + f'-bits{args.num_bits}')
    print(args.__dict__, file=open(f'{save_folder}/config.txt', 'w'))
    sys.stdout = utils.Logger(save_folder+'/log.txt')
    print(args, flush=True)
    writer = SummaryWriter(save_folder)

    utils.save_images(np.stack([train_dataset[i].detach().numpy() for i in range(16)]), f'{save_folder}/reals.png', imrange=[-0.5,0.5])

    print("Training posterior model")

    start_time = time.time()
    idx = 0
    dloader = iter(train_dataloader)
    while True:
        try:    x = next(dloader)
        except StopIteration: 
            dloader = iter(train_dataloader)
            x = next(dloader)

        x = x.to(device)
        post_optimizer.zero_grad()
    
        loss = post_model.get_loss(x, num_bits=args.num_bits)
        writer.add_scalar('loss', loss, idx)
        loss.backward()
        # warmup_lr = args.lr_post * min(1, idx * args.batch_size / (10000 * 10))
        # post_optimizer.param_groups[0]["lr"] = warmup_lr
        total_norm_before = torch.norm(
            torch.stack([torch.norm(p.grad) for p in post_model.parameters() if p.grad is not None])
        )
        print(f"before: {total_norm_before:.4f}")
        nn.utils.clip_grad_norm_(list(post_model.parameters()), args.clip)
        total_norm_after = torch.norm(
            torch.stack([torch.norm(p.grad) for p in post_model.parameters() if p.grad is not None])
        )
        print(f"after: {total_norm_after:.4f}")
        post_optimizer.step()

        if idx % 50 == 0:
            timesec = time.time() - start_time
            timesec = str(datetime.timedelta(seconds=int(timesec)))
            print(f"kImg. : {idx*args.batch_size/1000:.2f}, time : {timesec} Curr. loss : {loss}")
        if idx % 500 == 0:
            sample_img(post_model, idx*args.batch_size//1000, save_folder, temp=1)
            sample_img(post_model, idx*args.batch_size//1000, save_folder, temp=0.5)
            sample_img(post_model, idx*args.batch_size//1000, save_folder, temp=0.8)
            # post_model.save(f'{save_folder}/network_{(idx*args.batch_size//1000):06}.pt')
            # torch.save(post_model.state_dict(), f'{save_folder}/cond_network_{(idx*args.batch_size//1000):06}.pt')

        idx += 1
        if idx >= args.num_iters:
            break

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    # torch.autograd.set_detect_anomaly(True)


    # Multiprocessing arguments
    parser.add_argument('--gpu', default=1, type=int, help='gpu to operate on')

    # training arguments
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_post", type=float, default=1e-06)
    parser.add_argument("--results_dir", type=str, default='/home/baiweimin/yifei/flow-diff/results/flow')
    parser.add_argument("--resume_from", type=str, default='')
    parser.add_argument("--clip", default=0.1, type=float, help="gradient clip for neural network training")

    # model arguments
    parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 32, 32])
    parser.add_argument("--model_type", type=str, default='CondConvINN')
    parser.add_argument("--model_args", type=ast.literal_eval, default={'num_conv_layers':[4, 12], 'num_fc_layers':[4]})
    parser.add_argument("--actnorm", type=lambda b:bool(int(b)), help="0 or 1")

    # data args
    parser.add_argument("--data_type", type=str, default='MNISTDataset')
    parser.add_argument("--data_args", type=ast.literal_eval, default={})
    parser.add_argument("--num_bits", type=int, default=0)

    args = parser.parse_args()
    train(args)