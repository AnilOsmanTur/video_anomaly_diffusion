#!/usr/bin/env python3

import os
import argparse
from copy import deepcopy
from functools import partial
import math
import json
from pathlib import Path

import accelerate
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import datasets, transforms, utils
from tqdm.auto import trange, tqdm

import k_diffusion as K

from torchmetrics.functional import auroc, accuracy, precision, f1_score

import random
import numpy as np


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def get_gen_thresh(gen_pred):
    g_std, g_mean = torch.std_mean(gen_pred, unbiased=True)
    return g_std + g_mean

def get_dis_thresh(dis_pred):
    d_std, d_mean = torch.std_mean(dis_pred, unbiased=True)
    return d_mean + 0.1 * d_std


def args_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=8192,
                   help='the batch size')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--n_epochs', type=int, default=10,
                   help='number of epoch')
    p.add_argument('--evaluate', action='store_true', default=True,
                   help='evaluate step switcher')
    p.add_argument('--gns', action='store_true',
                   help='measure the gradient noise scale (DDP only)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--grow', type=str,
                   help='the checkpoint to grow from')
    p.add_argument('--grow-config', type=str,
                   help='the configuration file of the model to grow from')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--wandb-entity', type=str,
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    return p.parse_args()

def main_objective(config_tune, args):
    self_pretrain = False
    anoamly_fit = False
    neg_learning = True
    coop = True
    no_overlap = False

    config = K.config.load_config(open(args.config))
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']

    model_config['sigma_max'] = config_tune["sigma_max"]
    model_config['sigma_min'] = config_tune["sigma_min"]
    model_config['sigma_data'] = config_tune["sigma_data"]
    model_config['sigma_sample_density']['mean'] = config_tune["ssd_mean"]
    model_config['sigma_sample_density']['std'] = config_tune["ssd_std"]
    opt_config['lr'] = config_tune["lr"]
    opt_config['weight_decay'] = config_tune["weight_decay"]

    if dataset_config['type'] == 'clips':
        from custom_datasets import feature_dataset
        dataset_root: str = 'data/'
        dataset_name: str = 'shanghai'
        # feat_models_3D = ['rx101', 'r3d18', 'r3d50']
        train_set = feature_dataset.ClipDataset(
            dataset_root + dataset_name,
            clip_len=16,
            feat_model=dataset_config['feat_model'],
            split='train',
            clean=False,
            normals=False,  # to train ooc setting
        )
        test_set = feature_dataset.ClipDataset(
            dataset_root + dataset_name,
            clip_len=16,
            feat_model=dataset_config['feat_model'],
            split='test',
            clean=False,
        )
    else:
        raise ValueError('Invalid dataset type')

    feat, label = train_set[0]
    feat_size = feat.shape[-1]

    if model_config['input_size'] != feat_size:
        print('input size replace.')
        model_config['input_size'] = feat_size

    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=model_config['skip_stages'] > 0)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs],
                                         gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes],
                              generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])

    if coop:
        gcl_dis = K.models.GCLModelDiscriminator(feat_size,
                                                 neg_learning,
                                                 coop)
        gcl_dis = gcl_dis.to(device)

    # inner_model = K.config.make_model_aot(config)
    gcl_model = K.models.GCLModel(
        feat_size,
        neg_learning,
        coop,
        no_overlap
    )
    if accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        print('Parameters:', K.utils.n_params(gcl_model))
        com_str = ''
        com_str += f'_lr:{config_tune["lr"]}'
        com_str += f'_s_max:{config_tune["sigma_max"]}'
        com_str += f'_s_min:{config_tune["sigma_min"]}'
        com_str += f'_s_data:{config_tune["sigma_data"]}'
        com_str += f'_ssd_mean:{config_tune["ssd_mean"]}'
        com_str += f'_ssd_std:{config_tune["ssd_std"]}'
        com_str += f'_weight_decay:{config_tune["weight_decay"]}'
        writer = SummaryWriter(comment=com_str)



    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(gcl_model.parameters(),
                          lr=opt_config['lr'] if args.lr is None else args.lr,
                          betas=tuple(opt_config['betas']),
                          eps=opt_config['eps'],
                          weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(gcl_model.parameters(),
                        lr=opt_config['lr'] if args.lr is None else args.lr,
                        momentum=opt_config.get('momentum', 0.),
                        nesterov=opt_config.get('nesterov', False),
                        weight_decay=opt_config.get('weight_decay', 0.))
    else:
        raise ValueError('Invalid optimizer type')

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                  inv_gamma=sched_config['inv_gamma'],
                                  power=sched_config['power'],
                                  warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                      num_steps=sched_config['num_steps'],
                                      decay=sched_config['decay'],
                                      warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                  max_value=ema_sched_config['max_value'])

    if accelerator.is_main_process:
        try:
            print('Number of items in dataset:', len(train_set))
        except TypeError:
            pass



    # dataset key init if the dataset has a dict return
    image_key = dataset_config.get('image_key', 0)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True)

    test_dl = data.DataLoader(test_set, args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, persistent_workers=True)


    model, opt, train_dl = accelerator.prepare(gcl_model, opt, train_dl)
    inner_model = model.ae
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    model_denoiser = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = deepcopy(model_denoiser)

    epoch = 0
    step = 0

    evaluate_enabled = args.evaluate
    if evaluate_enabled:
        test_size = len(test_set)
        if accelerator.is_main_process:
            metrics_log = K.utils.CSVLogger(f'{args.name}_metrics.csv', ['step', 'acc', 'auc', 'precision', 'f1'])
            loss_log = K.utils.CSVLogger(f'{args.name}_loss.csv', ['step', 'loss'])

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate():
        if not evaluate_enabled:
            return
        if accelerator.is_main_process:
            tqdm.write('Evaluating...')
        sigmas = K.sampling.get_sigmas_karras(10, sigma_min, sigma_max, rho=7., device=device)
        sample_noise = torch.randn([1, feat_size], device=device) * sigma_max

        def sample_fn(x_real):
            x_real = x_real.to(device)
            x = sample_noise + x_real
            x_0 = K.sampling.sample_lms(model_ema, x, sigmas, disable=True)
            return x_0

        gen_preds, dis_preds, labels = K.evaluation.compute_eval_outs_aot(accelerator, sample_fn, test_size, test_dl)
        if accelerator.is_main_process:

            preds_auc = auroc(gen_preds, labels)
            preds_acc = accuracy(gen_preds, labels) * 100
            preds_prec = precision(gen_preds, labels) * 100
            preds_f1 = f1_score(gen_preds, labels)

            print(f'acc: {preds_acc.item():g}, auc: {preds_auc.item():g}, precision: {preds_prec.item():g}, f1: {preds_f1.item():g}')
            if accelerator.is_main_process:
                metrics_log.write(step,
                                  preds_acc.item(),
                                  preds_auc.item(),
                                  preds_prec.item(),
                                  preds_f1.item())
                writer.add_scalar('Acc/test', preds_acc.item(), step)
                writer.add_scalar('AUC/test', preds_auc.item(), step)
                writer.add_scalar('Prec/test', preds_prec.item(), step)
                writer.add_scalar('F1/test', preds_f1.item(), step)

            return preds_acc.item(), preds_auc.item(), preds_prec.item(), preds_f1.item()


    if evaluate_enabled:
        evaluate()

    try:
        for i in range(args.n_epochs):
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                with accelerator.accumulate(model):
                    reals = batch[image_key]
                    noise = torch.randn_like(reals)
                    sigma = sample_density([reals.shape[0]], device=device)
                    g_losses = model_denoiser.loss(reals, noise, sigma) # losses with the batch
                    gen_dist = accelerator.gather(g_losses)

                    d_loss = model.loss(batch, gen_dist)
                    d_loss = accelerator.gather(d_loss)
                    g_loss = gen_dist.mean()
                    loss = g_loss + d_loss
                    accelerator.backward(loss)
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals.shape[0],
                                         reals.shape[0] * accelerator.num_processes)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    if accelerator.sync_gradients:
                        ema_decay = ema_sched.get_value()
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if accelerator.is_main_process:
                    writer.add_scalar('Epoch/train', epoch, step)
                    writer.add_scalar('Loss/train', loss.item(), step)
                    writer.add_scalar('Lr/train', sched.get_last_lr()[0], step)
                    writer.add_scalar('ema_decay/train', ema_decay, step)

                    if step % 25 == 0:
                        if args.gns:
                            tqdm.write(
                                f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')


                step += 1
            loss_log.write(step, loss.item())
            preds_acc, preds_auc, preds_prec, preds_f1 = evaluate()

            epoch += 1
    except KeyboardInterrupt:
        pass


def main_param_search(num_samples=200, n_epochs=30):
    args = args_parser()
    args.n_epochs = n_epochs
    seed_everything(seed=68)
    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True

    for i in range(num_samples):
        config = {
            "sigma_min": np.random.uniform(1e-4, 1e-1, 1)[0],
            "sigma_max": np.random.uniform(1e-1, 5, 1)[0],
            "sigma_data": np.random.uniform(1e-2, 1, 1)[0],
            "ssd_mean": np.random.uniform(-2, 2, 1)[0],
            "ssd_std": np.random.uniform(1e-1, 2, 1)[0],
            "lr": np.random.uniform(1e-6, 1e-4, 1)[0],
            "weight_decay": np.random.uniform(1e-6, 1e-1, 1)[0]
        }
        main_objective(config, args)


def main(n_epochs=30):
    args = args_parser()
    args.n_epochs = n_epochs
    seed_everything(seed=68)
    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = {
        "sigma_min": 0.04,
        "sigma_max": 1.8,
        "sigma_data": 0.4,
        "ssd_mean": 0.2,
        "ssd_std": 0.54,
        "lr": 8e-6,
        "weight_decay": 0.0016,

    }
    main_objective(config, args)


if __name__ == '__main__':
    main()