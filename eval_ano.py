#!/usr/bin/env python3

import os
import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import accelerate
import torch
from torch import optim
from torch import multiprocessing as mp
from torch.utils import data
from tqdm.auto import tqdm

import k_diffusion as K
import matplotlib.pyplot as plt

import random
import numpy as np

GLOBAL_SEED = 42


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    # torch.use_deterministic_algorithms(True)

def args_parser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, required=False,
                   help='the configuration file')

    p.add_argument('--batch_test', type=int, default=256,
                   help='the test batch size')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    
    p.add_argument('--n_sample', type=int,
                   help='evaluate sample size')

    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=6,
                   help='the number of data loader workers')
                   
    p.add_argument('--seed', type=int,  default=42,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='fork',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    
    return p.parse_args()


def get_datasets():
    from feat_load import ClipDataset
    
    test_set = ClipDataset(
        root_dir='data',
        dataset_name='shanghai',
        )
    
    return test_set

def main():
    args = args_parser()

    seed_everything(seed=args.seed)

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True


    config = K.config.load_config(open('configs/config_ano.json'))
    model_config = config['model']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']
    feat_size = model_config['input_size']
    test_set = get_datasets()
    

    if model_config["sampler"] == "lms":
        p_sampler_fn = partial(K.sampling.sample_lms, disable=True)
    elif model_config["sampler"] == "heun":
        p_sampler_fn = partial(K.sampling.sample_heun, disable=True)
    elif model_config["sampler"] == "dpm2":
        p_sampler_fn = partial(K.sampling.sample_dpm_2, disable=True)
    else:
        print('unknown sampler method')
        ValueError('Invalid sampler method')


    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1,
                                         cpu=False)
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)


    gvad_model = K.models.GVADModel(
        feat_size,
    )


    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(gvad_model.parameters(),
                          lr=opt_config['lr'] if args.lr is None else args.lr,
                          betas=tuple(opt_config['betas']),
                          eps=opt_config['eps'],
                          weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(gvad_model.parameters(),
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
            print('Number of items in testset:', len(test_set))
        except TypeError:
            pass

    

    test_dl = data.DataLoader(test_set, args.batch_test, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, persistent_workers=True)

    model, opt, test_dl = accelerator.prepare(gvad_model, opt, test_dl)


    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']

    seed_noise_path = f'seed_noise_{feat_size}.pth'
    if os.path.exists(seed_noise_path):
        seed_noise = torch.load(seed_noise_path, map_location=device)
    else:
        seed_noise = torch.randn([1, feat_size], device=device)
        torch.save(seed_noise, seed_noise_path)



    model = K.config.make_denoiser_wrapper(config)(model)
    model_ema = deepcopy(model)

    ckpt_path = 'trained_model/model.pth'
    if Path(ckpt_path).exists():
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        accelerator.unwrap_model(model.gvad_model).load_state_dict(ckpt['model'])
        accelerator.unwrap_model(model_ema.gvad_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        del ckpt
    else:
        print('state path does not exist.')
        print(ckpt_path)
        exit(1)
    
    
    def plot_prediction(g_dist, label, vids, idx, start_n):
        g_dist = K.utils.normalize(g_dist)
        uniques = np.unique(vids)
        for item in uniques:
            v_path = f'plots/{item}'
            K.utils.mkdir(v_path)
            mask = np.where(vids == item)[0]
            sort_idx = np.argsort(idx[mask])
            
            vid_dist = g_dist[mask]
            vid_dist = vid_dist[sort_idx]
            
            vid_label = label[mask]
            vid_label = vid_label[sort_idx]

            plt.title(f'video_{item}_{start_n}')
            plt.subplot(211); plt.plot(vid_dist, label='dist', c='b')
            plt.tick_params(axis='x',
                            which='both',
                            bottom=False,
                            top=False,
                            labelbottom=False)
            plt.ylim(0, 1.1)
            plt.legend()
            plt.subplot(212); plt.plot(vid_label, label='gt', c='r')
            plt.legend()
            plt.ylim(0, 1.1)
            plt.savefig(f'{v_path}/video_{item}_{start_n}.png')
            plt.close()


    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate(n_start):
        
        if accelerator.is_main_process:
            tqdm.write('Evaluating...')
        
        sigmas = K.sampling.get_sigmas_karras(10, sigma_min, sigma_max, rho=7., device=device)
        sigmas = sigmas[n_start:]
        sample_noise = seed_noise.to(device) * sigmas[0]
        
        def sample_fn(x_real):
            x_real = x_real.to(device)
            x = sample_noise + x_real
            x_0 = p_sampler_fn(model_ema.forward, x, sigmas)
            
            g_dists = model_ema.gvad_model.loss(x_real, x_0)
            
            return g_dists

        g_dists, labels, vid, idx = K.evaluation.compute_eval_outs_aot(accelerator, sample_fn, test_dl)
        
        vid = np.concatenate(vid)
        return g_dists.cpu().numpy(), labels.cpu().numpy(), vid, idx.cpu().numpy()

    for i in range(10):
        g_dists, labels, vids, idx = evaluate(i)
        plot_prediction(g_dists, labels, vids, idx, i)
        




if __name__ == '__main__':
    print('Hello there!')
    main()
    print('Obiwan Kenobi.')

