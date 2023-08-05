import math
import os
from pathlib import Path


import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm

from . import utils


def compute_eval_outs_aot(accelerator, sample_fn, dl):
    outputs = []
    labels = []
    video_id = []
    idx = []

    for batch in tqdm(dl, disable=not accelerator.is_main_process):
        feat = batch['data']
        y = batch['label']
        vid = batch['vid_id']
        i = batch['idx']

        g_dists = sample_fn(feat)
        g_dists = accelerator.gather(g_dists)

        outputs.append(g_dists)
        labels.append(y)
        video_id.append(vid)
        idx.append(i)
        
    labels = torch.cat(labels)
    g_dists = torch.cat(outputs)
    idx = torch.cat(idx)
    return g_dists, labels, video_id, idx




def polynomial_kernel(x, y):
    d = x.shape[-1]
    dot = x @ y.transpose(-2, -1)
    return (dot / d + 1) ** 3


def squared_mmd(x, y, kernel=polynomial_kernel):
    m = x.shape[-2]
    n = y.shape[-2]
    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    kxx_sum = kxx.sum([-1, -2]) - kxx.diagonal(dim1=-1, dim2=-2).sum(-1)
    kyy_sum = kyy.sum([-1, -2]) - kyy.diagonal(dim1=-1, dim2=-2).sum(-1)
    kxy_sum = kxy.sum([-1, -2])
    term_1 = kxx_sum / m / (m - 1)
    term_2 = kyy_sum / n / (n - 1)
    term_3 = kxy_sum * 2 / m / n
    return term_1 + term_2 - term_3


@utils.tf32_mode(matmul=False)
def kid(x, y, max_size=5000):
    x_size, y_size = x.shape[0], y.shape[0]
    n_partitions = math.ceil(max(x_size / max_size, y_size / max_size))
    total_mmd = x.new_zeros([])
    for i in range(n_partitions):
        cur_x = x[round(i * x_size / n_partitions):round((i + 1) * x_size / n_partitions)]
        cur_y = y[round(i * y_size / n_partitions):round((i + 1) * y_size / n_partitions)]
        total_mmd = total_mmd + squared_mmd(cur_x, cur_y)
    return total_mmd / n_partitions


class _MatrixSquareRootEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        vals, vecs = torch.linalg.eigh(a)
        ctx.save_for_backward(vals, vecs)
        return vecs @ vals.abs().sqrt().diag_embed() @ vecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output):
        vals, vecs = ctx.saved_tensors
        d = vals.abs().sqrt().unsqueeze(-1).repeat_interleave(vals.shape[-1], -1)
        vecs_t = vecs.transpose(-2, -1)
        return vecs @ (vecs_t @ grad_output @ vecs / (d + d.transpose(-2, -1))) @ vecs_t


def sqrtm_eig(a):
    if a.ndim < 2:
        raise RuntimeError('tensor of matrices must have at least 2 dimensions')
    if a.shape[-2] != a.shape[-1]:
        raise RuntimeError('tensor must be batches of square matrices')
    return _MatrixSquareRootEig.apply(a)


@utils.tf32_mode(matmul=False)
def fid(x, y, eps=1e-8):
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    mean_term = (x_mean - y_mean).pow(2).sum()
    x_cov = torch.cov(x.T)
    y_cov = torch.cov(y.T)
    eps_eye = torch.eye(x_cov.shape[0], device=x_cov.device, dtype=x_cov.dtype) * eps
    x_cov = x_cov + eps_eye
    y_cov = y_cov + eps_eye
    x_cov_sqrt = sqrtm_eig(x_cov)
    cov_term = torch.trace(x_cov + y_cov - 2 * sqrtm_eig(x_cov_sqrt @ y_cov @ x_cov_sqrt))
    return mean_term + cov_term
