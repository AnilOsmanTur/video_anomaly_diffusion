import math

import torch
from torch import nn
from torch.nn import functional as F

from . import utils

# Karras et al. preconditioned denoiser

class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, gvad_model, sigma_data=1.):
        super().__init__()
        self.gvad_model = gvad_model
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        x_hat = self.gvad_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        gloss = self.gvad_model.loss(x_hat, target)
        return gloss
    
    def loss1(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        x_hat = self.gvad_model.forward1(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        gloss = self.gvad_model.loss(x_hat, target)
        return gloss
    
    def loss2(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        x_hat = self.gvad_model.forward2(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        gloss = self.gvad_model.loss(x_hat, target)
        return gloss

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        x_hat = self.gvad_model(input * c_in, sigma, **kwargs)
        return x_hat * c_out + input * c_skip
    
    def forward1(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        x_hat = self.gvad_model.forward1(input * c_in, sigma, **kwargs)
        return x_hat * c_out + input * c_skip
    
    def forward2(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        x_hat = self.gvad_model.forward2(input * c_in, sigma, **kwargs)
        return x_hat * c_out + input * c_skip




class DenoiserWithVariance(Denoiser):
    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output, logvar = self.inner_model(noised_input * c_in, sigma, return_variance=True, **kwargs)
        logvar = utils.append_dims(logvar, model_output.ndim)
        target = (input - c_skip * noised_input) / c_out
        losses = ((model_output - target) ** 2 / logvar.exp() + logvar) / 2
        return losses.flatten(1).mean(1)



# Embeddings

class FourierFeatures_aot(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return f.cos(), f.sin()


