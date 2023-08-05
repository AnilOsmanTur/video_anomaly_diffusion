import math

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions

from .. import layers, utils


class GVADModel(nn.Module):
    def __init__(self,
                 feat_size,):
        super(GVADModel, self).__init__()
        self.ae = FeatureDenoiserModel(feat_size)


    def get_dis_thresh(self, dis_pred, th_alpha=0.1):
        d_std, d_mean = torch.std_mean(dis_pred, unbiased=True, dim=-1)
        return d_mean + th_alpha * d_std


    def loss(self, g_y, g_target):
        g_dist = (g_y - g_target).pow(2)
        g_loss = g_dist.mean(dim=1)
        return g_loss

    def forward(self, x_noised, sigma, cond=None):
        x_hat = self.ae(x_noised, sigma)
        return x_hat





class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, gammas, betas):
    # gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
    # betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    return (gammas * x) + betas


def orthogonal_(module):
    nn.init.orthogonal_(module.weight)
    return module

class MappingNet(nn.Sequential):
    def __init__(self, feats_in, feats_out, n_layers=2):
        layers = []
        for i in range(n_layers):
            layers.append(orthogonal_(nn.Linear(feats_in if i == 0 else feats_out, feats_out)))
            layers.append(nn.GELU())
        super().__init__(*layers)


# generator model with timeembedding
class FeatureDenoiserModel(nn.Module):
    def __init__(self, feat_size=512):
        super(FeatureDenoiserModel, self).__init__()
        
        self.enc_tembed = layers.FourierFeatures_aot(1, feat_size)
        self.dec_tembed = layers.FourierFeatures_aot(1, 256)

        self.enc_in = FiLM()
        self.dec_in = FiLM()

        self.encoder = nn.Sequential(
            nn.Linear(feat_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256 , 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, feat_size)
        )

    def forward(self, x, sigma, model_cond=None):
        c_noise = sigma.log() / 4
        enc_t = self.enc_tembed(utils.append_dims(c_noise, 2))
        dec_t = self.dec_tembed(utils.append_dims(c_noise, 2))
        
        x = self.enc_in(x, enc_t[0], enc_t[1])
        z = self.encoder(x)
        z = self.dec_in(z, dec_t[0], dec_t[1])
        return self.decoder(z)



