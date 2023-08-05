#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:13:38 2022

@author: anil
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchmetrics.functional import auroc, precision, f1_score


from feat_load import ClipDataset
from torch.utils.data import DataLoader, ConcatDataset


class GCLModel(pl.LightningModule):
    def __init__(self,
                 path='../data/shanghai',
                 feat_model='r3d18',
                 batch_size=16,
                 lr=1e-5,
                 lr_step = 10,
                 worker_n=1,
                 clip_len=16,
                 self_pretrain=False,
                 anoamly_fit=False,
                 neg_learning=True,
                 coop=True):
                 
        super(GCLModel, self).__init__()
        self.lr = lr
        self.lr_step = lr_step
        self.feat_model = feat_model
        self.self_pretrain = self_pretrain
        self.anoamly_fit = anoamly_fit
        self.data_dir = path
        self.batch_size = batch_size
        self.num_workers = worker_n
        self.clip_len = clip_len
        
        feat_size = self.get_feat_size()
        
        self.ae = Autoencoder(feat_size=feat_size)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss()

        if coop:
            self.dis = Discriminator(feat_size=feat_size)
            
            self.forward = self.forward_coop
            self.video_predict = self.video_predict_coop

            if neg_learning:
                self.gen_dis_loss = self.gen_dis_loss_neg_learn
            else:
                self.gen_dis_loss = self.gen_dis_loss_no_neg
            
            self.val_auc_loss = self.val_auc_loss_coop

            self.training_step = self.training_step_coop
            self.validation_step = self.validation_step_coop
        else:
            self.forward = self.forward_ae
            self.video_predict = self.video_predict_ae
            self.training_step = self.training_step_ae
            self.validation_step = self.validation_step_ae
        
        


    def get_dis_thresh(self, dis_pred):
        d_std, d_mean = torch.std_mean(dis_pred, unbiased=True)
        return d_mean + 0.1 * d_std

    def get_gen_thresh(self, gen_pred):
        g_std, g_mean = torch.std_mean(gen_pred, unbiased=True)
        return g_std + g_mean

    
    def val_auc_loss_coop(self, gen_pred, dis_pred, label):
        g_thresh = self.get_gen_thresh(gen_pred)
        d_thresh = self.get_dis_thresh(dis_pred)

        gen_pred = (gen_pred > g_thresh).float()
        # discriminator predictions
        dis_pred = (dis_pred > d_thresh).float()

        # gen_bce = self.bce_loss(gen_pred, label.float())
        # dis_bce = self.bce_loss(dis_pred, label.float())
        gen_auc = auroc(gen_pred.cpu(), label.cpu(), num_classes=None)
        dis_auc = auroc(dis_pred.cpu(), label.cpu(), num_classes=None)
        return gen_auc, dis_auc
    
    def gen_dis_loss_neg_learn(self, x_hat, gen_y, dis_y):
        # generator pseudo labels from recreation loss
        g_dist = self.mse_loss(gen_y, x_hat)
        batch_dist = g_dist.mean(dim=1)
        g_thresh = self.get_gen_thresh(batch_dist)

        dis_y_pseu = (batch_dist > g_thresh).float()
        # discriminator loss
        d_loss = self.bce_loss(dis_y, dis_y_pseu)
        # discriminator pseudo labels from predictions
        d_thresh = self.get_dis_thresh(dis_y)
        mask = dis_y < d_thresh
        gen_y[mask] = torch.ones_like(gen_y[mask])
        # generator loss
        g_loss = self.mse_loss(gen_y, x_hat).mean()
        return g_loss, d_loss

    def gen_dis_loss_no_neg(self, x_hat, gen_y, dis_y):
        # generator pseudo labels from recreation loss
        g_dist = self.mse_loss(gen_y, x_hat)
        batch_dist = g_dist.mean(dim=1)
        g_thresh = self.get_gen_thresh(batch_dist)

        dis_y_pseu = (batch_dist > g_thresh).float()
        # discriminator loss
        d_loss = self.bce_loss(dis_y, dis_y_pseu)
        
        # generator loss
        g_loss = self.mse_loss(gen_y, x_hat).mean()
        return g_loss, d_loss

    def forward_ae(self, x):
        g_y = self.ae(x)
        return x, g_y
        
    def forward_coop(self, x):
        g_y = self.ae(x)
        d_y = self.dis(x)
        return x, g_y, d_y
    
    def video_predict_ae(self, x):
        x_hat, gen_y = self(x)
        # generator predictions
        g_dist = self.mse_loss(gen_y, x_hat)
        g_preds = g_dist.mean(dim=1)
        return g_preds

    def video_predict_coop(self, x):
        x_hat, gen_y, dis_pred = self(x)
        # generator predictions
        g_dist = self.mse_loss(gen_y, x_hat)
        gen_pred = g_dist.mean(dim=1)
        return gen_pred, dis_pred


    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.6)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step)
        return [optimizer], [lr_scheduler]
    
    def training_step_ae(self, batch, batch_idx):
        x, label = batch
        x_hat, gen_y = self(x)
        g_dist = self.mse_loss(gen_y, x_hat)

        batch_dist = g_dist.mean(dim=1)
        g_thresh = self.get_gen_thresh(batch_dist)
        g_preds = (batch_dist > g_thresh).float()

        # generator loss
        loss = batch_dist.mean()
        g_auc = auroc(g_preds, label, num_classes=None)

        self.log('train/g_auc', g_auc, prog_bar=True)
        self.log('train/loss', loss)
        return loss

    def training_step_coop(self, batch, batch_idx):
        x, label = batch
        x_hat, gen_y, dis_y = self(x)
        g_loss, d_loss = self.gen_dis_loss(x_hat, gen_y, dis_y)
        loss = g_loss + d_loss

        self.log('train/g_loss', g_loss, prog_bar=True)
        self.log('train/d_loss', d_loss, prog_bar=True)
        self.log('train/loss', loss)
        return loss
    
    def validation_step_ae(self, batch, batch_idx):
        x, label = batch
        batch_dist = self.video_predict(x)

        g_thresh = self.get_gen_thresh(batch_dist)
        g_preds = (batch_dist > g_thresh).float()
        # generator loss
        loss = batch_dist.mean()
        g_auc = auroc(g_preds, label, num_classes=None)

        self.log("test/g_mse", loss, prog_bar=True)
        self.log("test/g_auc", g_auc, prog_bar=True)
        return g_auc

    def validation_step_coop(self, batch, batch_idx):
        x, label = batch
        gen_pred, dis_pred = self.video_predict(x)
        loss = gen_pred.mean()
        gen_auc, dis_auc = self.val_auc_loss(gen_pred, dis_pred, label)

        self.log("test/g_mse", loss, prog_bar=True)
        self.log("test/g_auc", gen_auc, prog_bar=True)
        self.log("test/d_auc", dis_auc, prog_bar=True)
        
        return dis_auc
    
    def get_feat_size(self):
        dataset = ClipDataset(self.data_dir,
                              feat_model=self.feat_model,
                              )
        return len(dataset[0][0])
    
    def train_dataloader(self):
        dataset = ClipDataset(self.data_dir,
                              feat_model=self.feat_model,
                              clean=self.self_pretrain,
                              split='train',
                              )
                                      
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

    
    def val_dataloader(self):
        dataset = ClipDataset(self.data_dir,
                              feat_model=self.feat_model,
                              split='test',
                              )
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True)


# Generative Cooperative Learning Framework

# Generator model
class Autoencoder(nn.Module):
    def __init__(self, feat_size=512):
        super(Autoencoder, self).__init__()
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
                            nn.Linear(256, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Linear(512, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Linear(1024, feat_size)
                            )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, feat_size=512):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                            nn.Linear(feat_size, 1024),
                            nn.BatchNorm1d(1024),
                            nn.LeakyReLU(),
                            nn.Linear(1024, 512),
                            nn.BatchNorm1d(512),
                            nn.LeakyReLU(),
                            nn.Linear(512, 32),
                            nn.BatchNorm1d(32),
                            nn.LeakyReLU(),
                            nn.Linear(32, 1)
                            )
        # self.act = nn.Softmax(dim=1)
        self.act = nn.Sigmoid()        
        
    def forward(self, x):
        x = self.model(x)
        x = self.act(x)
        return torch.flatten(x)
    
        
if __name__ == "__main__":
    from time import time
    import os

    print('Hello there!')

    p = 16
    batch_s = 2
    x = torch.rand([batch_s,512])
    
    
    feat_models_3D = ['rx101', 'r3d18', 'r3d50']

    # feat_model = 'rx3D'
    feat_model = feat_models_3D[1]
    # depth = 18; feat_model=f'r3D{depth}'

    dataset_root: str = '../data/'
    dataset: str = 'shanghai'
    
    
    model = GCLModel(
        path=dataset_root + dataset,
        feat_model=feat_model,
        batch_size=batch_s,
        ).cuda()
    
    tic = time()
    x_hat, g_y, d_y = model(x.cuda())
    print(time()-tic)
    print(x_hat.shape)
    print(g_y.shape)
    print(d_y.shape)
    
    print('Obiwan Kenobi!')
