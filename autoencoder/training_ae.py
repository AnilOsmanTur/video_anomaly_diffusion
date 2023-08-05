#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 17:00:43 2022

@author: anil
"""

# import os
from glob import glob
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from feat_models import GCLModel
import time

seed_everything(42, workers=True)
# seed_everything(time.time(), workers=True)


def main_training(dataset_dir='',
                  feat_model='',
                  accelerator='',
                  devices_u='',
                  worker_n=6):

    dataset = dataset_dir.split('/')[-1]
    logger_path = f'runs/ae_all_{dataset}'
    logger = TensorBoardLogger(logger_path, name=f'model_{feat_model}')
    
    print('unsupervised ae training')
    model = GCLModel(path=dataset_dir,
                    feat_model=feat_model,
                    batch_size=8192,
                    lr=2e-5,
                    lr_step=20,
                    worker_n=worker_n,
                    self_pretrain=False,
                    coop=False)

    trainer = Trainer(accelerator=accelerator,
                    devices=devices_u,
                    logger=logger,
                    max_epochs=20)
    
    trainer.fit(model)
    


if __name__ == "__main__":
    from time import time
    import os
    
    print('Hello there!')
    
    accelerator='gpu'
    gpu_id = 0
    devices = [gpu_id] if accelerator == 'gpu' else 'auto'
    
    feat_models_3D = ['rx101', 'r3d18', 'r3d50']

    

    dataset_root: str = '../data/'
    dataset: str = 'shanghai'
    worker_n = 6

    for feat_model in feat_models_3D:
        print(feat_model)
        tic = time()
        main_training(dataset_dir=dataset_root + dataset,
                    feat_model=feat_model,
                    accelerator=accelerator,
                    devices_u=devices,
                    worker_n=worker_n)
        print('total run time:', time()-tic)

    
    print('Obiwan Kenobi!')