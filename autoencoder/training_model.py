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

seed_everything(42, workers=True)

def main_no_neg(dataset_dir='',
                feat_model='',
                accelerator='',
                devices_u='',
                worker_n=6
                ):

    print('No negative learning Colearning training')
    # normal training
    model = GCLModel(path=dataset_dir,
                     feat_model=feat_model,
                     batch_size=8192,
                     lr=2e-5,
                     lr_step=15,
                     worker_n=worker_n,
                     self_pretrain=False,
                     coop=True,
                     neg_learning=False)

    dataset = dataset_dir.split('/')[-1]
    logger_path = f'runs/no_neg_{dataset}'
    logger = TensorBoardLogger(logger_path, name=f'model_{feat_model}')

    trainer = Trainer(accelerator=accelerator,
                      devices=devices_u,
                      logger=logger,
                      max_epochs=15)

    trainer.fit(model)

def main_GCL(dataset_dir='',
             feat_model='',
             accelerator='',
             devices_u='',
             self_supervised=False,
             worker_n=6,
             pass_pretrain=False):

    dataset = dataset_dir.split('/')[-1]
    # logger_path = f'runs/{depth}' if self_supervised else f'runs/{depth}_base'
    # logger = TensorBoardLogger(logger_path, name=f'model_{feat_model}')
    
    if self_supervised and not pass_pretrain:
        logger_path = f'runs/pretrain_{dataset}'
        logger = TensorBoardLogger(logger_path, name=f'model_{feat_model}')
        
        print('Selfsupervised ae pretraining')
        model = GCLModel(path=dataset_dir,
                         feat_model=feat_model,
                         batch_size=8192,
                         lr=2e-5,
                         lr_step=15,
                         worker_n=worker_n,
                         coop=True,
                         self_pretrain=True)

        trainer = Trainer(accelerator=accelerator,
                      devices=devices_u,
                      logger=logger,
                      max_epochs=15)
        
        trainer.fit(model)
    

    print('Colearning training')
    # normal training
    model = GCLModel(path=dataset_dir,
                     feat_model=feat_model,
                     batch_size=8192,
                     lr=2e-5,
                     lr_step=15,
                     worker_n=worker_n,
                     self_pretrain=False,
                     coop=True,
                     neg_learning=True)

    if self_supervised:
        logger_path = f'runs/pretrain_{dataset}'
        PATH = glob(f'{logger_path}/model_{feat_model}/**/**/*.ckpt')[-1]
        model.load_state_dict(torch.load(PATH)['state_dict'])
    
    logger_path = f'runs/gcl_{dataset}' if self_supervised else f'runs/gcl_base_{dataset}'
    logger = TensorBoardLogger(logger_path, name=f'model_{feat_model}')

    trainer = Trainer(accelerator=accelerator,
                      devices=devices_u,
                      logger=logger,
                      max_epochs=15)

    trainer.fit(model)


if __name__ == "__main__":
    from time import time
    import os
    
    print('Hello there!')
    
    accelerator='gpu'
    gpu_id = 0
    devices = [gpu_id] if accelerator == 'gpu' else 'auto'
    
    feat_models_3D = ['rx101', 'r3d18', 'r3d50']

    dataset_root: str = '../data'
    dataset: str = 'shanghai'
    worker_n = 6

    # root_data_path = '../Datasets'
    for feat_model in feat_models_3D:
        print(feat_model)
        tic = time()
        main_no_neg(dataset_dir=dataset_root + dataset,
                    feat_model=feat_model,
                    accelerator=accelerator,
                    devices_u=devices,
                    worker_n=worker_n)
        print('total run time:', time()-tic)
    
        
        tic = time()
        main_GCL(dataset_dir=dataset_root + dataset,
                 feat_model=feat_model,
                 accelerator=accelerator,
                 devices_u=devices,
                 self_supervised=False,
                 worker_n=worker_n,
                 pass_pretrain=False)
        print('total run time:', time()-tic)
    
        tic = time()
        main_GCL(dataset_dir=dataset_root + dataset,
                 feat_model=feat_model,
                 accelerator=accelerator,
                 devices_u=devices,
                 self_supervised=True,
                 worker_n=worker_n,
                 pass_pretrain=False)
        print('total run time:', time()-tic)
    
    print('Obiwan Kenobi!')
