#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:13:38 2022

@author: anil
"""

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms



class ClipDataset(Dataset):
    
    def __init__(self,
                 root_dir='data',
                 dataset_name='shanghai',
                 clip_len=16
                 ):
        
        self.dataset_name = dataset_name
        self.path = f'{root_dir}/{self.dataset_name}'
        self.clip_len = clip_len
        self.header_df = pd.read_csv(self.path+'/splits_header.csv')
        self.lenght = len(self.header_df)
            
        self.transform_fn = transforms.Compose([
            transforms.Resize((256, 256),
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989]),
            ])
            
        
    def __len__(self):
        return self.lenght
    
    def __getitem__(self, index):
        return self.load_clip(index)
    
    def load_clip(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        clip = []
        for i in range(start, end+1):
            img_path = f'{self.path}/frames/{video_id}/{i}.jpg'
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = self.transform_fn(image)
            clip.append(image)
        return {'clip_id': f'{video_id}_{start}', 'label': label, 'data': torch.stack(clip, dim=0)}




if __name__ == '__main__':
    print('Hello there!')


    ds = ClipDataset(root_dir='data',
                    dataset_name='shanghai')
    print(len(ds))
    clip = ds[1]['data']
    print(clip.shape)


    
    print('Obiwan Kenobi!')
