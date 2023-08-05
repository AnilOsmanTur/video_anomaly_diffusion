#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:13:38 2022

@author: anil
"""

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset



class ClipDataset(Dataset):
    
    def __init__(self,
                 root_dir='data',
                 dataset_name='shanghai',
                 ):
        
        self.dataset_name = dataset_name
        self.path = f'{root_dir}/{self.dataset_name}'
        self.header_df = pd.read_csv(self.path+'/splits_header.csv')
        self.lenght = len(self.header_df)

    
    def __len__(self):
        return self.lenght
    
    def __getitem__(self, index):
        return self.load_clip(index)
    
    def load_clip(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        img_path = f'{self.path}/features/{video_id}/{start}.npy'
        clip = torch.from_numpy(np.load(img_path))
        label = np.array(label, dtype=np.int32)
        return {'vid_id': video_id, 'idx': start, 'label': label, 'data': clip}




if __name__ == '__main__':
    print('Hello there!')


    ds = ClipDataset(root_dir='data',
                    dataset_name='shanghai')
    print(len(ds))
    clip = ds[1]['data']
    print(clip.shape)


    
    print('Obiwan Kenobi!')
