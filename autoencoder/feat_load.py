#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:26:52 2022

@author: anil
"""



import os
import numpy as np

from tqdm import tqdm, trange
import pandas as pd

import torch


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def normalize(preds):
    return (preds-min(preds))/(max(preds)-min(preds))



class ClipDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_dir='../data/shanghai',
                 clip_len=16,
                 feat_model='resnet18',
                 split='test',
                 clean=False,
                 split_folders=False,
                 ):
        assert split in ['test', 'train'], 'split type can be train or test'
        self.clean=clean
        self.split = split
        self.clip_len = clip_len
        self.dir_name = dataset_dir
        self.dataset_name = dataset_dir.split('/')[-1]
        self.feat_model = feat_model
        feat_models_3D = ['rx101', 'r3d18', 'r3d50']
        if feat_model.lower() in feat_models_3D:
            # features from 3d convolutinal network
            if split_folders:
                self.load_clip = self.load_clip_3d_split_folders
            else:
                self.load_clip = self.load_clip_3d
        else:
            self.load_clip = self.load_clip_frames
        # labelled header load
        self.header_df = pd.read_csv(self.dir_name + f'/splits_header_{split}.csv')
        # self cleaning part adding
        if clean:
            print('self cleaning dataset.')
            indexes_path = f'{self.dir_name}/clean_indexes_{self.feat_model}_{self.split}.npy'
            if os.path.isfile(indexes_path):
                print('using precomputed indexes.')
                clean_indexes = np.load(indexes_path)
                self.header_df = self.header_df.iloc[clean_indexes]
            else:
                print('computing indexes.')
                clean_indexes = self.generate_self_cleaning_indexes()
                self.header_df = self.header_df.iloc[clean_indexes]
                
        self.lenght = len(self.header_df)

    def __len__(self):
        return self.lenght

    def __getitem__(self, index):
        return self.load_clip(index)

    def load_clip_3d_split_folders(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        
        label = np.array(label, dtype=np.int)
        
        feat_path = f'{self.dir_name}/features/{self.feat_model}_{self.split}/{video_id}/{start}.npy'
        clip = np.load(feat_path)
        
        return clip, label

    def load_clip_3d(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        
        label = np.array(label, dtype=np.int)
        
        feat_path = f'{self.dir_name}/features/{self.feat_model}/{video_id}/{start}.npy'
        clip = np.load(feat_path)
        
        return clip, label
    
    def load_clip_frames(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
               
        clip = []
        for i in range(start, end + 1):
            feat_path = f'{self.dir_name}/features2D/{self.feat_model}/{video_id}/{i}.npy'
            feat = np.load(feat_path)
            clip.append(feat)
        clip = np.stack(clip, axis=0)
        
        return clip, label

    
    

    
if __name__ == '__main__':
    print('Hello there!')
    

    feat_models_3D = ['rx101', 'r3d18', 'r3d50']
    feat_model = feat_models_3D[1]

    dataset_root: str = '../data/'
    dataset: str = 'shanghai'


    cds = ClipDataset(dataset_dir=dataset_root + dataset,
                      clip_len=16,
                      feat_model=feat_model,
                      split='test',
                      clean=False,)

    print(len(cds.header_df))
    print(cds.header_df.keys())
    
    
    print('Obiwan Kenobi!')