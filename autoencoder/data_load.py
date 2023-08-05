#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:13:38 2022

@author: anil
"""

import os
import numpy as np

from tqdm import tqdm, trange
import pandas as pd
from PIL import Image
import imageio.v3 as iio

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# action recognition pre-trained on Kinetics-400 normalization
# normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
#                                  std=[0.22803, 0.22145, 0.216989])

# classification pre-trained on ImageNet normalization
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

# example transform composition
# transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])


class ClipDataset(Dataset):
    
    def __init__(self,
                 root_dir='../Datasets/avenue_peds_shanghai',
                 k=1,
                 clip_len=16,
                 transform_fn=None):
        assert k < 4 , 'k must be in between [0,3]'
        datasets = ['avenue', 'shanghai', 'ped1', 'ped2']
        self.dataset_name = datasets[k]
        self.path = f'{root_dir}/{self.dataset_name}'
        
        im_resize = transforms.Resize((128, 171),
                                      interpolation=transforms.InterpolationMode.BICUBIC,
                                      antialias=True)
        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                         std=[0.22803, 0.22145, 0.216989])
        
        if transform_fn:
            self.transform_fn = transforms.Compose([
                im_resize,
                transforms.ToTensor(),
                transform_fn,
                normalize,
                ])
        else:
            self.transform_fn = transforms.Compose([
                im_resize,
                transforms.ToTensor(),
                normalize,
                ])
            
        self.clip_len = clip_len
        self.header_df = pd.read_csv(self.path+'/splits_header.csv')
        self.lenght = len(self.header_df)
    
    def __len__(self):
        return self.lenght
    
    def __getitem__(self, index):
        return self.load_clip(index)
    
    def load_clip(self, idx):
        video_id, start, end = self.header_df.iloc[idx]
        clip = []
        for i in range(start, end+1):
            img_path = f'{self.path}/{video_id}/{i}.jpg'
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = self.transform_fn(image)
            clip.append(image)
        while len(clip) != self.clip_len:
            clip.append(image)
        return {'clip_id': f'{video_id}_{idx}', 'data': torch.stack(clip, dim=0)}

    
class VideoDataset(Dataset):
    
    def __init__(self,
                 root_dir='../data',
                 dataset_name='shanghai',
                 transform_fn=None):
        if 'shanghai' == dataset_name:
            self.load_clip = self.load_clip_shanghai
        elif 'UCFC' == dataset_name:
            self.load_clip = self.load_clip_ucfc
            
        self.path = f'{root_dir}/{dataset_name}'
        self.header_df = pd.read_csv(self.path+'/header.csv')
        self.lenght = len(self.header_df)
        
        im_resize = transforms.Resize((256, 256),
                                      interpolation=transforms.InterpolationMode.BICUBIC,
                                      antialias=True)
        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                         std=[0.22803, 0.22145, 0.216989])
        
        if transform_fn:
            self.transform_fn = transforms.Compose([
                im_resize,
                transforms.ToTensor(),
                transform_fn,
                normalize,
                ])
        else:
            self.transform_fn = transforms.Compose([
                im_resize,
                transforms.ToTensor(),
                normalize,
                ])
        
    
    def __len__(self):
        return self.lenght
    
    def __getitem__(self, index):
        return self.load_clip(index)
    
    
    def load_clip_ucfc(self, idx):
        video_id, label, frame_count, video_path = self.header_df.iloc[idx][['video_id', 'label', 'frame_count', 'video_path']]
        file_path = os.path.join(self.path, video_path)
        video = []
        with iio.get_reader(file_path,  'ffmpeg', 'I') as vid_reader:
            for image in vid_reader:
                image = Image.fromarray(image)
                image = self.transform_fn(image)
                video.append(image)
                
        return {'clip_id': video_id, 'label':label, 'data': torch.stack(video, dim=0)}

    
    def load_clip_shanghai(self, idx):
        video_id, label, frame_count = self.header_df.iloc[idx][['video_id', 'label', 'frame_count']]
        video = []
        for i in range(frame_count):
            img_path = f'{self.path}/{video_id}/{i}.jpg'
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = self.transform_fn(image)
            video.append(image)
        return {'clip_id': video_id, 'label':label, 'data': torch.stack(video, dim=0)}


class VideoClipDataset(Dataset):
    def __init__(self,
                 root_dir='../data',
                 dataset_name='UCFC',
                 clip_len=16,
                 split='test',
                 load_reminder=False,
                 transform_fn=None):
        assert dataset_name in ['UCFC', 'shanghai'], 'dataset_name must be UCFC or shanghai'

        self.path = f'{root_dir}/{dataset_name}'
        self.clip_len = clip_len
        self.vid_header_df = pd.read_csv(self.path+f'/header_{split}.csv')
        if load_reminder:
            self.header_df = pd.read_csv(self.path+f'/reminders_{split}.csv')
        else:
            self.header_df = pd.read_csv(self.path+f'/splits_header_{split}.csv')
        self.lenght = len(self.header_df)
        
        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                         std=[0.22803, 0.22145, 0.216989])
        
        if transform_fn:
            self.transform_fn = transforms.Compose([
                transforms.ToTensor(),
                transform_fn,
                normalize,
                ])
        else:
            self.transform_fn = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])

    def __len__(self):
        return self.lenght

    def __getitem__(self, index):
        return self.load_clip(index)

    def load_clip(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        file_path = os.path.join(self.path,'frames', video_id)

        clip = []
        for i in range(start, end):
            img_path = f'{file_path}/{i}.jpg'
            with Image.open(img_path) as image:
                image = self.transform_fn(image)
            clip.append(image)
        return {'clip_id': video_id, 'start': start, 'data': torch.stack(clip, dim=0)}


def make_viedo_batch(x, p=16):
    batch = []
    start = 0
    for i in range(p, len(x)+1):
        b = x[start:i]
        start += 1
        batch.append(b)
    batch = torch.stack(batch, dim=0)
    return batch


if __name__ == '__main__':
    print('Hello there!')
    

    root_dir='../data'

    ds = VideoDataset(root_dir=root_dir)
    
    clip = ds[0]['data']

    print(clip.shape)

    print('Obiwan Kenobi!')
