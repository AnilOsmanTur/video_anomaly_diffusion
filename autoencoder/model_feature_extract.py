#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:32:49 2022

@author: anil
"""

import torch

from ResNext_models import resnet, resnext

import os
import argparse
import numpy as np
from tqdm import trange, tqdm

from data_load import VideoDataset, VideoClipDataset
from torch.utils.data import DataLoader, Subset
from collections import defaultdict


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    

def make_batched_video(x, p=16):
    batch = []
    start = 0
    for i in range(p, len(x)+1):
        b = x[start:i]
        start += 1
        batch.append(b)
    batch = torch.stack(batch, dim=0)
    return batch



def load_resnet3D(depth=50):
    model = resnet.generate_model(model_depth=depth,
                                  n_classes=1039,
                                  n_input_channels=3,
                                  shortcut_type='B',
                                  conv1_t_size=7,
                                  conv1_t_stride=1,
                                  no_max_pool=False,
                                  widen_factor=1.0)
    
    
    PATH = f'ResNext_models/trained/r3d{depth}_KM_200ep.pth'
    checkpoint = torch.load(PATH, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model


    
def key_name_change(cp_dict):
    print('dict key change.')
    new_keys = []
    old_keys = []
    for key in cp_dict:
        new_keys.append('.'.join(key.split('.')[1:]))
        old_keys.append(key)
    
    for idx in range(len(cp_dict)):
        cp_dict[new_keys[idx]] = cp_dict[old_keys[idx]]
        del cp_dict[old_keys[idx]]

def load_resnext3D():
    model = resnext.generate_model(model_depth=101,
                                   n_classes=400,
                                   cardinality=32,
                                   n_input_channels=3,
                                   shortcut_type='B',
                                   conv1_t_size=7,
                                   conv1_t_stride=1,
                                   no_max_pool=False)
    
    
    PATH = 'ResNext_models/trained/resnext-101-kinetics.pth'
    checkpoint = torch.load(PATH, map_location='cpu')
    cp_dict = checkpoint['state_dict']
    key_name_change(cp_dict)
    model.load_state_dict(cp_dict)
    return model




def main_shangai():
    gpu_id = 0
    clip_len = 16
    
    depth = 18; feat_model = f'r3D{depth}'; model = load_resnet3D(depth)
    # feat_model = 'rx3D'; model = load_resnext3D()
    
    batch_size = 128
    
    model = model.eval()
    model = model.cuda(gpu_id)
    
    with torch.no_grad():
        
        root_dir='../data'
        
        ds = VideoDataset(root_dir=root_dir,
                          dataset_name='shanghai')
        dst_data_path = ds.path + f'_{feat_model}_vfeat'

        mkdir(dst_data_path)
        
        for idx in trange(len(ds)):
            data = ds[idx]
            clip_id = data['clip_id']
            x = data['data']
            
            batched = make_batched_video(x, clip_len)
            outs = []
            for i in range(0,len(batched),batch_size):
                x = batched[i:i+batch_size]
                x = x.permute((0,2,1,3,4))
                x = x.cuda(gpu_id)
                out = model(x)
                outs.append(out.cpu())
            outs = torch.cat(outs, dim=0).numpy()
            np.save(f'{dst_data_path}/{clip_id}', outs)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch_size', type=int, default=60,
                   help='the batch size')
    p.add_argument('--gpu_id', type=int, default=0,
                   help='gpu id to choose the gpu')
    p.add_argument('--split_id', type=int, default=0,
                   help='split id to choose the split')
    p.add_argument('--split_n', type=int, default=1,
                   help='n split to divide')
    p.add_argument('--split_name', type=str, default='train',
                   help='dataset split to process')
    p.add_argument('--reminders', action='store_true', default=False,
                   help='reminder switch')
    return p.parse_args()



if __name__ == "__main__":
    # from time import time
    print('Hello there!')
    args = args_parser()
    
    gpu_id = args.gpu_id
    
    split_id = args.split_id
    split_n = args.split_n
    
    clip_len = 16
    split_name = args.split_name
    reminders = args.reminders
    
    depth = 18; feat_model = f'r3D{depth}'; model = load_resnet3D(depth)
    # feat_model = 'rx3D'; model = load_resnext3D()
    
    batch_size = args.batch_size
    
    model = model.eval()
    model = model.cuda(gpu_id)
    
    with torch.no_grad():
        
        root_dir = '../data'
        
        
        ds = VideoClipDataset(root_dir=root_dir,
                              dataset_name='UCFC',
                              split=split_name,
                              load_reminder=reminders)
        
        dst_data_path = ds.path + f'/features/{feat_model}_{split_name}'
        mkdir(dst_data_path)
        
        # generating split
        video_ids = list(ds.vid_header_df['video_id'])
        file_names = [x.split('.')[0] for x in os.listdir(dst_data_path)]
        
        for name in file_names:
            video_ids.remove(name)
        video_ids = np.array(video_ids)
        
        n_vid = len(video_ids)
        indexes = np.arange(n_vid)
        
        np.random.seed(42)
        np.random.shuffle(indexes)
        splits = np.array_split(indexes, split_n)
        
        clip_ids = ds.header_df['video_id'].values
        mask = np.zeros_like(clip_ids, dtype=np.bool8)
        print('split generation')
        for name in tqdm(video_ids[splits[split_id]]):
            mask += clip_ids == name
        indexes = np.where(mask)[0]
        
        if not mask.sum() > 0 and reminders:
            print('second split generation')
            n_ds = len(ds)
            indexes = np.arange(n_ds)
            splits = np.array_split(indexes, split_n)
            indexes = splits[split_id]

        chosen_subset = Subset(ds, indexes)
        loader = DataLoader(chosen_subset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=6,
                            drop_last=False,
                            prefetch_factor=2)
        
        
        result_dict = defaultdict(list)
        for i, data in enumerate(tqdm(loader)):
            clip_ids = data['clip_id']
            starts = data['start']
            x = data['data']
            
            x = x.permute((0,2,1,3,4))
            x = x.cuda(gpu_id)
            outputs = model(x).cpu()
            
            
            for clip_id, start, out in zip(clip_ids, starts, outputs):
                path = f'{dst_data_path}/{clip_id}'
                mkdir(path)
                np.save(f'{path}/{start}.npy', out)
            
            
    print('Obiwan Kenobi!')