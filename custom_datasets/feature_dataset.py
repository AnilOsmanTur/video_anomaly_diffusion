from multiprocessing import Pool
from tqdm import trange, tqdm
import pandas as pd
import numpy as np
from time import time
import torch
import os
import pdb
import h5py

# Batch = namedtuple('Batch', 'clip cond')
# ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class ClipDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_dir='/Datasets/avenue_peds_shanghai/shanghai',
                 feat_dir='',
                 clip_len=16,
                 feat_model='resnet18',
                 split='test',
                 clean=False,
                 split_folders=False,
                 check=False,
                 no_overlap=False,
                 anomaly=False,
                 preload=False,
                 hd5=False,
                 ):

        assert split in ['test', 'train'], 'split type can be train or test'
        self.clean = clean
        self.split = split
        self.clip_len = clip_len
        self.header_dir = dataset_dir
        self.dir_name = feat_dir if feat_dir else dataset_dir
        self.dataset_name = dataset_dir.split('/')[-1]
        self.feat_model = feat_model
        self.hd5 = hd5
        feat_models_3D = ['rx3d', 'r3d18', 'r3d50', 'rx101']
        if hd5:
            self.f = h5py.File(f'{self.dir_name}/data_{feat_model}_{split}.hdf5', 'r')
            self.data = self.f['data']

        if feat_model.lower() in feat_models_3D:
            # features from 3d convolutinal network
            if split_folders:
                self.load_clip = self.load_clip_3d_split_folders
                if check:
                    self.load_clip = self.check_clip
            else:
                self.load_clip = self.load_clip_3d
        else:
            self.load_clip = self.load_clip_frames
        # labelled header load
        self.header_df = pd.read_csv(self.header_dir + f'/splits_header_{split}.csv')

        if no_overlap:
            indexes = self.header_df['stride'] == 0
            self.header_df = self.header_df[indexes]

        if anomaly:
            indexes = self.header_df['label'] == 1
            self.header_df = self.header_df[indexes]

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

        if preload:
            sample = self.load_clip(0)[0]
            feat_size = sample.shape[0]
            print(sample.shape)
            self.data = None
            # self.data = np.zeros((self.lenght, feat_size), dtype=np.float32)
            tic = time()
            self.preload_data()
            print('preloading time:', time()-tic)
            self.load_clip = self.pre_loaded_clip

    def __del__(self):
        if self.hd5:
            self.f.close()

    def __len__(self):
        return self.lenght

    def __getitem__(self, index):
        return self.load_clip(index)

    def pre_loaded_clip(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        label = np.array(label, dtype=np.int32)
        clip = self.data[idx]
        return clip, label, video_id, start

    def preload_fn(self, idx):
        return self.load_clip(idx)[0]

    def preload_data(self):
        print('preloading start..')
        with Pool(6) as p:
            self.data = np.stack(p.map(self.preload_fn, list(range(self.lenght))), axis=0)
        print(self.data.shape)
        print('preloading finished..')

    def check_clip(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]

        feat_path = f'{self.dir_name}/features/{self.feat_model}_{self.split}/{video_id}/{start}.npy'
        return os.path.isfile(feat_path)

    def load_clip_3d_split_folders(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]

        label = np.array(label, dtype=np.int32)
        if self.hd5:
            clip = self.data[idx]
        else:
            feat_path = f'{self.dir_name}/features/{self.feat_model}_{self.split}/{video_id}/{start}.npy'
            clip = np.load(feat_path)

        return clip, label, video_id, start

    def load_clip_3d(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]

        label = np.array(label, dtype=np.int32)
        if self.hd5:
            clip = self.data[idx]
        else:
            # feat_path = f'{self.dir_name}/{self.feat_model}/{video_id}/{start}.npy'
            feat_path = f'{self.dir_name}/features3D/{self.feat_model}/{video_id}/{start}.npy'
            # feat_path = f'{self.dir_name}/features/{self.feat_model}/{video_id}/{start}.npy'
            clip = np.load(feat_path)

        return clip, label, video_id, start

    def load_clip_frames(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]

        clip = []
        for i in range(start, end + 1):
            feat_path = f'{self.dir_name}/features2D/{self.feat_model}/{video_id}/{i}.npy'
            feat = np.load(feat_path)
            clip.append(feat)
        clip = np.stack(clip, axis=0)

        return clip, label

    def generate_self_cleaning_indexes(self):
        video_ids = self.header_df['video_id'].values
        strides = self.header_df['stride'].values

        dif = torch.zeros(len(video_ids))
        past_stride = strides[0]
        past_vid = video_ids[0]
        past_feat = self[0][0]
        idx_mask = torch.zeros(len(video_ids))
        for idx in trange(1, len(video_ids)):
            if past_vid == video_ids[idx] and past_stride == strides[idx]:
                curr_feat = self[idx][0]
                dif[idx] = torch.dist(torch.from_numpy(past_feat), torch.from_numpy(curr_feat), 2)
                past_feat = curr_feat
                past_stride = strides[idx]
                past_vid = video_ids[idx]
                idx_mask[idx] = 1
            else:
                past_feat = self[idx][0]
                past_vid = video_ids[idx]
                past_stride = strides[idx]

        idx_mask = idx_mask.numpy()
        clean_indexes = []
        last_idx = len(idx_mask)
        part_starts = np.where(1 > idx_mask)[0]
        part_ends = np.zeros_like(part_starts)
        part_ends[:-1] = part_starts[1:]
        part_ends[-1] = last_idx
        parts = np.stack([part_starts, part_ends]).T
        for start, end in tqdm(parts):
            clip = dif[start:end]
            clip[0] = clip[1]
            std, mean = torch.std_mean(clip)
            dif_thresh = abs(mean + .1 * std)
            clean_indexes.append(clip < dif_thresh)
        clean_indexes = np.concatenate(clean_indexes)

        np.save(f'{self.dir_name}/clean_indexes_{self.feat_model}_{self.split}.npy', clean_indexes)
        return clean_indexes

