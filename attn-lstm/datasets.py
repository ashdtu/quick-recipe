# Dependancies
import torch
from torch.utils.data import Dataset
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
#         self.h = h5py.File(os.path.join(data_folder, self.split +  '_IMAGES_' + data_name + '.hdf5'), 'r')
        df = pd.read_pickle(Path(data_folder) / 'full_master_updated.pkl')
        cond = (df['Split'] == split.lower()) & (df['IsUsefulSentence'] == 1)
        self.df = df[cond]
        self.imgs = self.df['img_feat'].values

        # captions per image 
#         self.cpi = self.h.attrs['captions_per_image']
        self.cpi = 1

        # Load encoded captions (completely into memory)
#         with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
        self.captions = self.df['text_encodings(Attn-LSTM)'].values

        # Load caption lengths (completely into memory)
        # with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
        self.caplens = self.df['text_encodings_len(Attn-LSTM)'].values
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)


    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the ( N // captions_per_image)th image
        # try:
                        
        img = torch.FloatTensor(np.asarray(self.imgs[i], dtype=np.float32)).reshape(4, 4, 32) # 512 dim
        # Apply the transform to the image
        if self.transform is not None:
                img = self.transform(img)

        caption = torch.LongTensor(np.asarray(self.captions[i], dtype=np.int32))
        
        caplen = torch.LongTensor([self.caplens[i]])
        # except Exception as e:
        #         print(e, self.imgs[i], i)
        

        if self.split is 'TRAIN':
                return img, caption, caplen

        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        #     all_captions = torch.LongTensor(
        #         self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            all_captions = torch.LongTensor([np.asarray(self.captions[i], dtype=np.int32)])
            return img, caption, caplen, all_captions


    def __len__(self):
        return self.dataset_size

class CaptionDatasetMultimodal(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
#         self.h = h5py.File(os.path.join(data_folder, self.split +  '_IMAGES_' + data_name + '.hdf5'), 'r')
        df = pd.read_pickle(Path(data_folder) / 'full_master_updated.pkl')
        cond = (df['Split'] == split.lower()) & (df['IsUsefulSentence'] == 1)
        self.df = df[cond]
        self.imgs = self.df['img_feat'].values

        # captions per image 
#         self.cpi = self.h.attrs['captions_per_image']
        self.cpi = 1

        # Load encoded captions (completely into memory)
#         with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
        self.captions = self.df['text_encodings(Attn-LSTM)'].values

        # Load caption lengths (completely into memory)
        # with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
        self.caplens = self.df['text_encodings_len(Attn-LSTM)'].values

        self.text_embs = self.df['text_feat(all-MiniLM-L6-v2)'].values
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)


    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the ( N // captions_per_image)th image
        # try:
                        
        img = torch.FloatTensor(np.asarray(self.imgs[i], dtype=np.float32)).reshape(4, 4, 32) # 512 dim
        # Apply the transform to the image
        if self.transform is not None:
                img = self.transform(img)

        text_embs = torch.FloatTensor(np.asarray(self.text_embs[i], dtype=np.float32)).reshape(4, 4, 24) # 384 dim
        combined_embs = torch.cat([img, text_embs], dim=2)
        caption = torch.LongTensor(np.asarray(self.captions[i], dtype=np.int32))
        caplen = torch.LongTensor([self.caplens[i]])
        # except Exception as e:
        #         print(e, self.imgs[i], i)
        

        if self.split is 'TRAIN':
                return combined_embs, caption, caplen

        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        #     all_captions = torch.LongTensor(
        #         self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            all_captions = torch.LongTensor([np.asarray(self.captions[i], dtype=np.int32)])
            return combined_embs, caption, caplen, all_captions


    def __len__(self):
        return self.dataset_size