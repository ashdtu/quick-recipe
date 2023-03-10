import errno
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Any, Optional, Callable

class YouCookDataset(Dataset):
    #TODO - Add download feature
    #url = 'http://youcook2.eecs.umich.edu/static/YouCookII/YouCookII.tar.gz'

    #path based on file format
    path = 'features/feat_{format}/{phase}_frame_feat_{format}'
    phases = ['train', 'val', 'test']


    def __init__(self, annotation_file, 
        root:str,
        label_file: str,
        phase: int = 1,  #1 for train, #2 for validation, #3 for test
        file_format: str = 'csv', #csv(default) and dat format supported
        transform: Optional[Callable] = None, #used for transforming numerical video data to a required format
        download: bool = False) -> None:
        # Download case not handled for now.
        # It will throw error if the relevant files are not found.
        
        #annotation files
        annotation_path = os.path.join(root, annotation_file)
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                    annotation_path)
        #label files 
        label_path = os.path.join(root, label_file)
        if not os.path.exists(label_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                    label_path) #raise the correct error

        #self.annotations = pd.read_csv(annotation_path)
        #self.labels = pd.read_csv(label_path)
        self.labels = pd.DataFrame()
        #data files
        phase = self.phases[phase]
        data_path = self.path.format(format = file_format, phase = phase)
        data_path = os.path.join(root, data_path)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                    data_path) #raise the correct error

        print(data_path)
        #file_list = list(os.walk(f'{data_path}'))
        file_list = glob.glob(f'{data_path}/**/*.{file_format}', recursive=True)
        print(file_list)
        data: Any = []
        for file_path in file_list:
            vid_data = np.genfromtxt(file_path, delimiter=',')
            data.append(vid_data)
        data_np = np.concatenate([data], axis = 0)
        self.data = torch.from_numpy(data_np)


    def __getitem__(self, index) -> Tuple[Any, Any]:
        vid, label = self.data[index], self.labels[index]

        if self.transform is not None:
            vid = self.transform(vid)

        return vid, label

    def __len__(self) -> int:
        return self.data.size()[0]