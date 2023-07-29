import os
import numpy as np
import torch
import torch.utils.data

from utils.utils import pil_loader 

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, transforms=None):
        self.args = args
        
    def __getitem__(self, idx):
        return 

    def __len__(self):
        return len(self.imgs)


