import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from datetime import datetime


class MashupDataset(Dataset):
    """

    """
    def __init__(
        self,
        invocation: pd.DataFrame,
        num_candidates: int,
        mashup_index: bool,
        mashup_transform=None,
        api_transform=None,
        is_orderly: bool = False,
        is_triple: bool = False,
        negative_samples_ratio: int = 5,
    ):
        super().__init__()
      
        Xs = invocation['X'].tolist()
        Ys = invocation['Y'].tolist()
        self.num_candidates = num_candidates
        self.is_triple = is_triple
        self.mashup_transform = mashup_transform
        self.api_transform = api_transform
        self.mashups, self.apis, self.labels = [], [], []
        self.mashups = Xs 
        self.labels = Ys
        self.mashup_index = mashup_index


    def __len__(self):
        return len(self.mashups)

    def __getitem__(self, idx):
        mashup = self.mashup_transform(self.mashups[idx])
        if self.mashup_index:
            mashup = torch.tensor(mashup)
        else:
            mashup = torch.from_numpy(mashup)
        
        label = nn.functional.one_hot(torch.LongTensor(self.labels[idx]), num_classes=self.num_candidates)
        label = label.sum(dim=0)

        return mashup, label  
