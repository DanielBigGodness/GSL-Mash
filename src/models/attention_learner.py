import torch
import torch.nn as nn

import dgl

from .base_learner import BaseLearner
from .metric import *
from .processor import *
from .utils import knn_fast
import pickle
device=torch.device("cuda")

import torch.nn.init as init

class Attentive(nn.Module):
    def __init__(self, size): 
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.rand(size), requires_grad=True)


    def forward(self, x):
        return x @ torch.diag(self.w)


class AttLearner(BaseLearner):
    """Attentive Learner"""
    def __init__(self, metric, processors, nlayers, size, activation):

        super(AttLearner, self).__init__(metric, processors)
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        for _ in range(self.nlayers):
            self.layers.append(Attentive(size))
        self.activation = activation


    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != (self.nlayers - 1):
                x = self.activation(x)
        return x

    def forward(self, features):
        z = self.internal_forward(features)
        z = F.normalize(z, dim=1, p=2)
        similarities = self.metric(z)
        for processor in self.processors:
            similarities = processor(similarities)
        similarities = F.relu(similarities)
        return similarities

 
