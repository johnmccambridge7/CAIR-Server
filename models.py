import geffnet
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(context, i):
        result = i * sigmoid(i)
        context.save_for_backward(i)
        return result

    @staticmethod
    def backward(context, gradientOutput):
        i = context.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return gradientOutput * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class EfficientNet(nn.Module):
    def __init__(self, enet, outputDim, nMeta=0, metaDim=[512, 128], pretrained=True):
        super(EfficientNet, self).__init__()
        self.enet = geffnet.create_model(enet, pretrained=pretrained)
        self.dropouts = nn.ModuleList([ nn.Dropout(0.5) for _ in range(5) ])

        inputChannels = self.enet.classifier.in_features

        self.myfc = nn.Linear(inputChannels, outputDim)
        self.enet.classifier = nn.Identity()

    def forward(self, x, xMeta = None):
        x = self.enet(x).squeeze(-1).squeeze(-1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                output = self.myfc(dropout(x))
            else:
                output += self.myfc(dropout(x))

        output = output / len(self.dropouts)

        return output