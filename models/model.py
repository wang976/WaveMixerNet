import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import Permute, Reshape
from utils.RevIN import RevIN

import matplotlib.pyplot as plt
import numpy as np
from models.wavemixernet import WaveMixerNetCore


class WaveMixerNet(nn.Module):
    def __init__(self, configs, device):
        
        super(WaveMixerNet, self).__init__()

        self.configs = configs
        self.device = device
        self.channel_out = configs.c_out

        
        self.waveMixerNetCore = WaveMixerNetCore(self.configs, self.device)
        
        
    def forward(self, x):
        pred = self.waveMixerNetCore(x)
        pred = pred[:, :, -self.channel_out:]
        return pred 

