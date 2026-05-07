import torch.nn as nn
import torch
import numpy as np
from utils.RevIN import RevIN
from layers.decomposition import Decomposition
from layers.DFE import DFEBranch
from layers.AFE import AFEBranch

class WaveMixerNetCore(nn.Module):
    def __init__(self, configs, device):
        
        super(WaveMixerNetCore, self).__init__()
        self.input_length = configs.seq_len
        self.pred_length = configs.pred_len
        self.wavelet_name = configs.wavelet
        self.level = configs.level
        self.batch_size = configs.batch_size
        self.channel = configs.c_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.embedding_dropout = configs.embedding_dropout
        self.device = device
        self.no_decomposition = configs.no_decomposition
        self.tfactor = configs.tfactor
        self.dfactor = configs.dfactor
        self.cfactor = configs.cfactor
        self.use_amp = configs.use_amp

        self.Decomposition_model = Decomposition(configs, self.device)
        
        self.input_w_dim = self.Decomposition_model.input_w_dim # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim # list of the length of the predicted coefficient series

        self.patch_len = configs.patch_len
        self.patch_stride = configs.stride
        
        self.AFE = AFEBranch(input_seq = self.input_w_dim[0],
                                            pred_seq = self.pred_w_dim[0],
                                            batch_size = self.batch_size,
                                            channel = self.channel,
                                            d_model = self.d_model,
                                            dropout = self.dropout,
                                            embedding_dropout = self.embedding_dropout,
                                            tfactor = self.tfactor,
                                            dfactor = self.dfactor,
                                            patch_len = self.patch_len,
                                            patch_stride = self.patch_stride)
        
        self.revin = RevIN(self.channel, eps=1e-5, affine = True, subtract_last = False)

        self.DFE = DFEBranch(self.input_w_dim[-1], self.pred_w_dim[-1], self.patch_len, self.patch_stride, 'end', 1, self.cfactor)
        

    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        xT : Prediction time series: [Batch, prediction_length, output_channel]
        '''
        x = self.revin(xL, 'norm')
        x = x.transpose(1, 2) # [batch, channel, look_back_length]

        xA, xD = self.Decomposition_model.transform(x)

        yA = self.AFE(xA)

        yD = []
        yD_ = self.DFE(xD[0].transpose(1, 2))
        yD.append(yD_.transpose(1, 2))
        
        y = self.Decomposition_model.inv_transform(yA, yD)
        
        y = y.transpose(1, 2)
        y = y[:, -self.pred_length:, :] # decomposition output is always even, but pred length can be odd
        xT = self.revin(y, 'denorm')
        
        return xT
    
