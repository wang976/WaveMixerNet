import torch
from torch import nn

class DFEBranch(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, trend, cfactor):
        super(DFEBranch, self).__init__()
        
        # Parameters
        self.pred_len = pred_len

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)


        self.conv = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=1)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(self.patch_num)


        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num//trend * patch_len, pred_len * cfactor)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * cfactor, pred_len)

    def forward(self, x):
        # x: [Batch, Input, Channel]
        
        x = x.permute(0,2,1) # to [Batch, Channel, Input]
        
        B = x.shape[0] # Batch size
        C = x.shape[1] # Channel size
        I = x.shape[2] # Input size
        x = torch.reshape(x, (B*C, I)) # [Batch and Channel, Input]

        # Patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [Batch and Channel, Patch_num, Patch_len]
        
        # Patch Embedding
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.bn1(x)

        x = self.conv(x)
        x = self.gelu(x)
        x = self.bn(x)

        # Flatten Head
        x = self.flatten1(x)
        x = self.fc3(x)
        x = self.gelu4(x)
        x = self.fc4(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x