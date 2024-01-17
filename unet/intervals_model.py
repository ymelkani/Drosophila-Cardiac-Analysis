import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(                                   #ks 9, pad 4 for SIModelCP
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2), #ks 5, pad 2 for SIModelCP(working)
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),                                           #ks 9, pad 4 for SIModelCP
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2), #ks 5, pad 2 for SIModelCP(working)
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.double_conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = Conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        

class IntervalsModel(nn.Module):    
    
    def __init__(self, sampleLen):
        super().__init__()
        c = [1,4,8,16] #channels
        self.maxpool = nn.MaxPool1d(2)
        self.expandDim = nn.Unflatten(1,(1,sampleLen))
        self.flattenDim = nn.Flatten(0,1)
        self.conv1 = Conv(c[0], c[1]) 
        self.conv2 = Conv(c[1],c[2])
        self.conv3 = Conv(c[2],c[3])
        self.up1 = Up(c[3], c[2])
        self.up2 = Up(c[2], c[1])
        self.final = nn.Conv1d(c[1], c[0], kernel_size=1)
        
    def forward(self, x):
        x = self.expandDim(x)
        x1 = self.conv1(x)
        x1mp = self.maxpool(x1)
        x2 = self.conv2(x1mp)
        x2mp = self.maxpool(x2)
        x3 = self.conv3(x2mp)
        x = self.up1(x3,x2) 
        x = self.up2(x,x1)
        x = self.final(x)
        x = self.flattenDim(x)
        # print("final:", x.size())
        return x