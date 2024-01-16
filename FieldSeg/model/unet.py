import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    def __init__(self,
                 in_planes,
                 weight="imagenet",
                 N_Class=1):
        super(UNet, self).__init__()
        self.unet = smp.Unet(in_channels=in_planes, classes=N_Class, encoder_weights=weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.unet(x)
        out = self.sigmoid(out)
        return out



