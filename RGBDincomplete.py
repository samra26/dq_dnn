import torch
import torch.nn as nn
from cswin import CSWinTransformer
import torch.nn.functional as F
from functools import partial
from torchsummary import summary
from timm.models.layers import DropPath, trunc_normal_
import os
import cv2
import numpy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
im_size=(320,320)


class RGBDInModule(nn.Module):
    def __init__(self, backbone):
        super(RGBDInModule, self).__init__()
        self.backbone = backbone
        

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {ka: va for ka, va in pretrained_dict.items() if ka in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=True)
        

    def forward(self, x):
        B,C,H,W=x.shape
        x,x1= self.backbone(x)
        '''for i in range(len(x1)):
            print('The backbone features are',x1[i].shape)
        print('finish looping')'''
        return x1,B,H,W


class RGBD_incomplete(nn.Module):
    def __init__(self,RGBDInModule):
        super(RGBD_incomplete, self).__init__()
        
        self.RGBDInModule = RGBDInModule
        self.conv1x1=nn.Conv2d(144, 1,1, 1)

        
    def forward(self, f_all):
        x1,B,H,W = self.RGBDInModule(f_all)
        x_out=x1[1]
        _,_,C=x_out.shape
        x_out=x_out.transpose(-2,-1).contiguous().view(B, C, int(H/4), int(W/4))
        #print('x_out',x_out.shape)
        x_out=self.conv1x1(x_out)
        #print('x_outa',x_out.shape)
        return x_out


def build_model(network='cswin', base_model_cfg='cswin'):
    backbone = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[6,12,24,24], mlp_ratio=4.0)
      
   

    return RGBD_incomplete(RGBDInModule(backbone))
