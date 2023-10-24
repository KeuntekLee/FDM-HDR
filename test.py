
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataset_kalantari import Dynamic_Scenes_Dataset
from models import FDM_HDR
from utils.utils import *
from decomposenet import Encoder_S, Encoder_E_FFT

data_root = '/data/HDR_KALANTARI'
batch_size = 8


#print("FIRST")

import os
import cv2
import glob
#tf.compat.v1.disable_eager_execution()
import glob
import torch

#file_paths = [file_path] # We have only one file
from skimage.metrics import structural_similarity as ssim


val_dataset = Dynamic_Scenes_Dataset(root_dir=data_root, is_training=False, transform=None, crop=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

model_test = FDMNet().cuda()
ee_test = Encoder_E_FFT(inc=3,n_downsample=4,outc=256,ndf=64,usekl=False)
ee_test.load_state_dict(torch.load("./checkpoints/ee_99000.pth"))
es_test = Encoder_S(n_downsample=2,ndf=64,norm_layer='LN')
es_test.load_state_dict(torch.load("./checkpoints/es_99000.pth"))
model_test.cuda()
ee_test.cuda()
es_test.cuda()

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
model_test.cuda()
model_test.eval()
#ee_test.eval()
requires_grad(model_test, False)
requires_grad(ee_test, False)
requires_grad(es_test, False)
print(next(model_test.parameters()).device)

checkpoint = torch.load("./checkpoints/FDM_"+str(test_epoch)+".pth")
model_test.load_state_dict(checkpoint)
print(test_epoch)
for idx,i in enumerate(val_loader):
    #print("Inferencing "+ str(idx))
    #print(i['input0'].shape)
    ldr1 = i['input0'].cuda()
    ldr2 = i['input1'].cuda()
    ldr3 = i['input2'].cuda()
    ldr1_tm = i['input0_tm'].cuda()
    ldr2_tm = i['input1_tm'].cuda()
    ldr3_tm = i['input2_tm'].cuda()
    gt = i['label'].cuda()

    batch_size,ch,H,W = ldr1.shape
    #print(H,W)
    
    ldr1 = ldr1*2 -1
    ldr2 = ldr2*2 -1
    ldr3 = ldr3*2 -1
    
    ldr1_tm = ldr1_tm*2 -1
    ldr2_tm = ldr2_tm*2 -1
    ldr3_tm = ldr3_tm*2 -1
    
    lumi1_real, lumi1_imag = ee_test(ldr1)
    lumi2_real, lumi2_imag = ee_test(ldr2)
    lumi3_real, lumi3_imag = ee_test(ldr3)
  
    sp1 = es_test(ldr1)
    sp2 = es_test(ldr2)
    sp3 = es_test(ldr3)

    ldr1_cat = torch.cat([ldr1,ldr1_tm], dim=1)
    ldr2_cat = torch.cat([ldr2,ldr2_tm], dim=1)
    ldr3_cat = torch.cat([ldr3,ldr3_tm], dim=1)

    pred = model_test(ldr1_cat, ldr2_cat, ldr3_cat, [lumi1_real,lumi2_real,lumi3_real], [lumi1_imag,lumi2_imag,lumi3_imag],
            [sp1, sp2, sp3])

    #print(interpolated1.shape)
    #pred = model(ldr1_cat, ldr2_cat, ldr3_cat)
    #pred = torch.clamp(pred, -1, 1)
    pred = torch.clamp(pred, -1, 1)
    pred = pred[0,:,:,:]#.astype(np.float32)
    pred = (pred+1.)/2.


    
