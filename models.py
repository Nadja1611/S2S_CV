#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 12:13:48 2023

@author: nadja
"""

from layer import *

import torch
import torch.nn as nn
from torch.nn import init, ReflectionPad2d
from torch.optim import lr_scheduler
from utils import *
from Functions_pytorch import *
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

# U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/abs/1505.04597
class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.pad = ReflectionPad2d(1)

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """
        self.enc1_1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=3,padding = 0, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        """
        Decoder part
        """

        self.dec5_1 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool4 = UnPooling2d(pool=2, type='nearest')

        self.dec4_2 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec4_1 = DECNR2d(8 * self.nch_ker,     4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool3 = UnPooling2d(pool=2, type='nearest')

        self.dec3_2 = DECNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec3_1 = DECNR2d(4 * self.nch_ker,     2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool2 = UnPooling2d(pool=2, type='nearest')

        self.dec2_2 = DECNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec2_1 = DECNR2d(2 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool1 = UnPooling2d(pool=2, type='nearest')

        self.dec1_2 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec1_1 = DECNR2d(1 * self.nch_ker,     1 * self.nch_out, kernel_size=3, stride=1, norm=[],        relu=[],  drop=[], bias=False)

    def forward(self, x):

        """
        Encoder part
        """
        enc0 = self.pad(x)
        enc1 = self.enc1_2(self.enc1_1(enc0))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        x = dec1

        return x

# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802
class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=16):
        super(ResNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(self.nch_in, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=[], relu=0.0)

        res = []
        for i in range(self.nblk):
            res += [ResBlock(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]
        self.res = nn.Sequential(*res)

        self.dec1 = CNR2d(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=norm, relu=[])

        self.conv1 = Conv2d(self.nch_ker, self.nch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.enc1(x)
        x0 = x

        x = self.res(x)

        x = self.dec1(x)
        x = x + x0

        x = self.conv1(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256
        # dsc4 : 32 x 32 x 256 -> 16 x 16 x 512
        # dsc5 : 16 x 16 x 512 -> 16 x 16 x 1

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        # x = torch.sigmoid(x)

        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class Denseg:
    def __init__(
        self,
        learning_rate: float = 4e-3,
        patch_size: int = 64,
        patience: int = 20,
        lam: float = 0.01,
        N_patches: int = 200,
        N_cycles: int = 1,
        NW_depth: int = 5,
        nch_kernel: int = 64,
        window_size: int = 5,
        ratio: float = 0.9,
        device: str = 'cuda:0',
    ):
        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.lam = lam
        self.N_patches = N_patches
        self.N_cycles = N_cycles
        self.NW_depth = NW_depth
        self.window_size = window_size
        self.ratio = ratio
        self.patience = patience
        self.DenoisNet = UNet(1,1, nch_ker=nch_kernel,norm = []).to(device)
        self.optimizer = Adam(self.DenoisNet.parameters(),
                     lr=self.learning_rate, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                        factor=0.5, patience = self.patience, threshold=0.000001, threshold_mode='rel',verbose = True,cooldown = 0)
        self.sigma = 1.0/np.sqrt(10)
        self.tau = 1.0/np.sqrt(10)
        self.theta = 1.0
        self.p = []
        self.q=[]
        self.r = []
        self.x_tilde = []
        self.device = device
        self.f_std = []
        
    def normalize(self,f):
        f = torch.tensor(f).float()
        f = (f-torch.min(f))/(torch.max(f)-torch.min(f))
        return f
    def standardise(self,f):
        f = torch.tensor(f).unsqueeze(3).float()
        f = (f - torch.mean(f))/torch.std(f)
        return f
        
    def initialize(self,f):
        #prepare input for denoiser
        f_train = torch.tensor(f).unsqueeze(3).float()
        f_train = (f_train - torch.mean(f_train))/torch.std(f_train)
        dataset = TensorDataset((f_train[:, :, :]), (f_train[:, :, :]))
        self.train_loader = DataLoader(dataset, batch_size=1, pin_memory=False)
        
        f_val = torch.clone(f_train)
        dataset_val = TensorDataset(f_val, f_val)
        self.val_loader = DataLoader(dataset_val, batch_size=1, pin_memory=False)
        self.f_std = torch.clone(f_train)


        #prepare input for segmentation
        f = self.normalize(f)
        self.p = gradient(f)
        self.q = f
        self.r = f
        self.x_tilde = f
        self.f = torch.clone(f)
        

        
    def segmentation_step(self,f):
        #print(torch.min(f))
        f_orig = torch.clone(f)
        
        # for segmentaion process, the input should be normalized and the values should lie in [0,1]
        
        '''-------------now the segmentation process starts-------------------'''
        ''' Update dual variables of image f'''
        p1 = proj_l1_grad(self.p + self.sigma*gradient(self.x_tilde), self.lam)  # update of TV
        # Fidelity term without norm (change this in funciton.py)
        q1 = proj_l1(self.q + self.sigma*Fid1(self.x_tilde, f), 1)
        r1 = proj_l1(self.r + self.sigma*Fid2(self.x_tilde, f), 1)
        self.p = p1.clone()
        self.q = q1
        self.r = r1
        # Update primal variables
        x_old = self.x_tilde  
        x = proj_unitintervall(x_old + self.tau*div(p1) - self.tau*adjoint_der_Fid1(x_old, f, self.q) - self.tau *
                               adjoint_der_Fid2(x_old, f, self.r))  # proximity operator of indicator function on [0,1]
        self.x_tilde = euclidean_proj_simplex(x)
        update = self.x_tilde + self.theta*(self.x_tilde-x_old)

        self.x_tilde = update
        
    def denoising_step(self):
        f = self.f_std
        loss_val_G = 0
        val_losslist =[]
        self.DenoisNet.train()
        for cycle in range(self.N_cycles):

            for im, im_true in self.train_loader:
                im, im_true = im.to(self.device), im_true.to(self.device)
                #print(torch.mean(im))
                patches, masked_patches, mask_patches = extract_random_patches2(
                    im_true, im_true, im_true, self.patch_size, self.N_patches)
                patches, masked_patches,mask_patches = augment_images2(patches, masked_patches, mask_patches)
                im_true = torch.clone(patches)
                masked_im, mask = generate_mask(im_true, ratio=self.ratio)

                im_true = im_true.movedim(3, 1)
                masked_im = masked_im.movedim(3, 1)
                im = torch.clone(im_true).detach()
                self.optimizer.zero_grad()
                mask = mask.to(self.device)
                masked_im = masked_im.to(self.device
                                         )
                im_true = im_true.to(self.device)
                denois_im = self.DenoisNet(masked_im)
                mask = mask.moveaxis(3, 1).detach()
               # loss = torch.mean(mask*torch.abs(denois_im-im_true))
                loss = torch.sum(
                    (mask*(denois_im-im_true))**2)/(torch.sum(mask))
                loss.backward()
                self.optimizer.step()
            '''for updating the learning rate, we check validation loss'''
        with torch.no_grad():
            for im_val, im_val_true in self.val_loader:
                im_val, im_val_true = im_val.to(self.device), im_val_true.to(self.device)
                im_val = torch.clone(im_val_true)
                masked_im_val, mask_val = generate_mask(im_val, ratio=self.ratio)
                mask_val = mask_val.to(self.device)
                masked_im_val = masked_im_val.movedim(3, 1)
                self.DenoisNet.eval()
                output = self.DenoisNet(masked_im_val)
                mask_val = mask_val.moveaxis(3, 1)
                    # loss_val = torch.mean((torch.abs(output-im_val_true)))#/(torch.sum(mask_val))
                loss_val = torch.sum(
                (mask_val*(output-im_val_true))**2)/torch.sum(mask_val)
                loss_val_G += loss_val.item()
                
        self.scheduler.step(loss_val_G)
        val_losslist.append(loss_val_G)
        #print("lr"+str(get_lr(self.optimizer)))

        '''---learning rate halving if we are at a plateau of the val loss---'''

        # if len(val_losslist) > 3:
        #     if torch.mean(torch.tensor(val_losslist[-4:])) < loss_val_G:
        #         self.learning_rate = self.learning_rate/2
        #         for g in self.optimizer.param_groups:
        #             g['lr'] = self.learning_rate
        #         print('learning rate halved, new learning rate is:',
        #                   self.learning_rate)

               # f_orig = f_orig.unsqueeze(0)
        denois_im = self.DenoisNet(f.movedim(3,1))
        f = denois_im[0].detach()
        '''again, normalization of the image f to [0,1] for CV---'''
        f = (f-torch.min(f))/(torch.max(f)-torch.min(f))
                
        self.f=torch.clone(f)
        


