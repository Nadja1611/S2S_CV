# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 08:40:29 2023

@author: johan
"""


import numpy as np # linear algebra

import os
import glob

from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import util_S2S
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import utils,models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from partialconv2d_S2S import PartialConv2d
from model_S2S import self2self
from torch.optim import lr_scheduler

dataset = "DSB2018_n20"


path = "C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/results/"+dataset
patients = []



     

data1 = np.load("C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/data/"+dataset+"/train/train_data.npz", allow_pickle = True)
gt = data1["Y_train"][1:26]
im = data1["X_train"][16:17]
im = im-np.min(im)
im = im/np.max(im)
#im = im
def image_loader(image, device, p1, p2):
    """load image, returns cuda tensor"""
    loader = T.Compose([T.ToPILImage(),T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),T.RandomVerticalFlip(torch.round(torch.tensor(p2))),T.ToTensor()])
    image = torch.tensor(image).float()
    image= loader(image)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)










if __name__ == "__main__":
	##Enable GPU
	USE_GPU = True
	val_loss_list = []
	dtype = torch.float32
	
	if USE_GPU and torch.cuda.is_available():
	    device = torch.device('cuda')
	else:
	    device = torch.device('cpu')
	
	print('using device:', device)
	
	model = self2self(1,0.3)
	img = (np.moveaxis(im,2,1))    
	learning_rate = 1e-6
	model = model.cuda()
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience = 20, threshold_mode='rel',verbose = True,cooldown = 0)


	w,h,c = img.shape
	p=0.3
	NPred=10
	slice_avg = torch.tensor([1,1,128,128]).to(device)
	for itr in range(10000):
	    p_mtx = np.random.uniform(size=[img.shape[0],img.shape[1],img.shape[2]])
	    mask = (p_mtx>p).astype(np.double)*0.7
	    #img_input = img*mask
	    img_input = np.copy(img)
	    #y = img - img_input
	    y = img
	    p1 = np.random.uniform(size=1)
	    p2 = np.random.uniform(size=1)
	    img_input_tensor = image_loader(img_input, device, p1, p2)
	    y = image_loader(y, device, p1, p2)
	    mask = np.expand_dims(mask,0)            
	    mask = torch.tensor(mask).to(device, dtype=torch.float32)	    
	    model.train()
	    img_input_tensor = img_input_tensor*mask
	    output = model(img_input_tensor, mask)	            
	    loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),T.RandomVerticalFlip(torch.round(torch.tensor(p2)))])
	    loss = torch.sum((output-y)*(output-y)*(1-mask))/torch.sum(1-mask)
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()
	    

	    loss_val_G = 0
	    if (itr+1)%50 == 0:
	        model.eval()
	        sum_preds = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
	        for j in range(NPred):
                #generate bernoulli sampled instances
	            p_mtx_vl = np.random.uniform(size=img.shape)
	            mask_val = (p_mtx_vl>p).astype(np.double)*0.7
	            img_input_val = np.copy(img)*mask_val
	            img_input_tensor_val = image_loader(img_input_val, device,0.0,0.0)
	            y_val = image_loader(img_input_val, device ,0.0,0.0)

	            mask_val = torch.tensor(mask_val).to(device, dtype=torch.float32).unsqueeze(0)
	            with torch.no_grad():
	                output_test = model(img_input_tensor_val,mask_val)
	                sum_preds[:,:,:] += output_test.detach().cpu().numpy()[0]
	                loss_val = torch.sum((output_test - y_val)*(output_test-y_val)*(1-mask_val))/torch.sum(1-mask_val)
    
	                loss_val_G += loss_val.item()
	        print('validation loss:',loss_val_G)
	        val_loss_list.append(loss_val_G)
                
                
            
	        avg_preds = np.squeeze(sum_preds/np.max(sum_preds))
	        plt.subplot(1,3,1)
	        plt.imshow(avg_preds)
	        plt.subplot(1,3,2)
	        plt.imshow(img[0])
	        plt.subplot(1,3,3)
	        plt.plot(val_loss_list)
	        plt.show()



