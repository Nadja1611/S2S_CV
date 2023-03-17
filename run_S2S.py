# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:28:16 2023

@author: johan
"""

import os
import numpy as np
from math import sqrt
from Functions_pytorch import *
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from skimage.color import rgb2gray
import scipy
from skimage.transform import resize
from skimage.color import rgb2gray
from evaluation import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model_S2S import *
from utils import *
from torch.optim import Adam
import time
import argparse
VERBOSE = 1
# ----

parser = argparse.ArgumentParser(description='Arguments for segmentation network.')
parser.add_argument('--output_directory', type=str, 
                    help='directory for outputs', default="/home/fs71765/ngruber/joint_denoising_segmentation/outputs_S2S")
parser.add_argument('--input_directory', type=str, 
                    help='directory for outputs', default="/home/fs71765/ngruber/joint_denoising_segmentation")
parser.add_argument('--learning_rate', type=float, 
                    help='learning rate', default=0.00001)
parser.add_argument('--method', type = str, help="joint/sequential or only Chan Vese cv", default = "joint")
parser.add_argument('--lam', type = float, help = "regularization parameter of CV", default = 0.0000001)
parser.add_argument('--ratio', type = int, help = "What is the ratio of masked pixels in N2v", default = 0.3)
parser.add_argument('--experiment', type = str, help = "What hyperparameter are we looking at here? Assigns the folder we want to produce with Lambda, if we make lambda tests for example", default = "/Lambda")
parser.add_argument('--patient', type = int, help = "Which patient index do we use", default = 0)
parser.add_argument('--dataset', type = str, help = "Which dataset", default = "DSB2018_n20")
parser.add_argument('--GPU', type = int, help = "Which GPU do you want to use", default = 0)



args = parser.parse_args()


args.output_directory = args.output_directory+"/" + args.dataset + "/patient_"+ str(args.patient).zfill(2) +  args.experiment
#define directory where you want to have your outputs saved
name = "/S2S_Method_"+ args.method + "_Lambda_" + str(args.lam) + "_ratio_"+ str(args.ratio) +'_lr_'+str(args.learning_rate)
path = args.output_directory+  name


print(path)

def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
  else:
    print("Directory already existed : ", dir)
  return dir
args.patient = 3
create_dir(path)

gpu_ids = []
gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
print(gpu_ids)
try:
    device = torch.device(f'cuda:{gpu_ids[args.GPU]}')
except:
    device = torch.device(f'cuda:{gpu_ids[0]}')

#data = np.load(
 #   'D:/DenoiSeg/DSB2018_n20.npz', allow_pickle=True)
data = np.load(
    "/home/fs71765/ngruber/joint_denoising_segmentation/data/" + args.dataset +"/train/train_data.npz", allow_pickle=True)
f = torch.tensor(data["X_train"][args.patient:args.patient+1]).to(device)
gt = data["Y_train"]
f_denoising = torch.clone(f)
mynet = Denseg_S2S(learning_rate = args.learning_rate, lam = args.lam)
mynet.initialize(f)
f=mynet.normalize(f)
f_norm = torch.mean(f**2)
print(f_norm)
mynet = Denseg_S2S(learning_rate = (1-f_norm)*args.learning_rate, lam = args.lam)
mynet.initialize(f)
f=mynet.normalize(f)
n_it = 200
args.method == "joint"

if args.method == "joint" or args.method == "cv":
    for i in range(n_it):
        mynet.segmentation_step(mynet.f)
       # plt.imshow(mynet.f[0].cpu())
        if args.method == "joint":
            mynet.denoising_step()
        if i%10 ==0:
            plt.subplot(2,2,1)
            plt.imshow(mynet.x_tilde[0].cpu())
            plt.colorbar()
            plt.subplot(2,2,2)
            plt.imshow(f[0].cpu())
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow(mynet.f.cpu(),cmap ='inferno')
            plt.colorbar()

            plt.subplot(2,2,4)
            plt.imshow(gt[args.patient],cmap ='inferno')
            # plt.show()
            plt.show()




if args.method == "joint":
    mynet_sequential = Denseg_S2S(learning_rate = (1-f_norm)*args.learning_rate, lam = args.lam)
    mynet_sequential.initialize(mynet.f.unsqueeze(0))

    for i in range(n_it):
        mynet_sequential.segmentation_step(mynet.f)        
if args.method == "joint":        
    np.savez_compressed(path + "/results.npz", denoisings = mynet.f[0].cpu(), segmentations_joint = mynet.x_tilde[0].cpu(), segmentations_sequential = mynet_sequential.x_tilde[0].cpu(), learning_rate = args.learning_rate, lam = args.lam)        

if args.method == "cv":
    np.savez_compressed(path + "/results.npz",  segmentations_cv = mynet.x_tilde[0].cpu(), learning_rate = args.learning_rate, lam = args.lam )        
     
 

       
     
 



