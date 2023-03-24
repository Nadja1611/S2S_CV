# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:36:10 2023

@author: johan
"""

import os
import numpy as np 
from evaluation import *
import argparse
import matplotlib.pyplot as plt

'''This script allows you to plot for either the mehtod joint or cv the results and gives you
the dice curves and AP curves for the different methods. With feature we define which hyperparameter
we are interested in and name could either be "cv", "sequential" or "joint" for the different methods'''

parser = argparse.ArgumentParser(description='Arguments for segmentation network.')
parser.add_argument('--method', type = str, help = "Which method do we investigate", default = "joint")
parser.add_argument('--name', type = str, help = "Which method do we investigate", default = "joint")
parser.add_argument('--dataset', type = str, help = "Which dataset do we investigate", default = "DSB2018_n20")
parser.add_argument('--feature', type = str, help = "Which feature do we investigate", default = "patience")
args = parser.parse_args()
args.dataset = "DSB_n20"

args.method = "cv"
args.feature = "Lambda"
args.name="cv"
path = "D:/Cambridge/outputs_S2S/"+args.dataset
patients = []


for file in os.listdir(path):
    if file.startswith("pat"):
        patients.append(file)

PP=[]
patients = patients[:20]



for file in patients:
    if "_10" not in file:
        p = (path + "/" + file + "/" + args.feature) 
        PP.append(p)       

data1 = np.load("C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/data/"+args.dataset+"/train/train_data.npz", allow_pickle = True)
gt = data1["Y_train"][1:26]
im = data1["X_train"][1:26]

N_cycles=[]
PP = PP[:]  
PP_result = [] 
Patient=[]  
pat_Dice=[]     
denoisings=[]
results = []
empty=[]
segmentations_joint=[]
segmentations_sequential = []
segmentations_cv= []
segmentations = []
Recall = []
Precision = []
Lambda=[]
c=0
index = 0
pat_AP =[]
A=[]
Dice = []
IOU=[]
patience=[]
window_size=[]
learning_rate=[]
denoisings=[]

for file in PP:
    for data in os.listdir(file):
        pfad = file+ "/" + data
        print(pfad)

        if args.method in data :
            
            da = np.load(pfad+"/results.npz", allow_pickle = True)

            Patient.append(da)
            AP =  AP_score(gt[index], np.round(da["segmentations_"+args.name]), 0.5, 0.5)
            A.append(AP)
            gt_bin = np.copy(gt)
            gt_bin[gt_bin>1]=1

            seg = np.round(da["segmentations_"+args.name])

            fp = np.sum(seg*(1-gt_bin[index]))
            fn = np.sum((1-seg)*gt_bin[index])
            tp = np.sum(seg*gt_bin[index])
            tn = np.sum((1-seg)*(1-gt_bin[index]))
            dice = 2*tp/(2*tp + fn + fp )
            segmentation_metrics= seg_metric(gt[index], seg)
            segmentations_cv.append(np.round(da["segmentations_"+args.name]))
            try:
                denoisings.append(da["denoisings"])
            except:
                pass
            Lambda.append(da["lam"])

            Dice.append(dice)
            IOU.append(segmentation_metrics)
            pat_AP.append(np.mean(A))   
            Precision.append((tp+0.000001)/(tp+fp+ 0.000001))
            Recall.append(tp/(tp+fn))
            
            pat_Dice.append(np.mean(Dice))
            
            learning_rate.append(da["learning_rate"])
            if da["lam"]==0.03:
                fig = plt.figure(figsize=(30,25))
                plt.subplot(2,2,1)
                plt.imshow(im[index])
                plt.subplot(2,2,2)
                plt.imshow(seg)
                plt.subplot(2,2,3)
                plt.plot(A)
                try:
                    plt.subplot(2,2,4)
                    plt.imshow(da["denoisings"])
                except:
                    pass   
                plt.savefig("results_"+args.name+args.feature+str(da["lam"])+str(index)+".png")
                plt.show()         
    
    if index < len(PP) and index != 8:
        index = index+1
    elif index < len(PP) and index ==8:
        index = index+2

        
    results.append(Patient)
    Patient=[]
    
# fig = plt.figure(figsize=(30,25))
# for i in range(len(np.unique(globals()[args.feature]))):
#     ax = fig.add_subplot(3, 4, i+1)
#     ax.imshow(segmentations_cv[i+96])
    

Dices = np.zeros((len(np.unique(globals()[args.feature])),1))
APs = np.zeros((len(np.unique(globals()[args.feature])),1))   
IOUs = np.zeros((len(np.unique(globals()[args.feature])),1))
Precs = np.zeros((len(np.unique(globals()[args.feature])),1))
Recs = np.zeros((len(np.unique(globals()[args.feature])),1))

for i in range(len(Dice)):
    k = i%len(np.unique(globals()[args.feature]))
    Dices[k] = Dices[k] + Dice[i]
    APs[k] = A[i] + APs[k]
    IOUs[k] = IOU[i] + IOUs[k]
    Recs[k] = Recs[k] + Recall[i]
    Precs[k] = Precs[k] + Precision[i]
    
Dices = Dices/len(PP)
APs = APs/len(PP)
IOUs = IOUs/len(PP)
Precs = Precs/len(PP)
Recs = Recs/len(PP)

number_of_different_parms = len(np.unique(globals()[args.feature]))
plt.scatter(globals()[args.feature][:number_of_different_parms],Dices,label="Dice "+ args.name)
plt.scatter(globals()[args.feature][:number_of_different_parms], APs, label="AP " + args.name)
plt.scatter(globals()[args.feature][:number_of_different_parms], IOUs, label="IoU " + args.name)
plt.scatter(globals()[args.feature][:number_of_different_parms], Recs, label="Recall " + args.name)
plt.scatter(globals()[args.feature][:number_of_different_parms], Precs, label="Precision " + args.name)
plt.legend()
plt.title(args.feature)
plt.show()

np.savez_compressed("results_"+args.name+"_data_"+args.dataset+ "_feature_"+args.feature+".npz",Rec = Recs, Prec = Precs, Dices = Dices, APs = APs, IOU = IOUs, seg = segmentations_cv, den = denoisings)
data_seq = np.load("results_sequential_data_"+args.dataset+ "_feature_"+args.feature+".npz")
d_s = data_seq["Dices"]
s_s = data_seq["IOU"]
a_s = data_seq["APs"]
r_s = data_seq["Rec"]
p_s = data_seq["Prec"]
im_s = data_seq["seg"]

data_joint = np.load("results_joint_data_"+args.dataset+ "_feature_"+args.feature+".npz")
s_j = data_joint["IOU"]
d_j = data_joint["Dices"]
a_j = data_joint["APs"]
r_j = data_joint["Rec"]
p_j = data_joint["Prec"]
im_j = data_joint["seg"]
den_s = data_joint["den"]


if args.feature == "Lambda":
    plot_seq = []
    for i in range(len(im_s)):
       if Lambda[i]==Lambda[np.argmax(a_s)]:
               plot_seq.append(im_s[i]) 
    plot_joint = []
    for i in range(len(im_j)):
        if Lambda[i]==Lambda[np.argmax(a_j)]:
            plot_joint.append(im_j[i])   
            
            
            
    data_cv = np.load("results_cv_data_"+args.dataset+ "_feature_"+args.feature+".npz")
    d_c = data_cv["Dices"]
    s_c = data_cv["IOU"]
    a_c = data_cv["APs"]
    r_c = data_cv["Rec"]
    p_c = data_cv["Prec"]
    im_c = data_cv["seg"]
    
    plot_cv = []
    for i in range(len(im_c)):
        if Lambda[i]==Lambda[np.argmax(a_c)]:
            plot_cv.append(im_c[i]) 
            
    plot_den = []
    for i in range(len(den_s)):
        if Lambda[i]==Lambda[np.argmax(a_j)]:
            plot_den.append(den_s[i]) 
                        
    
    images = []
    for i in range(19):
        if i < 9:
            images.append(im[i]) 
            images.append(plot_joint[i])
            images.append(plot_seq[i])
            images.append(plot_cv[i])
            images.append(plot_den[i][0])
            images.append(gt[i])
        else:    
            images.append(im[i+1]) 
            images.append(plot_joint[i])
            images.append(plot_seq[i])
            images.append(plot_cv[i])
            images.append(plot_den[i][0])
            images.append(gt[i+1])


            
            
            
    fig = plt.figure(figsize=(30,20))
    for i in range(24):
        ax = fig.add_subplot(4, 6, i+1)
        ax.imshow(images[i+48])
         
if args.feature == "Lambda":
    number_of_different_parms = len(np.unique(globals()[args.feature]))

    plt.scatter(globals()[args.feature][:number_of_different_parms], d_s, label="Dice sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], d_j, label="Dice joint ", marker= ">" )
    plt.scatter(globals()[args.feature][:number_of_different_parms], d_c, label="Dice CV ")
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylim([0,1])

    plt.show()


    plt.scatter(globals()[args.feature][:number_of_different_parms], s_s,label="IOU sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], s_j, label="IOU joint ", marker = ">" )
    plt.scatter(globals()[args.feature][:number_of_different_parms], s_c, label="IoU CV " )
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylim([0,1])

    plt.show()



    plt.scatter(globals()[args.feature][:number_of_different_parms], a_s,label="AP sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], a_j, label="AP joint ", marker = ">" )
    plt.scatter(globals()[args.feature][:number_of_different_parms], a_c, label="AP CV ",  )
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylim([0,1])

    plt.show()
    
    
    
    plt.scatter(globals()[args.feature][:number_of_different_parms], r_s,label="Recall sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], r_j, label="Recall joint ", marker = ">" )
    plt.scatter(globals()[args.feature][:number_of_different_parms], r_c, label="Recall CV ",  )
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylim([0,1])

    plt.show()
    
    
        
    
    plt.scatter(globals()[args.feature][:number_of_different_parms], p_s,label="Precision sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], p_j, label="Precision joint ", marker = ">" )
    plt.scatter(globals()[args.feature][:number_of_different_parms], p_c, label="Precision CV ",  )
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylim([0,1])

    plt.show()
else: 
    number_of_different_parms = len(np.unique(globals()[args.feature]))

    plt.scatter(globals()[args.feature][:number_of_different_parms], d_s, label="Dice sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], d_j, label="Dice joint ", marker= ">" )
    plt.legend()
    plt.xlabel(args.feature)
    plt.ylim([0,1])

    plt.show()


    plt.scatter(globals()[args.feature][:number_of_different_parms], s_s,label="IOU sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], s_j, label="IOU joint ", marker = ">" )
    plt.legend()
    plt.xlabel(args.feature)
    plt.ylim([0,1])

    plt.show()



    plt.scatter(globals()[args.feature][:number_of_different_parms], a_s,label="AP sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], a_j, label="AP joint ", marker = ">" )
    plt.legend()
    plt.xlabel(args.feature)
    plt.ylim([0,1])

    plt.show()
    
    
    
    plt.scatter(globals()[args.feature][:number_of_different_parms], r_s,label="Recall sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], r_j, label="Recall joint ", marker = ">" )
    plt.legend()
    plt.xlabel(args.feature)
    plt.ylim([0,1])

    plt.show()
    
    
        
    
    plt.scatter(globals()[args.feature][:number_of_different_parms], p_s,label="Precision sequential ", marker = "x")
    plt.scatter(globals()[args.feature][:number_of_different_parms], p_j, label="Precision joint ", marker = ">" )
    plt.legend()
    plt.xlabel(args.feature)
    plt.ylim([0,1])