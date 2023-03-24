#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:35:49 2023

@author: nadja
"""

import numpy as np
#from numba import jit
from scipy import ndimage
from tqdm import tqdm, tqdm_notebook

'''here, we define the functions for computing the AP score as in DenoiSeg paper---'''

def pixel_sharing_bipartite(lab1, lab2):
    assert lab1.shape == lab2.shape
    psg = np.zeros((lab1.max() + 1, lab2.max() + 1), dtype=int)
    for i in range(lab1.size):
        psg[lab1.flat[i], lab2.flat[i]] += 1
    return psg



def intersection_over_union(psg):
    """
    Computes IOU.
    :Authors:
        Coleman Broaddus
     """
    rsum = np.sum(psg, 0, keepdims=True)
    csum = np.sum(psg, 1, keepdims=True)
    return psg / (rsum + csum - psg)


def matching_iou(psg, fraction):
    """
    Computes IOU.
    :Authors:
        Coleman Broaddus
     """
    iou = intersection_over_union(psg)
    matching = iou > fraction
    matching[:, 0] = False
    matching[0, :] = False
    return matching


def compute_labels(prediction_fg, threshold):
    
    pred_thresholded = prediction_fg > threshold
    labels, _ = ndimage.label(pred_thresholded)

    prediction_binary = np.where(prediction_fg > threshold, np.ones_like(prediction_fg), np.zeros_like(prediction_fg))

    return labels, prediction_binary



def matching_overlap(psg, fractions=(0.5,0.5)):
    """
    create a matching given pixel_sharing_bipartite of two label images based on mutually overlapping regions of sufficient size.
    NOTE: a true matching is only gauranteed for fractions > 0.5. Otherwise some cells might have deg=2 or more.
    NOTE: doesnt break when the fraction of pixels matching is a ratio only slightly great than 0.5? (but rounds to 0.5 with float64?)
    """
    afrac, bfrac = fractions
    tmp = np.sum(psg, axis=1, keepdims=True)
    m0 = np.where(tmp==0,0,psg / tmp)
    tmp = np.sum(psg, axis=0, keepdims=True)
    m1 = np.where(tmp==0,0,psg / tmp)
    m0 = m0 > afrac
    m1 = m1 > bfrac
    matching = m0 * m1
    matching = matching.astype('bool')
    return matching

def seg_metric(lab_gt, lab):
    """
    calculate seg from pixel_sharing_bipartite
    seg is the average conditional-iou across ground truth cells
        conditional-iou gives zero if not in matching
    ----
    calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
        for a fraction > 0.5 matching. Any CIoU between matching pairs will be > 1/3. But there may be some
    IoU as low as 1/2 that don't match, and thus have CIoU = 0.
        """
    lab, lab_gt = lab.astype("int32"), lab_gt.astype("int32")

    psg = pixel_sharing_bipartite(lab_gt, lab)
    iou = intersection_over_union(psg)
    matching = matching_overlap(psg, fractions=(0.5, 0))
    matching[0, :] = False
    matching[:, 0] = False
    n_gt = len(set(np.unique(lab_gt)) - {0})
    n_matched = iou[matching].sum()


    return n_matched 



'''computation of AP score, this function depends on the threshold for binaarization of segmentation masks'''
'''------------------------------------and the iou we define-----------------------------------------------''
''''---------------------------for a sample to be counted positive------------------------------------------'''

def AP_score(lab_gt, lab,threshold, fraction):
        '''computes the average precsision between gt and predictions, which is         
        precision = TP / (TP + FP + FN) i.e. "intersection over union" for a graph matching'''
        lab, lab_gt = lab.astype("int32"), lab_gt.astype("int32")
        labels, prediction_binary = compute_labels(lab, threshold)

        psg = pixel_sharing_bipartite(lab_gt, labels)
        matching = matching_iou(psg, fraction=fraction)
        assert matching.sum(0).max() < 2
        assert matching.sum(1).max() < 2
        n_gt = len(set(np.unique(lab_gt)) - {0})
        print(n_gt)
        n_hyp = len(set(np.unique(labels)) - {0})
        print(n_hyp)
        n_matched = matching.sum()
        print(n_matched)
        return n_matched / (n_gt + n_hyp - n_matched)