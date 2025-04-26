#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Epichide
# @Email: no email
# @Time: 2025/4/27 1:34
# @File: minimal_multiimg.py
# @Software: PyCharm
import os
import  time
import  cv2
import matplotlib.pyplot as plt
import  numpy as np
import torch
from modules.xfeat import XFeat
from scipy.spatial import cKDTree,distance

from utils import imread, match_multi_by_star
from visualize_result import draw_multiimg_points

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU, comment for GPU
# Load the XFeat model
xfeat = XFeat()



if __name__ == '__main__':
    imgfiles=[
        #"./assets/ref.png",
        #"assets/tgt.png",
        #"assets/IMG_20250427_034650.jpg",
        #"assets/IMG_20250427_034700.jpg"
        # 'data/1.jpg',
        # 'data/2.jpg',
        "assets\IMG_20250427_034717.jpg",
        "assets\IMG_20250427_034743.jpg",
        "assets\IMG_20250427_035818.jpg"

    ]
    #Super parameters
    radius=256
    scale=4
    maxcnt=16



    imgs=[imread(imgfile) for imgfile in imgfiles]
    maxshape=tuple((max([img.shape[1] for img in imgs]),max([img.shape[0] for img in imgs])))
    imgs=[cv2.resize(img,maxshape) for img in imgs]
    refimg=imgs[0]
    h,w,c=refimg.shape
    t1=time.time()
    # Computer matches
    match_kpts_list = match_multi_by_star(imgs,xfeat,refidx=0,
                                          radius=radius,
                                          scale=scale,
                                          disth=2)
    good_matches=[cv2.DMatch(i,i,0) for i in range(len(match_kpts_list[0]))]
    kps=[[cv2.KeyPoint(x[0],x[1],2) for x in match_kpts[:maxcnt]] for match_kpts in match_kpts_list]
    print(match_kpts_list)
    draw_multiimg_points(imgs,kps,r=radius)