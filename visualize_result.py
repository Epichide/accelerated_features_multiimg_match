#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Epichide
# @Email: no email
# @Time: 2025/4/27 3:19
# @File: visualize_result.py
# @Software: PyCharm
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_multi(new_imgs,titles=[],nrow=1):
    """
    Show multiple images in a grid.
    :param new_imgs: List of images to show.
    :param titles: List of titles for each image.
    :param nrow: Number of images per row.
    """
    n = len(new_imgs)
    ncol = int(np.ceil(n / nrow))
    ratio=new_imgs[0].shape[1]/new_imgs[0].shape[0]
    fig, axes = plt.subplots( nrow,ncol,sharey=True,sharex=True,squeeze=True)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(new_imgs[i],aspect="equal")
            if titles:
                ax.set_title(titles[i])
        else:
            ax.axis('off')
    plt.show()



def draw_multiimg_points(imgs,kps,color=None,r=15,nrow=1):
    new_imgs=[]
    if imgs[0].ndim==3:
        color_list=[np.random.randint(0,256,3) for _ in range(len(kps[0]))]
    else:
        color_list=[np.random.rand(0.256) for _ in range(len(kps[0]))]
    for n,(img,kp) in enumerate(zip(imgs,kps)):
        new_img=img.copy()
        for i,k in enumerate(kp):
            if color is not None:
                color_list[i]=color
            c=color_list[i]
            c=tuple(int(xx) for xx in c)
            new_img=cv2.circle(new_img,tuple(np.round(k.pt).astype(int)),r,c,thickness=r//10+1)
        new_imgs.append(new_img)
    show_multi(new_imgs,nrow=nrow)
    plt.show()
