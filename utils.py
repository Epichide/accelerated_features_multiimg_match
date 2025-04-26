#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: Epichide
# @Email: no email
# @Time: 2025/4/27 1:42
# @File: utils.py
# @Software: PyCharm
import cv2
import numpy as np
import os
import  time
import  cv2
import matplotlib.pyplot as plt
import  numpy as np
import torch
from modules.xfeat import XFeat
from scipy.spatial import cKDTree,distance


def process_intersect(idxs_list_s):
    B=len(idxs_list_s)
    idx_refs=set([idxss.item() for idxss in idxs_list_s[0][1]])
    for idxs_list in idxs_list_s:
        idx_refs=idx_refs.intersection(set([idxss.item() for idxss in idxs_list[1]]))
    batched_matches=[]
    for idxs_list in idxs_list_s:
        idx1s,idx2s=idxs_list
        new_idxs=[]
        for idx1,idx2 in zip(idx1s,idx2s):
            if idx2.item() in idx_refs:
                new_idxs.append((idx1.item(),idx2.item()))
        #sort
        new_idxs.sort(key=lambda x:x[1])
        new_idx1s,new_idx2s=zip(*new_idxs)
        batched_matches.append((list(new_idx1s),list(new_idx2s)))
    return batched_matches

def _ensure_spacing(coord,spacing=1,p_norm=np.inf,max_out=None,return_idx=True):
    """
    Ensure that the coordinates are spaced apart by at least the specified spacing.
    :param coord: Coordinates to check.
    :param spacing: Minimum spacing between coordinates.
    :param p_norm: Norm to use for distance calculation.
    :param max_out: Maximum number of coordinates to return.
    :param return_idx: Whether to return indices of the coordinates.
    :return: Spaced coordinates or indices of spaced coordinates.
    """
    if max_out is None:
        max_out = len(coord)
    tree = cKDTree(coord)
    indices=tree.query_ball_point(coord,r=spacing,p=p_norm)
    rejected_peaks = set()
    idxs=[True]*len(coord)
    naccepted=0
    for idx, candidates in enumerate(indices):
        if idx not in rejected_peaks:
            candidates.remove(idx)
            dist=distance.cdist([coord[idx]],coord[candidates],"minkowski",p=p_norm).reshape(-1)
            candidates=[c for c,d in zip(candidates,dist) if d<spacing]
            for candi in candidates:
                idxs[candi]=False
            rejected_peaks.update(candidates)
            naccepted+=1
            if max_out is not None and naccepted>=max_out:
                break
    #remove the too close points
    output=np.delete(coord,tuple(rejected_peaks),axis=0)
    if max_out is not None:
        output=output[:max_out]
    if return_idx:
        return output,np.array(idxs)
    return output


    dist, idx = tree.query(coord, k=max_out, p=p_norm)
    mask = dist[:, 1:] > spacing
    mask = np.all(mask, axis=1)
    if return_idx:
        return np.where(mask)[0]
    else:
        return coord[mask]

def check_edge(keypoints, radius, h, w):
    return np.logical_and(
        np.logical_and(keypoints[:, 0] > radius, keypoints[:, 0] < w - radius),
        np.logical_and(keypoints[:, 1] > radius, keypoints[:, 1] < h - radius)
    )



def remove_points(keypointss, refpoints,
                  radius, h, w,disth=1):
    ransac_thr=4.0
    final_inliers=np.ones_like(refpoints[:,0],dtype=bool)
    for i, keypoints in enumerate(keypointss):
        H,inliers=cv2.findHomography(refpoints.reshape(-1,2,1),
                                     keypoints.reshape(-1,2,1),
                                     cv2.RANSAC, ransac_thr,
                                     maxIters=1000,confidence=0.995)
        final_inliers=np.logical_and(final_inliers,inliers.flatten()>0)
    final_keypoints=[keypoints[final_inliers] for keypoints in keypointss]
    # remove edge points
    mask_wh=np.ones_like(final_keypoints[0][:,0],dtype=bool)
    for i, keypoints in enumerate(final_keypoints):
        mask_wh=np.logical_and(mask_wh,
                                check_edge(keypoints,radius,h,w))
    final_keypoints=[keypoints[mask_wh] for keypoints in final_keypoints]
    #remove near points
    _,refidxs=_ensure_spacing(final_keypoints[0],spacing=radius*disth)
    final_refpoints=final_keypoints[0][refidxs]
    final_keypointss=[keypoints[refidxs] for keypoints in final_keypoints]
    return final_keypointss



    pass


def match_multi_by_star(imgs, xfeat, refidx=0, radius=256,
                        scale=8,top_k=8182,disth=1):
    """
    Match multiple images using XFeat.
    :param imgs: List of images to match.
    :param xfeat: XFeat model.
    :param refidx: Index of the reference image.
    :param radius: Radius for matching.
    :param scale: Scale for matching.
    :return: List of matches for each image.
    """
    h,w=imgs[refidx].shape[:2]
    newh,neww=h//scale*scale,w//scale*scale
    scaledh,scaledw=h//scale,w//scale
    newradius=radius//scale
    imgs=[img[:newh,:neww] for img in imgs]
    scale_imgs=[cv2.resize(img,(scaledw,scaledh)) for img in imgs]
    # compute features
    imgsets=[xfeat.parse_input(img) for img in scale_imgs]
    imgfeats=[xfeat.detectAndComputeDense(imgset, top_k=top_k) for imgset in imgsets]
    reffeat=imgfeats[refidx]
    idxs_list_s=[xfeat.batch_match(feat1["descriptors"],reffeat["descriptors"])[0] for feat1 in imgfeats]
    intersect_idx_list=process_intersect(idxs_list_s)
    matches_s=[]
    B=len(imgs)
    mask_good=torch.ones(len(intersect_idx_list[0][0]),dtype=torch.bool)
    for b in range(B):
        idxs_list=intersect_idx_list[b]
        feat1=imgfeats[b]
        matches,mask_good_=xfeat.refine_matches(feat1,reffeat,matches=[idxs_list],batch_idx=0,return_idx=True)
        mask_good=mask_good&mask_good_
        matches_s.append(matches)
    final_matches_s=[matches[mask_good] for matches in matches_s]

    # removepoint
    keypointss=[]
    for matches in matches_s:
        points=matches[:,:2].detach().cpu().numpy()
        keypointss.append(points)
    refpoints=keypointss[refidx]
    final_keypointss= remove_points(keypointss,refpoints,
                                    newradius,scaledh,scaledw,disth=disth)
    final_keypointss=[keypoints*scale for keypoints in final_keypointss]
    return final_keypointss

def imread(imgfile):
    """
    Read an image file and convert it to a numpy array.
    """
    print(os.path.abspath(imgfile))
    img = cv2.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image file: {imgfile}")
    return img
