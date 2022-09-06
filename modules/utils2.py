import os
import tensorflow as tf
import cv2
import numpy as np
from absl import logging

from bfm.bfm import BFMModel
from modules.dataset2 import load_tfds_dataset
from modules.anchor2 import decode_tf

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def load_dataset(cfg, priors, load_train=True, load_valid=False):
    """load dataset"""
    logging.info("load dataset from {}".format(cfg['tfds_name']))
    
    bfm = BFMModel(
        bfm_fp=os.path.join("bfm", "bfm_noneck_v3.pkl"),
        shape_dim=40,
        exp_dim=10
    )
        
    (train_dataset, train_data_num), (val_dataset, val_data_num) = load_tfds_dataset(
        bfm,
        load_train=load_train,
        load_valid=load_valid,
        dataset_dir=cfg['dataset_dir'],
        tfds_name=cfg['tfds_name'],
        batch_size=cfg['batch_size'],
        img_dim=cfg['input_size'],
        using_encoding=True,
        priors=priors,
        match_thresh=cfg['match_thresh'],
        ignore_thresh=cfg['ignore_thresh'],
        variances=cfg['variances'])
    
    return (train_dataset, train_data_num), (val_dataset, val_data_num), bfm


###############################################################################
#   Visulization                                                              #
############################################################################### 
def draw_result(image, pts, facebox, outputPath=None, color=(0,255,0)):

    """
        image: uint8
        pts: float32 (2, 68) -> range[0, 1]
        facebox: float32 (4,) -> range[0, 1]
        color: RGB order
    """
    
    
    img = image

    height, width = img.shape[:2]
    
    pts = (pts * height).astype(np.int64)
    facebox = (facebox * height).astype(np.int64)
    
    num_face = facebox.shape[0]
    
    if num_face>0:
        for i in range(1):
            alpha = 0.8
            markersize = 2
            lw = 1 
            color = color
            box_color = tuple(int(c*0.8) for c in color)

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]


            for idx in range(68):
                img = cv2.drawMarker(img, (pts[i][0, idx], pts[i][1, idx]), color=color, thickness=lw, 
                                     markerType=cv2.MARKER_DIAMOND, markerSize=5)
            
            # close eyes and mouths
            def plot_close(img, i1, i2):
                
                img = cv2.line(img, (pts[i][0, i1], pts[i][1, i1]), (pts[i][0, i2], pts[i][1, i2]), color=color, thickness=lw)
                return img
            
            img = plot_close(img, 41, 36)
            img = plot_close(img, 47, 42)
            img = plot_close(img, 59, 48)
            img = plot_close(img, 67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                points = np.transpose(pts[i][:,l:r])
                points = points.reshape((-1, 1, 2))
                img = cv2.polylines(img, [points], isClosed=False, color=color, thickness=lw)

        
            x1, y1, x2, y2 = facebox[i]
            bbox = np.array([(x1,y1), (x2,y1), (x2,y2), (x1,y2), (x1,y1)])
            img = cv2.drawContours(img, [bbox], 0, box_color, thickness=1)

        
        if (outputPath != None):
            img_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(outputPath, img_write)
        
    
    return img


def post_process_pred(predictions, priors, variances, iou_th, score_th):
    
    decode = decode_tf(predictions, priors, variances)
    selected_indices = tf.image.non_max_suppression(
        boxes=decode[:, :4],
        scores=decode[:, -1],
        max_output_size=tf.shape(decode)[0],
        iou_threshold=iou_th,
        score_threshold=score_th)
        
    out = tf.gather(decode, selected_indices)
    
    return out