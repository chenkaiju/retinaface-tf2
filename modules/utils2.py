import os
import cv2
import matplotlib.pyplot as plt
from absl import logging

from bfm.bfm import BFMModel
from modules.dataset2 import load_tfds_dataset

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def load_dataset(cfg, priors, load_train=True, load_valid=False):
    """load dataset"""
    logging.info("load dataset from {}".format(cfg['tfds_name']))
    
    bfm = BFMModel(
        bfm_fp=os.path.join("bfm", "bfm_noneck_v3.pkl"),
        shape_dim=40,
        exp_dim=10
    )
        
    dataset = load_tfds_dataset(
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
    
    return dataset


###############################################################################
#   Visulization                                                              #
###############################################################################
def draw_landmarks(image, pts, facebox, img_dim, outputPath):

    """
        image: uint8
        pts: float32 (2, 68) -> range[0, 1]
        facebox: float32 (4,) -> range[0, 1]
    """
    pts = pts * img_dim
    facebox = facebox * img_dim
    image = cv2.resize(image, (img_dim, img_dim), interpolation=cv2.INTER_AREA)
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    my_dpi = 100
    display_scale = 1 # suggested
    height, width = img.shape[:2]
    figure = plt.figure(figsize=(width / my_dpi, height / my_dpi))
    plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    
    for i in range(len(pts)):
        alpha = 0.8
        markersize = 1.5
        lw = 0.7 
        color = 'g'
        markeredgecolor = 'green'

        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

        # close eyes and mouths
        plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                color=color, lw=lw, alpha=alpha - 0.1)
        plot_close(41, 36)
        plot_close(47, 42)
        plot_close(59, 48)
        plot_close(67, 60)

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

            plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                        color=color,
                        markeredgecolor=markeredgecolor, alpha=alpha)
    
    x1, y1, x2, y2 = facebox        
    xs = [x1, x2, x2, x1, x1]
    ys = [y1, y1, y2, y2, y1]        
    plt.plot(xs, ys, color='red', lw=lw, alpha=alpha - 0.1)
    
    plt.savefig(outputPath, dpi=my_dpi*display_scale)
    #print('Save landmark result to {}'.format(wfp))
    plt.close()