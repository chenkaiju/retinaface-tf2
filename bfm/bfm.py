# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import os.path as osp
import numpy as np
import pickle

FILE_DIR = osp.dirname(osp.realpath(__file__))

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))
    
def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class BFMModel(object):
    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        bfm = _load(bfm_fp)
        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
            self.tri = _load(osp.join(FILE_DIR, 'tri.pkl'))  # this tri/face is re-built for bfm_noneck_v3
        else:
            self.tri = bfm.get('tri')

        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
