# -*- coding: utf-8 -*-
import os
import cv2
import random
import numpy as np
from glob import glob

def load_data(data_dir):
    return glob(os.path.join(data_dir,'*'))

def next_batch(im_list, global_step, batch_size=16, re_size=256):
    max_batch = len(im_list) // batch_size
    nxt_batch = global_step % max_batch
    # 获取下一个batch
    batch_im = im_list[nxt_batch*batch_size:(nxt_batch+1)*batch_size]
    
    batch_x, batch_y = [], []
    for im_path in batch_im:
        im = cv2.imread(im_path)
        w = im.shape[1] // 2
        # 将batch分割成输出input和标签label
        inputs = cv2.resize(im[:,0:w,:], (re_size,re_size))
        labels = cv2.resize(im[:,w:,:], (re_size,re_size))
        
        batch_x.append(inputs)
        batch_y.append(labels)
        
    batch_x = np.array(batch_x, dtype=np.float32)
    batch_y = np.array(batch_y, dtype=np.float32)
    
    return batch_x, batch_y


def random_batch(im_list, sample_size, re_size=256):
    sample_size = min(sample_size, len(im_list))
    # 随机采样得到一个batch
    rand_im = random.sample(im_list, sample_size)
    
    rand_x, rand_y = [], []
    for im_path in rand_im:
        im = cv2.imread(im_path)
        w = im.shape[1] // 2
        # 将batch分割成输出input和标签label
        inputs = cv2.resize(im[:,0:w,:], (re_size,re_size))
        labels = cv2.resize(im[:,w:,:], (re_size,re_size))
        
        rand_x.append(inputs)
        rand_y.append(labels)
        
    rand_x = np.array(rand_x, dtype=np.float32)
    rand_y = np.array(rand_y, dtype=np.float32)
    
    return rand_x, rand_y