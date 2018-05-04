# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

def create_dir(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            
def process(x):
    # change range[0-255] to [-1,1]
    return x / 127.5 - 1.0
    
def process_transpose(x):
    return (x + 1.0) * 127.5

def image_save(path, x, y, gt, rows, cols):

    if len(x.shape) == 2:
        h, w = x.shape
        c = 1
        n = 1
    elif len(x.shape) == 3:
        h, w, c = x.shape
        n = 1
    else:
        n, h, w, c = x.shape
        
    image = np.zeros((rows*h, cols*3*w, c))
    # 填充图像
    for idx in range(n):
        r = idx // cols
        c = idx % cols
        image[h*r:h*(r+1),c*3*w:c*3*w+w,:] = x[idx]
        image[h*r:h*(r+1),c*3*w+w:c*3*w+2*w,:] = y[idx]
        image[h*r:h*(r+1),c*3*w+2*w:c*3*w+3*w,:] = gt[idx]
    # 转换成uint8格式
    image = image.astype(np.uint8)  
    cv2.imwrite(path, image)