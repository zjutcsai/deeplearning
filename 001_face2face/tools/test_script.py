# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from .train_script import NETS
from lib.utils import process
from lib.utils import process_transpose

def test_net(config):
    batch_size = config.batch_size
    image_size = config.image_size
    model_dir = config.model_dir
    net_name = config.net_name
    test_video = config.test_video
    
    # 获取神经网络
    if net_name.lower() in NETS.keys():
        net = NETS[net_name.lower()](image_size, batch_size)
        print('[Info] Use net: %s' % net.name)
    else:
        raise KeyError('[Error] Current version only support nets: %s, but get net: %s' \
                             % (', '.join(NETS.keys()), net_name))
    
    # 测试阶段
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载训练好的模型
        ckpt = tf.train.get_checkpoint_state(os.path.join(model_dir, net.name))
        if ckpt is None:
            raise IOError('[Error] Trained model does not exists.')
        else:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        im = cv2.imread(test_video)
        im = im[:,0:256,:]
        im = process(im)
        im = im.reshape(1,256,256,3)
        
        result = sess.run(net.sample, feed_dict={net.inputs: im})
        
        result = process_transpose(np.squeeze(result))
        
        result = np.array(result, dtype=np.uint8)
        
        cv2.imshow('aaa', result)
        cv2.waitKey(5000)