# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from lib.utils import create_dir
from lib.utils import image_save
from lib.utils import process
from lib.utils import process_transpose
from lib.dataset import load_data
from lib.dataset import next_batch
from lib.dataset import random_batch
from lib.nets.pix2pix import Pix2Pix
from lib.nets.pix2pix_gan import Pix2PixGAN
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# 可供使用的网络
NETS = {
    'pix2pix' : Pix2Pix,
    'pix2pixgan' : Pix2PixGAN
}

def train_net(config):
    # 获取训练参数
    max_step = config.max_step      # 最大训练次数
    save_step = config.save_step    # 训练阶段保存步数
    vis_step = config.vis_step      # 训练阶段验证步数
    log_step = config.log_step      # 训练信息打印步数
    model_dir = config.model_dir    # 模型保存路径
    vis_dir = config.vis_dir        # 验证图片保存路径
    train_dir = config.train_dir    # 训练&验证数据集路径
    test_dir = config.test_dir      # 测试数据集路径
    batch_size = config.batch_size  # 网络batch大小
    image_size = config.image_size  # 网络输入大小
    net_name = config.net_name      # 训练网络
    
    # 获取训练与验证数据(图像路径)
    datas = load_data(train_dir)
    datas_val = load_data(test_dir)
    
    print('[INFO] Total data, #train: %d, #val: %d' % (len(datas), len(datas_val)))

    # 获取神经网络
    if net_name.lower() in NETS.keys():
        net = NETS[net_name.lower()](image_size, batch_size)
        print('[INFO] Use net: %s' % net.name)
    else:
        raise KeyError('[Error] Current version only support nets: %s, but get net: %s' \
                           % (', '.join(NETS.keys()), net_name))
    
    # 生成网络对应的文件夹
    dirs = [vis_dir, os.path.join(model_dir, net.name), os.path.join(vis_dir, net.name)]
    create_dir(dirs)

    loss_type = []
    loss_value = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化神经网络参数
        sess.run(tf.global_variables_initializer())
        # 训练神经网络
        for step in range(max_step):
            # 获取当前步数的训练数据batch
            batch_x, batch_y = next_batch(datas, step, batch_size)
            batch_x = process(batch_x)
            batch_y = process(batch_y)
            # 调用优化器，反向传播优化神经网络
            sess.run(net.train_items, feed_dict={net.inputs: batch_x, net.labels: batch_y})
            # 打印训练信息
            if step % log_step == 0:
                # 计算神经网络误差，并打印误差信息
                loss = sess.run(net.loss_items, feed_dict={net.inputs: batch_x, net.labels: batch_y})
                loss_str = []
                for key, value in loss.items():
                    loss_str.append(str(key) + ': ' + str(value))
                print('[Train] net: %s, step: %d, %s' % (net.name, step, ', '.join(loss_str)))
                # 保存当前误差，便于最后画出误差趋势图
                loss_value.append(list(loss.values()))
                if len(loss_type) == 0:
                    loss_type = list(loss.keys())
            # 保存模型
            if step % save_step == 0 and step != 0:
                saver.save(sess, os.path.join(model_dir, net.name, 'model_%s.cpkt' % net.name))
            # 验证过程，使用神经网络输出图片
            if step % vis_step == 0:
                test_x, test_y = random_batch(datas_val, 2)
                test_px = process(test_x)
                outputs = sess.run(net.samples, feed_dict={net.inputs: test_px})
                outputs = process_transpose(outputs)
                image_save(os.path.join(vis_dir, net.name, 'step_%d.jpg' % step), test_x, outputs, test_y, 2, 1)
          
    # 画出loss趋势图
    loss_value = np.array(loss_value)
    joblib.dump([loss_type, loss_value], os.path.join(model_dir,  net.name, 'loss.pkl'))
    for idx, ltype in enumerate(loss_type):
        plt.plot(range(loss_value.shape[0]), loss_value[:,idx])
        plt.savefig(os.path.join(model_dir,  net.name, '%s.jpg' % ltype)) 