# -*- coding: utf-8 -*-
import tensorflow as tf
from .layers import add_conv2d
from .layers import add_lrelu
from .layers import add_batchnorm
from .layers import add_deconv2d
from .layers import add_dropout
from .layers import add_relu
from .layers import add_tanh

class Pix2Pix:
    learning_rate = 1e-4
    ngf = 64
    
    def __init__(self, image_size, batch_size):
        self.m_dim = image_size
        self.c_dim = 3
        self.name = 'Pix2Pix'
        self.batch_size = batch_size
        self.build_graph()

    def build_graph(self):
        # 输入和标签placeholder
        with tf.variable_scope('data'):
            self.inputs = tf.placeholder(tf.float32, [None,self.m_dim,self.m_dim,self.c_dim])
            self.labels = tf.placeholder(tf.float32, [None,self.m_dim,self.m_dim,self.c_dim])
        # 自编码器输出，包含训练阶段输出与测试阶段输出
        with tf.name_scope('net'):
            mvars, self.outputs = self.build_net(self.inputs, is_training=True)
            _,     self.samples = self.build_net(self.inputs, is_training=False, reuse=True)
        # 网络损失函数
        with tf.name_scope('loss'):
            loss = 0.5 * tf.sqrt(tf.nn.l2_loss(self.outputs-self.labels)) / self.batch_size
        # 优化器，默认Adam
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss, var_list=mvars)
            net_train = optimizer.apply_gradients(grads_and_vars)
        # 平均移动，对loss对平滑
        with tf.name_scope('average_move'):
            with tf.control_dependencies([net_train]):
                ema = tf.train.ExponentialMovingAverage(decay=0.99)
                update_loss = ema.apply([loss])
        
        self.loss_items = {'net_loss' : ema.average(loss)}
        self.train_items = update_loss
        
    def build_net(self, inputs, is_training, reuse=None):
        with tf.variable_scope('unet', reuse=reuse) as scope:
            layers = []
            # encoder_1: [batch,256,256,c_dim] ==> [batch,256,256,ngf]
            with tf.variable_scope('encoder_1'):
                output =  add_conv2d(inputs, self.ngf)
                layers.append(output)  
            layer_specs = [
                self.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                self.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                self.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                self.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                self.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                self.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
                self.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
            ]
            for idx, channels in enumerate(layer_specs):
                with tf.variable_scope('encoder_%d' % (idx+2)):
                    # leaky relu, convolve and batch normalization
                    rectified = add_lrelu(layers[-1], 0.2)
                    convolved = add_conv2d(rectified, channels)
                    output = add_batchnorm(convolved, is_training)
                    layers.append(output)
                
            layer_specs = [
                (self.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (self.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (self.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (self.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                (self.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (self.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
                (self.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            ]
            num_encoder_layers = len(layers)
            for decoder_layer, (channels, drop) in enumerate(layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                with tf.variable_scope('decoder_%d' % (skip_layer+1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        inputs_ = layers[-1]
                    else:
                        inputs_ = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                    # leaky relu, deconvolve and batch normalization
                    rectified = add_lrelu(inputs_, 0.2)
                    deconved = add_deconv2d(rectified, channels)
                    output = add_batchnorm(deconved, is_training)   
                    if drop > 0.0:
                        output = add_dropout(output, 1-drop, is_training)      
                    layers.append(output)
            
            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, 3]
            with tf.variable_scope('decoder_1'):
                inputs_ = tf.concat([layers[-1], layers[0]], axis=3)
                rectified = add_relu(inputs_)
                output = add_deconv2d(rectified, 3)
                output = add_tanh(output)
                layers.append(output)
                
        mvars = tf.contrib.framework.get_trainable_variables(scope)    
        return mvars, layers[-1]