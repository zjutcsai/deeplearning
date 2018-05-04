# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops

def add_conv2d(inputs, num_outputs, kernel_size=4, stride=2, padding='SAME', activation_fn=None):
    weights_initializer = initializers.xavier_initializer()
    bias_initializer = init_ops.zeros_initializer()
    x = slim.conv2d(inputs, num_outputs, kernel_size, stride, padding=padding,
                    activation_fn=activation_fn, 
                    weights_initializer=weights_initializer, 
                    biases_initializer=bias_initializer)
    return x

def add_deconv2d(inputs, num_outputs, kernel_size=4, stride=2, padding='SAME', activation_fn=None):
    x = slim.conv2d_transpose(inputs, num_outputs, kernel_size, stride, padding,activation_fn=activation_fn)
    return x

def add_relu(x):
    return tf.nn.relu(x)
    
def add_lrelu(x, negatice_slop=0.2):
    return tf.maximum(x, x*negatice_slop)

def add_fc(x, num_outputs, activation_fn=None):
    weights_initializer = initializers.xavier_initializer()
    bias_initializer = init_ops.zeros_initializer()
    return slim.fully_connected(x, num_outputs, 
                                activation_fn=activation_fn,
                                weights_initializer=weights_initializer,
                                biases_initializer=bias_initializer)

def add_lrn(x, depth_radius=5, bias=2, alpha=0.0005, beta=0.75):
    return tf.nn.local_response_normalization(x, depth_radius, bias, alpha, beta)

def add_maxpool(x, kernel_size, strides=2, padding='SAME'):
    return slim.max_pool2d(x, kernel_size, strides, padding)

def add_flatten(x):
    return slim.flatten(x)

def add_dropout(x, keep_prob=0.5, is_training=True):
    return slim.dropout(x, keep_prob=keep_prob, is_training=is_training)

def add_tanh(x):
    return tf.nn.tanh(x)

def add_sigmoid(x):
    return tf.nn.sigmoid(x)
    
def add_batchnorm(x, is_training=True):
    return slim.batch_norm(inputs=x,
                           decay=0.9,
                           center=True,
                           scale=True,
                           epsilon=1e-5,
                           updates_collections=None,
                           is_training=is_training)