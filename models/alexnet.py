import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

from utils import LOG_DIR, flatten_convolution, dense_layer, get_activation_f

IMAGE_SIZE = 277
mnist = learn.datasets.load_dataset('mnist')


def alex_conv_pool_layer(tensor_in, n_filters, kernel_size, stride, pool_size, pool_stride,
                         activation_fn=tf.nn.tanh, padding='SAME'):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=n_filters,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=activation_fn,
                                           padding=padding)
    lrn = tf.nn.lrn(conv, bias=1.0, depth_radius=5, alpha=0.0001, beta=0.75)
    pool = tf.nn.max_pool(lrn, ksize=pool_size, strides=pool_stride, padding=padding)
    return pool


def alex_3_convs_pool_layer(tensor_in, activation_fn=tf.nn.tanh, padding='SAME'):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=384,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           padding=padding)
    conv = tf.contrib.layers.convolution2d(conv,
                                           num_outputs=384,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           padding=padding)
    conv = tf.contrib.layers.convolution2d(conv,
                                           num_outputs=256,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           padding=padding)
    pool = tf.nn.max_pool(conv, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding=padding)
    return pool


def alexnet_model(x, model_conf, keep_prob=0.5):
    activation_f = get_activation_f(model_conf['activation'])

    with tf.variable_scope('layer1'):
        layer1 = alex_conv_pool_layer(x, 96, [11, 11], 4, (1, 3, 3, 1), (1, 2, 2, 1), activation_fn=activation_f)

    with tf.variable_scope('layer2'):
        layer2 = alex_conv_pool_layer(layer1, 256, [5, 5], 2, (1, 3, 3, 1), (1, 2, 2, 1), activation_fn=activation_f)

    with tf.variable_scope('layer3'):
        layer3 = alex_3_convs_pool_layer(layer2, activation_fn=activation_f)
        layer3_flat = flatten_convolution(layer3)

    fc1 = dense_layer(layer3_flat, [4096, 4096], activation_fn=activation_f, keep_prob=keep_prob)
    fc2 = dense_layer(fc1, [10], activation_fn=None)
    return fc2
