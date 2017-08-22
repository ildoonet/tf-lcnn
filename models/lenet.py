import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from layers.LookupConvolution2d import lookup_conv2d
from utils import dense_layer, flatten_convolution, get_activation_f

IMAGE_SIZE = 28
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def lenet_layer(tensor_in, n_filters, kernel_size, pool_size, activation_fn=tf.nn.tanh, padding='SAME'):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=n_filters,
                                           kernel_size=kernel_size,
                                           activation_fn=activation_fn,
                                           padding=padding)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_size, padding=padding)
    return pool


def lookup_layer(tensor_in, n_filters, kernel_size, dict_size, pool_size, stride=1,
                 init_sparsity=0.5, activation_fn=tf.nn.tanh, padding='SAME'):
    conv = lookup_conv2d(tensor_in,
                         dict_size=dict_size,
                         stride=stride,
                         num_outputs=n_filters,
                         kernel_size=kernel_size,
                         activation_fn=activation_fn,
                         initial_sparsity=init_sparsity,
                         padding=padding)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_size, padding=padding)
    return pool


def lenet_model(X, image_size=(-1, IMAGE_SIZE, IMAGE_SIZE, 1), pool_size=(1, 2, 2, 1),
                conv='lcnn', model_conf=None, keep_prob=0.5):
    X = tf.reshape(X, image_size)

    activation_f = get_activation_f(model_conf['activation'])

    with tf.variable_scope('layer1'):
        """
        Valid:
         * input: (?, 28, 28, 1)
         * filter: (5, 5, 1, 4)
         * pool: (1, 2, 2, 1)
         * output: (?, 12, 12, 4)
        Same:
         * input: (?, 28, 28, 1)
         * filter: (5, 5, 1, 4)
         * pool: (1, 2, 2, 1)
         * output: (?, 14, 14, 4)
        """
        if conv == 'normal':
            layer1 = lenet_layer(X, 4, [5, 5], pool_size, activation_fn=activation_f)
        elif conv == 'lcnn':
            layer1 = lookup_layer(X, 4, [5, 5], dict_size=model_conf['dictionary'][0],
                                  init_sparsity=model_conf['initial_sparsity'][0],
                                  pool_size=pool_size, activation_fn=activation_f)
        else:
            raise

    with tf.variable_scope('layer2'):
        """
        VALID:
         * input: (?, 12, 12, 4)
         * filter: (5, 5, 4, 6)
         * pool: (1, 2, 2, 1)
         * output: (?, 4, 4, 6)
         * flat_output: (?, 4 * 4 * 6)
        SAME:
         * input: (?, 14, 14, 4)
         * filter: (5, 5, 4, 6)
         * pool: (1, 2, 2, 1)
         * output: (?, 7, 7, 6)
         * flat_output: (?, 7 * 7 * 6)
        """
        if conv == 'normal':
            layer2 = lenet_layer(layer1, 6, [5, 5], pool_size, activation_fn=activation_f)
        elif conv == 'lcnn':
            layer2 = lookup_layer(layer1, 6, [5, 5], dict_size=model_conf['dictionary'][1],
                                  init_sparsity=model_conf['initial_sparsity'][1],
                                  pool_size=pool_size, activation_fn=activation_f)
        else:
            raise
        layer2_flat = flatten_convolution(layer2)

    fc1 = dense_layer(layer2_flat, [1024], activation_fn=activation_f, keep_prob=keep_prob)
    fc2 = dense_layer(fc1, [10], activation_fn=None)
    return fc2
