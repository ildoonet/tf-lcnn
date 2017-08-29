import tensorflow as tf

from models.lenet import lenet_layer
from utils import flatten_convolution, dense_layer, get_activation_f


def alex_conv_pool_layer(tensor_in, n_filters, kernel_size, stride, pool_size, pool_stride,
                         bias_initializer=tf.zeros_initializer(),
                         activation_fn=tf.nn.tanh, padding='SAME'):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=n_filters,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=activation_fn,
                                           weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                           biases_initializer=bias_initializer,
                                           padding=padding)
    lrn = tf.nn.lrn(conv, bias=1.0, depth_radius=5, alpha=0.0001, beta=0.75)
    pool = tf.nn.max_pool(lrn, ksize=pool_size, strides=pool_stride, padding=padding)
    return pool


def alex_3_convs_pool_layer(tensor_in, activation_fn=tf.nn.tanh):
    conv = tf.contrib.layers.convolution2d(tensor_in,
                                           num_outputs=384,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                           biases_initializer=tf.zeros_initializer(),
                                           padding='SAME')
    conv = tf.contrib.layers.convolution2d(conv,
                                           num_outputs=384,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                           biases_initializer=tf.constant_initializer(0.001),
                                           padding='SAME')
    conv = tf.contrib.layers.convolution2d(conv,
                                           num_outputs=256,
                                           kernel_size=[3, 3],
                                           stride=1,
                                           activation_fn=activation_fn,
                                           weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                           biases_initializer=tf.constant_initializer(0.001),
                                           padding='SAME')
    pool = tf.nn.max_pool(conv, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
    return pool


def alexnet_model(x, class_size, conv='lcnn', model_conf=None, keep_prob=0.5):
    activation_f = get_activation_f(model_conf['activation'])

    with tf.variable_scope('layer1'):
        layer1 = alex_conv_pool_layer(x, 96, [11, 11], 4, (1, 3, 3, 1), (1, 2, 2, 1), activation_fn=activation_f,
                                      bias_initializer=tf.zeros_initializer(), padding='SAME')

    with tf.variable_scope('layer2'):
        layer2 = alex_conv_pool_layer(layer1, 256, [5, 5], 2, (1, 3, 3, 1), (1, 2, 2, 1), activation_fn=activation_f,
                                      bias_initializer=tf.constant_initializer(0.001), padding='SAME')

    with tf.variable_scope('layer3'):
        layer3 = alex_3_convs_pool_layer(layer2, activation_fn=activation_f)
        layer3_flat = flatten_convolution(layer3)

    # alexnet v1
    with tf.variable_scope('fc'):
        fc1 = dense_layer(layer3_flat, [4096, 4096], activation_fn=activation_f, keep_prob=keep_prob)
        fc2 = tf.contrib.layers.fully_connected(fc1, class_size,
                                                activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                biases_initializer=tf.zeros_initializer())

    return fc2
