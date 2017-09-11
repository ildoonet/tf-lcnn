import tensorflow as tf

from layers.LookupConvolution2d import lookup_conv2d
from utils import flatten_convolution


def alex_conv_pool_layer(tensor_in, n_filters, kernel_size, stride, pool_size, pool_stride, param_lambda,
                         bias_initializer=tf.zeros_initializer(),
                         activation_fn=tf.nn.relu, padding='SAME', convtype='conv', dict_size=None, init_sparsity=None):
    if convtype == 'lcnn':
        conv = lookup_conv2d(tensor_in,
                             dict_size=dict_size,
                             initial_sparsity=init_sparsity,
                             param_lambda=param_lambda,
                             stride=stride,
                             num_outputs=n_filters,
                             kernel_size=kernel_size,
                             activation_fn=activation_fn,
                             biases_initializer=bias_initializer,
                             padding=2)
    else:
        conv = tf.contrib.layers.convolution2d(tensor_in,
                                               num_outputs=n_filters,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               activation_fn=activation_fn,
                                               weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                               biases_initializer=bias_initializer,
                                               padding=padding)
        conv = tf.nn.lrn(conv, bias=1.0, depth_radius=5, alpha=0.0001, beta=0.75)
    pool = tf.nn.max_pool(conv, ksize=pool_size, strides=pool_stride, padding=padding)
    return pool


def alexnet_model(x, class_size, convtype='lcnn', model_conf=None, keep_prob=0.5):
    with tf.variable_scope('layer1'):
        layer1 = alex_conv_pool_layer(x, 64, [11, 11], 4, (1, 3, 3, 1), (1, 2, 2, 1), activation_fn=tf.nn.relu,
                                      bias_initializer=tf.zeros_initializer(), padding='SAME', convtype=convtype,
                                      init_sparsity=model_conf['initial_sparsity'][0] if model_conf['initial_sparsity'] else None,
                                      dict_size=model_conf['dictionary'][0] if model_conf['dictionary'] else None,
                                      param_lambda=model_conf['lambda'])

    with tf.variable_scope('layer2'):
        layer2 = alex_conv_pool_layer(layer1, 192, [5, 5], 1, (1, 3, 3, 1), (1, 2, 2, 1), activation_fn=tf.nn.relu,
                                      bias_initializer=tf.constant_initializer(0.001), padding='SAME', convtype=convtype,
                                      init_sparsity=model_conf['initial_sparsity'][1] if model_conf['initial_sparsity'] else None,
                                      dict_size=model_conf['dictionary'][1] if model_conf['dictionary'] else None,
                                      param_lambda=model_conf['lambda'])

    if convtype == 'lcnn':
        with tf.variable_scope('layer3'):
            conv = lookup_conv2d(layer2,
                                 dict_size=model_conf['dictionary'][2],
                                 initial_sparsity=model_conf['initial_sparsity'][2],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=384,
                                 kernel_size=[3, 3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.zeros_initializer(),
                                 padding=1)
        with tf.variable_scope('layer4'):
            conv = lookup_conv2d(conv,
                                 dict_size=model_conf['dictionary'][3],
                                 initial_sparsity=model_conf['initial_sparsity'][3],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=256,
                                 kernel_size=[3, 3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.001),
                                 padding=1)
        with tf.variable_scope('layer5'):
            conv = lookup_conv2d(conv,
                                 dict_size=model_conf['dictionary'][4],
                                 initial_sparsity=model_conf['initial_sparsity'][4],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=256,
                                 kernel_size=[3, 3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.001),
                                 padding=1)
            pool = tf.nn.max_pool(conv, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
        with tf.variable_scope('layer6'):
            conv = lookup_conv2d(pool,
                                 dict_size=model_conf['dictionary'][5],
                                 initial_sparsity=model_conf['initial_sparsity'][5],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=4096,
                                 kernel_size=pool.get_shape().as_list()[1:3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.001),
                                 padding=0)
            conv = tf.contrib.layers.dropout(conv, keep_prob=keep_prob)
        with tf.variable_scope('layer7'):
            conv = lookup_conv2d(conv,
                                 dict_size=model_conf['dictionary'][6],
                                 initial_sparsity=model_conf['initial_sparsity'][6],
                                 param_lambda=model_conf['lambda'],
                                 stride=1,
                                 num_outputs=4096,
                                 kernel_size=conv.get_shape().as_list()[1:3],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.001),
                                 padding=0)
            conv = tf.contrib.layers.dropout(conv, keep_prob=keep_prob)
    else:
        with tf.variable_scope('layer3'):
            conv = tf.contrib.layers.convolution2d(layer2,
                                                   num_outputs=384,
                                                   kernel_size=[3, 3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                   biases_initializer=tf.zeros_initializer(),
                                                   padding='SAME')
        with tf.variable_scope('layer4'):
            conv = tf.contrib.layers.convolution2d(conv,
                                                   num_outputs=256,
                                                   kernel_size=[3, 3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                   biases_initializer=tf.constant_initializer(0.001),
                                                   padding='SAME')
        with tf.variable_scope('layer5'):
            conv = tf.contrib.layers.convolution2d(conv,
                                                   num_outputs=256,
                                                   kernel_size=[3, 3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                   biases_initializer=tf.constant_initializer(0.001),
                                                   padding='SAME')
            pool = tf.nn.max_pool(conv, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
        with tf.variable_scope('layer6'):
            conv = tf.contrib.layers.convolution2d(pool,
                                                   num_outputs=4096,
                                                   kernel_size=pool.get_shape().as_list()[1:3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                                                   biases_initializer=tf.constant_initializer(0.001),
                                                   padding='VALID')
            conv = tf.contrib.layers.dropout(conv, keep_prob=keep_prob)
        with tf.variable_scope('layer7'):
            conv = tf.contrib.layers.convolution2d(conv,
                                                   num_outputs=4096,
                                                   kernel_size=conv.get_shape().as_list()[1:3],
                                                   stride=1,
                                                   activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                                                   biases_initializer=tf.constant_initializer(0.001),
                                                   padding='VALID')
            conv = tf.contrib.layers.dropout(conv, keep_prob=keep_prob)

    with tf.variable_scope('layer8'):
        flatten = flatten_convolution(conv)
        output = tf.contrib.layers.fully_connected(flatten, class_size,
                                                   activation_fn=None,
                                                   weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                   biases_initializer=tf.zeros_initializer())

    return output
