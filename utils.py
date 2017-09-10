import sys
import logging

import numpy as np
import tensorflow as tf

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('requests').setLevel(logging.WARNING)

LOG_DIR = '/data/private/tf-lcnn-logs'


optimizers = {
    'adagrad': tf.train.AdagradOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'sgd': tf.train.GradientDescentOptimizer
}

logstep = {
    'mnist': {
        'validation': 2000,
        'training': 1000,
    },
    'mnist224': {
        'validation': 500,
        'training': 200,
    },
    'ilsvrc2012': {
        'validation': 10000,
        'training': 1000,
    }
}


def get_dataset_sizes(dataset_name):
    if dataset_name == 'mnist':
        class_size = 10
        dataset_size = 60000
    elif dataset_name == 'mnist224':
        class_size = 10
        dataset_size = 60000
    elif dataset_name == 'ilsvrc2012':
        class_size = 1000
        dataset_size = 1200000
    else:
        raise Exception('invalid dataset: %s' % dataset_name)
    return class_size, dataset_size


def flatten_convolution(tensor_in):
    tendor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
    return tensor_in_flat


def dense_layer(tensor_in, layers, activation_fn=tf.nn.tanh, keep_prob=None):
    tensor_out = tensor_in
    for idx, layer in enumerate(layers):
        tensor_out = tf.contrib.layers.fully_connected(tensor_out, layer,
                                                       activation_fn=activation_fn,
                                                       weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                                                       biases_initializer=tf.constant_initializer(0.001))
        tensor_out = tf.contrib.layers.dropout(tensor_out, keep_prob=keep_prob)

    return tensor_out


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
