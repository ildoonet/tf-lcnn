import math
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers.convolutional import _Conv

try:
    sparse_conv2d_m = tf.load_op_library('/root/repos/tensorflow/bazel-bin/tensorflow/core/user_ops/sparse_conv2d.so')
except Exception as e:
    logging.warning(str(e))

dense_layers = {}
dense_weights = {}


class LookupAlignConvolution2d(_Conv):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 param_lambda=1.0,
                 sparse_th=0.01,
                 kernel_initializer=None,
                 bias_initializer=slim.init_ops.zeros_initializer(),
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        # L1 Regularizer
        kernel_regularizer = slim.l1_regularizer(scale=param_lambda)

        # initialize
        super(LookupAlignConvolution2d, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            name=name, **kwargs)
        self.sparse_th = sparse_th
        self.kernel_pre = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis].value
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        # dense kernel
        self.kernel_pre = self.add_variable(name='kernel_pre',
                                            shape=kernel_shape,
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            trainable=True,
                                            dtype=self.dtype)
        conv_th = tf.ones_like(self.kernel_pre) * self.sparse_th
        conv_zero = tf.zeros_like(self.kernel_pre)
        cond = tf.less(tf.abs(self.kernel_pre), conv_th)
        self.kernel = tf.where(cond, conv_zero, self.kernel_pre, name='kernel')

        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={channel_axis: input_dim})
        self.built = True


def lookupalign_conv(inputs,
                     filters,
                     kernel_size,
                     strides=(1, 1),
                     padding='valid',
                     dilation_rate=(1, 1),
                     activation=None,
                     use_bias=True,
                     kernel_initializer=None,
                     bias_initializer=slim.init_ops.zeros_initializer(),
                     param_lambda=1.0,
                     sparse_th=0.01,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     trainable=True,
                     name=None,
                     reuse=None):
    layer = LookupAlignConvolution2d(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format='channels_last',
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        param_lambda=param_lambda,
        sparse_th=sparse_th,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs), layer


def lookup_conv2d(tensor_in, num_outputs, kernel_size, stride, dict_size, padding=1,
                  param_lambda=0.3,
                  initial_sparsity=None, activation_fn=None,
                  biases_initializer=slim.init_ops.zeros_initializer()):

    if not initial_sparsity:
        initial_sparsity = 0.5
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    sparse_th = initial_sparsity / math.sqrt(kernel_size[0] * kernel_size[1] * dict_size)
    stddev = 1./math.sqrt(kernel_size[0] * kernel_size[1] * dict_size)

    padded = tf.pad(tensor_in, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
    pool_conv = slim.convolution2d(inputs=padded, num_outputs=dict_size, kernel_size=[1, 1], stride=1,
                                   padding='SAME',
                                   activation_fn=None,
                                   biases_initializer=None,
                                   scope='pool_conv')

    scope = tf.get_default_graph().get_name_scope()
    gen_sparse_conv = False
    if len(dense_weights.keys()) > 0:
        kernel_dense = dense_weights['%s/%s' % (scope, 'kernel_dense')]
        density = np.count_nonzero(kernel_dense) / kernel_dense.size
        if density < 0.15:
            gen_sparse_conv = True

    # activation for kernel weight
    if gen_sparse_conv:
        dense_kernel_shp = dense_weights['%s/%s' % (scope, 'kernel_shape')]
        dense_kernel_idx = dense_weights['%s/%s' % (scope, 'kernel')].indices
        dense_kernel_val = dense_weights['%s/%s' % (scope, 'kernel')].values
        dense_bias = tf.constant(dense_weights['%s/%s' % (scope, 'bias')])
        mode = 'custom_op'

        if mode == 'tf_op':
            # sparse convolution using only tensorflow's operations. -- SLOW!
            # im2col - image patche matrix
            img2col = tf.extract_image_patches(pool_conv, [1, kernel_size[0], kernel_size[1], 1], [1, stride[0], stride[1], 1], [1, 1, 1, 1], 'VALID')
            img2col = tf.transpose(img2col, [0, 3, 1, 2])
            img2col_shape = img2col.get_shape().as_list()
            img2col = tf.reshape(img2col, [img2col_shape[1], img2col_shape[2] * img2col_shape[3]])

            # sparse kernel & bias
            sparse_kernel = tf.SparseTensor(dense_kernel_idx, dense_kernel_val, dense_kernel_shp)

            # multiplication
            matmul = tf.sparse_tensor_dense_matmul(sparse_kernel, img2col)
            matmul = tf.transpose(matmul)
            matmul = tf.reshape(matmul, [1, img2col_shape[2], img2col_shape[3], dense_kernel_shp[0]])

            # bias & activation
            output = tf.nn.bias_add(matmul, dense_bias) if dense_bias is not None else matmul
            output = tf.nn.relu(output)
            return output
        elif mode == 'custom_op':
            conv = sparse_conv2d_m.sparse_conv2d(pool_conv, dense_kernel_idx, dense_kernel_val, dense_shape=dense_kernel_shp, strides=stride)
            output = tf.nn.bias_add(conv, dense_bias) if dense_bias is not None else conv
            output = tf.nn.relu(output)
            return output
        else:
            raise
    else:
        # dense convolution
        align_conv, layer = lookupalign_conv(inputs=pool_conv, filters=num_outputs, kernel_size=kernel_size,
                                             strides=(stride[0], stride[1]), padding='valid',
                                             param_lambda=param_lambda * sparse_th,
                                             sparse_th=sparse_th,
                                             activation=activation_fn,
                                             kernel_initializer=tf.random_uniform_initializer(-1 * stddev, stddev),
                                             bias_initializer=biases_initializer,
                                             name='align_conv')

        scope = tf.get_default_graph().get_name_scope()
        dense_layers[scope] = layer

        return align_conv


def extract_dense_weights(sess):
    for key in dense_layers.keys():
        layer = dense_layers[key]

        # sparse kernel
        dense_kernel = layer.kernel
        dense_kernel_shape = dense_kernel.get_shape().as_list()
        # dense_kernel = tf.reshape(dense_kernel, [dense_kernel_shape[0] * dense_kernel_shape[1] * dense_kernel_shape[2],
        #                                          dense_kernel_shape[3]])
        # dense_kernel = tf.transpose(dense_kernel)
        idx = tf.where(tf.not_equal(dense_kernel, 0))
        sparse_kernel = tf.SparseTensor(idx, tf.gather_nd(dense_kernel, idx), dense_kernel.get_shape())

        if layer.bias is not None:
            dk, k, b = sess.run([dense_kernel, sparse_kernel, layer.bias])
        else:
            dk, k = sess.run([dense_kernel, sparse_kernel])
            b = None
        dense_weights['%s/%s' % (key, 'kernel_dense')] = dk
        dense_weights['%s/%s' % (key, 'kernel')] = k
        dense_weights['%s/%s' % (key, 'kernel_shape')] = dense_kernel_shape
        dense_weights['%s/%s' % (key, 'bias')] = b
