import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers.convolutional import _Conv


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

        # activation for kernel weight
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
                     data_format='channels_last',
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
        data_format=data_format,
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
    return layer.apply(inputs)


def lookup_conv2d(tensor_in, num_outputs, kernel_size, stride, dict_size, padding='same',
                  param_lambda=0.1,
                  initial_sparsity=None, activation_fn=None):

    if not initial_sparsity:
        initial_sparsity = 0.5
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    sparse_th = initial_sparsity / math.sqrt(kernel_size[0] * kernel_size[1] * dict_size)

    pool_conv = slim.convolution2d(inputs=tensor_in, num_outputs=dict_size, kernel_size=[1, 1], stride=1,
                                   padding=padding,
                                   activation_fn=None,
                                   biases_initializer=None,
                                   scope='/pool_conv')
    align_conv = lookupalign_conv(inputs=pool_conv, filters=num_outputs, kernel_size=kernel_size,
                                  strides=(stride[0], stride[1]), padding='valid',
                                  param_lambda=param_lambda * sparse_th,
                                  sparse_th=sparse_th,
                                  activation=activation_fn,
                                  name='align_conv')

    return align_conv
