#!/usr/bin/env python3
"""
Tests for the SparseConv2d Tensorflow operation.
"""

import unittest
import numpy as np
print('import tensorflow')
import tensorflow as tf
print('import libsparse_conv2d')
sparse_conv2d_m = tf.load_op_library('/root/repos/tensorflow/bazel-bin/tensorflow/core/user_ops/sparse_conv2d.so')


class InnerProductOpTest(unittest.TestCase):
    def test_sparse_conv2d(self):
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape=(1, 228, 228, 3))
            conv = sparse_conv2d_m.sparse_conv2d(x, [[0, 0]], [1.0], dense_shape=[11, 11, 3, 96], strides=[4, 4])

            self.assertListEqual([1, 55, 55, 96], conv.get_shape().as_list())

    def test_sparse_conv2d_simple(self):
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape=(1, 11, 11, 3))
            conv = sparse_conv2d_m.sparse_conv2d(x, [[0, 0]], [1.0], dense_shape=[11, 11, 3, 96], strides=[4, 4])

            inp = np.zeros((1, 11, 11, 3))
            out = sess.run(conv, feed_dict={x: inp})

            self.assertEqual(0, np.count_nonzero(out))

            inp[0][1][1][0] = 1
            out = sess.run(conv, feed_dict={x: inp})
            self.assertEqual(0, np.count_nonzero(out))

            inp[0][0][0][0] = 1
            out = sess.run(conv, feed_dict={x: inp})
            self.assertEqual(1, np.count_nonzero(out))


if __name__ == '__main__':
    unittest.main()
