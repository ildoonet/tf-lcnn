import sys
import os
import argparse
import cv2
import time
import yaml
import numpy as np
import logging
import tensorflow as tf
from tensorflow.python.client import timeline

from layers.LookupConvolution2d import extract_dense_weights
from utils import get_dataset_sizes
from networks.alexnet import alexnet_model

config = tf.ConfigProto(
    allow_soft_placement=False,
    log_device_placement=False,
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Inference using LCNN.')
    parser.add_argument('--path', default='./models/alexnet/mnist/lcnn-fast/', help='configuration file path')
    # parser.add_argument('--path', default='/Users/ildoonet/Downloads/lcnn-fast/', help='configuration file path')
    parser.add_argument('--imgpath', type=str, default='./images/mnist_5.jpg')
    parser.add_argument('--benchmark', type=int, default=10)
    parser.add_argument('--save', type=bool, default=False)

    args = parser.parse_args()

    # load config
    logging.info('config path : %s' % args.path)
    with open(os.path.join(args.path, 'conf.json'), 'r') as stream:
        conf = yaml.load(stream)
    class_size, _ = get_dataset_sizes(conf['dataset'])
    model_conf = {key: conf.get(key, []) for key in ['initial_sparsity', 'dictionary', 'lambda']}

    # placeholders
    if conf['dataset'] == 'mnist':
        image_w = image_h = 24
        image_ch = cv2.IMREAD_GRAYSCALE
    elif conf['dataset'] == 'mnist224':
        image_w = image_h = 224
        image_ch = cv2.IMREAD_GRAYSCALE
    elif conf['dataset'] == 'ilsvrc2012':
        image_w = image_h = 224
        image_ch = cv2.IMREAD_COLOR
    else:
        raise Exception('invalid dataset: %s' % args.dataset)

    # read image & resize & center-crop to input size
    logging.info('load image')
    img = cv2.imread(args.imgpath, image_ch)

    r = 225.0 / min(img.shape[0], img.shape[1])
    dim = (int(img.shape[1] * r), int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    x = (img.shape[1] - 224) // 2 if img.shape[1] > 224 else 0
    y = (img.shape[0] - 224) // 2 if img.shape[0] > 224 else 0
    img = img[y:y + 224, x:x + 224]

    img = img.reshape((1, image_w, image_h, (1 if image_ch == cv2.IMREAD_GRAYSCALE else 3)))

    # prepare dense network
    logging.info('prepare network')
    g1 = tf.Graph()
    with g1.as_default() as g:
        with tf.device('/cpu:0'):
            if conf['dataset'] == 'mnist':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 1])
                x_img = x_pre / 255
            elif conf['dataset'] == 'mnist224':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 1])
                x_img = x_pre / 255
            elif conf['dataset'] == 'ilsvrc2012':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 3])
                x_img = tf.subtract(x_pre, 128)

            # create network graph
            if conf['model'] == 'alexnet':
                model = alexnet_model(x_img, class_size=class_size, convtype=conf['conv'], model_conf=model_conf, keep_prob=1.0)
            else:
                raise Exception('invalid model: %s' % conf['model'])

            softmax = tf.nn.softmax(model)

    with tf.Session(config=config, graph=g1) as sess:
        logging.info('start to restore - dense')
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(args.path, 'model'))

        logging.info('start inference - dense')

        # warmup
        input, m, output = sess.run([x_img, model, softmax], feed_dict={
            x_pre: img
        }, options=run_options, run_metadata=run_metadata)

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        if args.save:
            with open('timeline_dense.json', 'w') as f:
                f.write(ctf)

        logging.info('network output = {}'.format(output))
        logging.info('predicted class = %d' % (np.argmax(output)))

        if conf['conv'] == 'lcnn':
            gr = tf.get_default_graph()
            tensors = [gr.get_tensor_by_name('layer%d/align_conv/kernel:0' % (convid + 1)) for convid in range(7)]
            kernel_vals = sess.run(tensors)
            logging.info('lcnn-densities: ' + ', '.join(['%.3f' % (np.count_nonzero(kernel_val) / kernel_val.size) for kernel_val in kernel_vals]))

        elapsed = 0
        for _ in range(args.benchmark):
            t = time.time()
            output = sess.run([model], feed_dict={
                x_pre: img
            })
            elapsed += time.time() - t
        logging.info('average elapsed time(dense) = %f' % (elapsed / args.benchmark))

        extract_dense_weights(sess)

    tf.reset_default_graph()

    if conf['conv'] == 'conv':
        sys.exit(0)

    g2 = tf.Graph()
    with g2.as_default() as g:
        with tf.device('/cpu:0'):
            if conf['dataset'] == 'mnist':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 1])
                x_img = x_pre / 255
            elif conf['dataset'] == 'mnist224':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 1])
                x_img = x_pre / 255
            elif conf['dataset'] == 'ilsvrc2012':
                x_pre = tf.placeholder(tf.float32, shape=[1, image_w, image_h, 3])
                x_img = tf.subtract(x_pre, 128)

            # create network graph
            if conf['model'] == 'alexnet':
                model = alexnet_model(x_img, class_size=class_size, convtype=conf['conv'], model_conf=model_conf,
                                      keep_prob=1.0)
            else:
                raise Exception('invalid model: %s' % conf['model'])

            softmax = tf.nn.softmax(model)

    with tf.Session(config=config, graph=g2) as sess:
        logging.info('start to restore - sparse')
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(args.path, 'model'))

        logging.info('start inference - sparse')

        # warmup
        output = sess.run([softmax], feed_dict={
            x_pre: img
        })
        output = sess.run([softmax], feed_dict={
            x_pre: img
        }, options=run_options, run_metadata=run_metadata)

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        if args.save:
            with open('timeline_sparse.json', 'w') as f:
                f.write(ctf)

        logging.info('network output = {}'.format(output))
        logging.info('predicted class = %d' % (np.argmax(output)))

        elapsed = 0
        for _ in range(args.benchmark):
            t = time.time()
            output = sess.run([model], feed_dict={
                x_pre: img
            })
            elapsed += time.time() - t
        logging.info('average elapsed time(sparse) = %f' % (elapsed / args.benchmark))
