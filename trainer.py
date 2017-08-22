import argparse
import logging
import sys

import numpy as np
import tensorflow as tf
import yaml

from models.lenet import lenet_model
from models.alexnet import alexnet_model
from utils import DataMNIST
from yellowfin.yellowfin import YFOptimizer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Training using LCNN.')
    parser.add_argument('--conf', default='./confs/lenet.yaml', help='configuration file path')
    parser.add_argument('--dataset', default='mnist', help='mnist, ilsvrc2012')
    parser.add_argument('--conv', default='lcnn', help='lcnn, normal')
    parser.add_argument('--path-mnist', default='/data/public/ro/dataset/images/MNIST/_original/')
    parser.add_argument('--model-conf', default='lcnnbest', help='lcnnbest, lcnn0.9, normal')
    args = parser.parse_args()

    with open(args.conf, 'r') as stream:
        conf = yaml.load(stream)

    # parse dataset
    if args.dataset == 'mnist':
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        x = tf.placeholder(tf.float32, shape=[None, 784])
        if conf['model'] == 'alexnet':
            x_reshape = tf.reshape(x, (-1, 28, 28, 1))
    else:
        raise Exception('invalid dataset: %s' % args.dataset)
    dataset = None
    for _dataset in conf['datasets']:
        if _dataset['name'] == args.dataset:
            dataset = _dataset
        break

    # parse model configuration
    model_conf = conf['model_conf'][args.model_conf]
    if conf['model'] == 'lenet':
        model = lenet_model(x, conv=args.conv, model_conf=model_conf)
    elif conf['model'] == 'alexnet':
        model = alexnet_model(x_reshape, model_conf=model_conf)
    else:
        raise Exception('invalid model: %s' % conf['model'])

    gr = tf.get_default_graph()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=model))
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    if not dataset:
        raise Exception('not configured dataset: %s' % args.dataset)

    if dataset['optimizer'] == 'adagrad':
        train_step = tf.train.AdagradOptimizer(dataset['learningrate']).minimize(cross_entropy)
    elif dataset['optimizer'] == 'yellowfin':
        train_step = YFOptimizer(learning_rate=dataset['learningrate'], momentum=0.0).minimize(cross_entropy)
    else:
        raise Exception('invalid optimizer')

    logging.info('learning start')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if args.dataset == 'mnist':
            data_mnist = DataMNIST(args.path_mnist, dataset['batchsize'])
            images_test, labels_test = data_mnist.test_batch()

            for i in range(dataset['iteration']):
                batch = data_mnist.next_batch()
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

                if i % 1000 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                    logging.info('step %d, training accuracy %g' % (i, train_accuracy))
                    if args.conv != 'normal':
                        conv1_kernel_val = gr.get_tensor_by_name('layer1/align_conv/kernel:0').eval(session=sess)
                        conv2_kernel_val = gr.get_tensor_by_name('layer2/align_conv/kernel:0').eval(session=sess)
                        logging.info('conv1-density %f, conv2-density %f' %
                                     (np.count_nonzero(conv1_kernel_val) / conv1_kernel_val.size,
                                      np.count_nonzero(conv2_kernel_val) / conv2_kernel_val.size))

                if i % 5000 == 0:
                    logging.info('test accuracy %g' % accuracy.eval(feed_dict={x: images_test, y_: labels_test, keep_prob: 1.0}))
        else:
            raise

        logging.info('optimization finished. test accuracy %g' % accuracy.eval(feed_dict={x: images_test, y_: labels_test, keep_prob: 1.0}))
