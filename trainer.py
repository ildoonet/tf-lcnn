import os
import argparse
import json
import logging
import sys

import numpy as np
import tensorflow as tf
import time
import yaml

from models.lenet import lenet_model
from models.alexnet import alexnet_model
from utils import DataMNIST, DataILSVRC2012, optimizers, logstep, LOG_DIR, DataMNIST224

config = tf.ConfigProto(allow_soft_placement = True)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Training using LCNN.')
    parser.add_argument('--conf', default='./confs/alexnet.yaml', help='configuration file path')
    parser.add_argument('--dataset', default='ilsvrc2012', help='mnist, mnist224, ilsvrc2012')
    parser.add_argument('--conv', default='normal', help='lcnn, normal')
    parser.add_argument('--path-mnist', default='/data/public/ro/dataset/images/MNIST/_original/')
    parser.add_argument('--path-ilsvrc2012', default='/data/public/ro/dataset/images/imagenet/ILSVRC/2012/object_localization/ILSVRC/')
    parser.add_argument('--model-conf', default='lcnnbest', help='lcnnbest, lcnn0.9, normal')
    parser.add_argument('--logpath', default=LOG_DIR)

    # arguments for multinode
    parser.add_argument('--cluster', default=False, type=bool, help='True, if you train the model with multiple nodes')
    parser.add_argument('--cluster-conf', default='./confs/cluster_cloud_local.yaml')
    parser.add_argument('--cluster-job', default='ps', help='ps, worker')
    parser.add_argument('--cluster-task', default=0, type=int)

    args = parser.parse_args()

    # load config
    with open(args.conf, 'r') as stream:
        conf = yaml.load(stream)
    model_conf = conf['model_conf'][args.model_conf]
    dataset = conf['datasets'][args.dataset]

    # load cluster
    if args.cluster:
        with open(args.cluster_conf, 'r') as stream:
            cluster_conf = yaml.load(stream)
        cluster = tf.train.ClusterSpec(cluster_conf)
        server = tf.train.Server(cluster, job_name=args.cluster_job, task_index=args.cluster_task)

        if args.cluster_job == 'ps':
            logging.info('parameter server %s %d' % (args.cluster_job, args.cluster_task))
            server.join()       # blocking call
            sys.exit(0)

        tfdevice = tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % args.cluster_task,
                                                  cluster=cluster)
        # change iteration configuration
        if 'worker' in cluster_conf.keys():
            divider = len(cluster_conf['worker'])
            conf['datasets'][args.dataset]['iteration'] //= divider
            conf['datasets'][args.dataset]['lrstep'] = [x // divider for x in conf['datasets'][args.dataset]['lrstep']]
    else:
        tfdevice = '/gpu:0'

    # parse dataset
    if args.dataset == 'mnist':
        class_size = 10
        y_ = tf.placeholder(tf.float32, shape=[None, class_size])
        x_img = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        x = tf.divide(x_img, 255)
    elif args.dataset == 'mnist224':
        class_size = 10
        y_ = tf.placeholder(tf.float32, shape=[None, class_size])
        x_img = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
        x = tf.divide(x_img, 255)
    elif args.dataset == 'ilsvrc2012':
        class_size = 1000
        y_ = tf.placeholder(tf.float32, shape=[None, class_size])
        x_img = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        x = tf.subtract(x_img, 128)
    else:
        raise Exception('invalid dataset: %s' % args.dataset)

    # initiate dataset feeder with augmenter
    if args.dataset == 'mnist':
        data_feeder = DataMNIST(args.path_mnist, dataset['batchsize'])
    elif args.dataset == 'mnist224':
        data_feeder = DataMNIST224(args.path_mnist, dataset['batchsize'])
    elif args.dataset == 'ilsvrc2012':
        data_feeder = DataILSVRC2012(args.path_ilsvrc2012, dataset['batchsize'])
    else:
        raise Exception()

    # parse model configuration
    with tf.device(tfdevice):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')    # dropout prob

        if conf['model'] == 'lenet':
            model = lenet_model(x, class_size=class_size, conv=args.conv, model_conf=model_conf, keep_prob=keep_prob)
        elif conf['model'] == 'alexnet':
            model = alexnet_model(x, class_size=class_size, model_conf=model_conf, keep_prob=keep_prob)
        else:
            raise Exception('invalid model: %s' % conf['model'])

        gr = tf.get_default_graph()
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=model))
        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # setting optimizer & learning rate
        optimizer_type = optimizers[dataset['optimizer']]

        if isinstance(dataset['learningrate'], float):
            train_step = optimizer_type(dataset['learningrate']).minimize(cross_entropy)
        else:
            global_step = tf.Variable(0, trainable=False, name='global_step')
            learning_rate = tf.train.piecewise_constant(global_step, dataset['lrstep'], dataset['learningrate'])
            train_step = optimizer_type(learning_rate).minimize(cross_entropy, global_step=global_step)

    logging.info('---- app configuration ---')
    print(json.dumps(conf))

    # prepare session
    if not args.cluster:
        saver = tf.train.Saver()
    is_chief = (args.cluster_task == 0)
    hooks = [tf.train.StopAtStepHook(last_step=dataset['iteration'])]
    with tf.Session(config=config) if not args.cluster else \
            tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, checkpoint_dir=args.logpath, hooks=hooks) as sess:
        logging.info('initialize variables')

        if not args.cluster:
            sess.run(tf.global_variables_initializer())

        logging.info('learning start')
        i = 0
        if args.cluster:
            def stop_condition(): return sess.should_stop()
        else:
            def stop_condition(): return i >= dataset['iteration']

        last_gs_num1 = last_gs_num2 = 0
        time_backprop = time_batch = 0
        while not stop_condition():
            t = time.time()
            batch = data_feeder.next_batch()
            time_batch += time.time() - t
            t = time.time()
            _, gs_num = sess.run([train_step, global_step], feed_dict={x_img: batch[0], y_: batch[1], keep_prob: dataset['dropkeep']})
            time_backprop += time.time() - t

            if gs_num - last_gs_num1 >= logstep[args.dataset]['training']:
                # log of training loss / accuracy
                train_loss, train_accuracy, lr_val, output = sess.run(
                    [cross_entropy, accuracy, learning_rate, model],
                    feed_dict={x_img: batch[0], y_: batch[1], keep_prob: 1.0}
                )
                train_loss, train_accuracy, lr_val = sess.run([cross_entropy, accuracy, learning_rate],
                                                                      feed_dict={x_img: batch[0], y_: batch[1],
                                                                                 keep_prob: 1.0})
                train_loss = cross_entropy.eval(feed_dict={x_img: batch[0], y_: batch[1], keep_prob: 1.0}, session=sess)
                logging.info('step=%d, lr=%f, loss=%g, accuracy=%.4g, elapsed_time(%.2f+%.2f batch:%.2f backprop:%.2f)' % (gs_num, lr_val, train_loss, train_accuracy, data_feeder.time_image, data_feeder.time_post_process, time_batch, time_backprop))
                data_feeder.time_image = 0
                data_feeder.time_post_process = 0
                time_backprop = time_batch = 0
                last_gs_num1 = gs_num

                if args.conv != 'normal':
                    conv1_kernel_val = gr.get_tensor_by_name('layer1/align_conv/kernel:0').eval(session=sess)
                    conv2_kernel_val = gr.get_tensor_by_name('layer2/align_conv/kernel:0').eval(session=sess)
                    logging.info('conv1-density %f, conv2-density %f' %
                                 (np.count_nonzero(conv1_kernel_val) / conv1_kernel_val.size,
                                  np.count_nonzero(conv2_kernel_val) / conv2_kernel_val.size))

            if is_chief and gs_num - last_gs_num2 >= logstep[args.dataset]['validation']:
                MAX_PAGE = 64
                page = 0
                total_acc = 0
                total_cnt = 0
                while True:
                    # log of test accuracy
                    images_test, labels_test, more_batch = data_feeder.validation_set(page)
                    total_acc += accuracy.eval(feed_dict={x_img: images_test, y_: labels_test, keep_prob: 1.0}, session=sess) * len(labels_test)
                    total_cnt += len(labels_test)
                    page += 1
                    if not more_batch or MAX_PAGE <= page:
                        break
                logging.info('validation(%d) accuracy %g' % (total_cnt, total_acc / total_cnt))
                last_gs_num2 = gs_num

            if not args.cluster and gs_num % 50000 == 0:
                saver.save(sess, os.path.join(args.logpath, 'curtis-model'), global_step=global_step)
            i += 1

        logging.info('optimization finished.')

    logging.info('app finished.')
