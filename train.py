import os
import argparse
import json
import logging
import sys

import numpy as np
import tensorflow as tf
import time
import yaml

from data_feeder import get_ilsvrc_data_alexnet, get_mnist_data, DataFlowToQueue
from networks.alexnet import alexnet_model
from utils import optimizers, logstep, LOG_DIR, average_gradients, get_dataset_sizes

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Training using LCNN.')
    parser.add_argument('--conf', default='./confs/alexnet.yaml', help='configuration file path')
    parser.add_argument('--model-conf', default='lcnntest', help='lcnnbest, lcnn0.9, normal')
    parser.add_argument('--dataset', default='mnist224', help='mnist, mnist224, ilsvrc2012')
    parser.add_argument('--conv', default='lcnn', help='lcnn, conv')
    parser.add_argument('--path-ilsvrc2012', default='/data/public/ro/dataset/images/imagenet/ILSVRC/2012/object_localization/ILSVRC/')
    parser.add_argument('--logpath', default=LOG_DIR)
    parser.add_argument('--restore', type=str, default='')

    # arguments for multinode / multigpu
    parser.add_argument('--cluster', default=False, type=bool, help='True, if you train the model with multiple nodes')
    parser.add_argument('--cluster-conf', default='./confs/cluster_cloud_localps.yaml')
    parser.add_argument('--cluster-job', default='ps', help='ps, worker, local')
    parser.add_argument('--cluster-task', default=0, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--gpubatch', default='more', help='more, split')
    parser.add_argument('--warmup-epoch', default=10, type=int)

    args = parser.parse_args()

    # load config
    logging.info('config path : %s' % args.conf)
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

        tfdevice = tf.train.replica_device_setter(worker_device='/job:{job}/task:{id}'.format(job=args.cluster_job, id=args.cluster_task),
                                                  cluster=cluster)
    else:
        tfdevice = '/gpu:0'

    # dataset
    class_size, dataset_size = get_dataset_sizes(args.dataset)

    # re-calculate iterations using number of gpu towers
    epochstep = dataset_size / dataset['batchsize']
    dataset['iteration'] = epochstep * dataset['epoch']
    dataset['lrstep'] = [int(x * epochstep) for x in dataset['lrepoch']]
    if args.gpubatch == 'more':
        dataset['iteration'] /= args.gpu
        dataset['lrstep'] = [x // args.gpu for x in dataset['lrstep']]
        logstep[args.dataset]['training'] = logstep[args.dataset]['training'] // args.gpu
        logstep[args.dataset]['validation'] = logstep[args.dataset]['validation'] // args.gpu
        batch_per_tower = dataset['batchsize']
        dataset['learningrate'] = [x * args.gpu for x in dataset['learningrate']]
    elif args.gpubatch == 'split':
        dataset['batchsize'] //= args.gpu

    if args.dataset == 'mnist':
        dataset_val = get_mnist_data('test', 24, batchsize=dataset['batchsize'])
    elif args.dataset == 'mnist224':
        dataset_val = get_mnist_data('test', 224, batchsize=dataset['batchsize'])
    elif args.dataset == 'ilsvrc2012':
        dataset_val = get_ilsvrc_data_alexnet('test', 224, batchsize=dataset['batchsize'], directory=args.path_ilsvrc2012)
    else:
        raise Exception('invalid dataset=%s' % args.dataset)

    # setting optimizer & learning rate
    lookup_sparse = tf.placeholder(tf.int32, name='lookup_sparse')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.name_scope('train'):
        optimizer_type = optimizers[dataset['optimizer']]
        if isinstance(dataset['learningrate'], float):
            learning_rate = dataset['learningrate']
        else:
            learning_rate = tf.train.piecewise_constant(global_step, dataset['lrstep'], dataset['learningrate'])

        # gradual warm-up
        if args.gpubatch == 'more':
            warmup_iter = dataset_size * args.warmup_epoch / float(dataset['batchsize'] * args.gpu)
            warmup_ratio = tf.minimum((1.0 - 1.0 / args.gpu) * (tf.cast(global_step, tf.float32) / tf.constant(warmup_iter)) ** 2 + tf.constant(1.0 / args.gpu), tf.constant(1.0))
            learning_rate = warmup_ratio * learning_rate

        train_step = optimizer_type(learning_rate)

    # parse model configuration
    towers_inp = []
    towers_th = []
    towers_grad = []
    towers_acc = []
    towers_acc5 = []
    towers_loss = []
    with tf.variable_scope(tf.get_variable_scope()):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout prob
        for gpu_id in range(args.gpu):
            logging.info('creating tower for gpu-%d' % (gpu_id + 1))
            with tf.device(('/gpu:%d' % gpu_id) if not args.cluster else tf.train.replica_device_setter(worker_device='/job:{job}/task:{id}/gpu:{gpu_id}'.format(job=args.cluster_job, id=args.cluster_task, gpu_id=gpu_id), cluster=cluster)):
                with tf.name_scope('TASK%d_TOWER%d' % (args.cluster_task, gpu_id)) as scope:

                    with tf.device('/cpu:0'):
                        if args.dataset == 'mnist':
                            dataset_train = get_mnist_data('train', 24, batchsize=dataset['batchsize'])

                            x_img = tf.placeholder(tf.float32, shape=[dataset['batchsize'], 24, 24, 1])
                            y_ = tf.placeholder(tf.int64, shape=[dataset['batchsize']])
                            inp_th = DataFlowToQueue(dataset_train, [x_img, y_])
                            x_pre, y = inp_th.dequeue()
                            x = x_pre
                        elif args.dataset == 'mnist224':
                            dataset_train = get_mnist_data('train', 224, batchsize=dataset['batchsize'])

                            x_img = tf.placeholder(tf.float32, shape=[dataset['batchsize'], 224, 224, 1])
                            y_ = tf.placeholder(tf.int64, shape=[dataset['batchsize']])
                            inp_th = DataFlowToQueue(dataset_train, [x_img, y_])
                            x_pre, y = inp_th.dequeue()
                            x = x_pre
                        elif args.dataset == 'ilsvrc2012':
                            dataset_train = get_ilsvrc_data_alexnet('train', 224, batchsize=dataset['batchsize'], directory=args.path_ilsvrc2012)

                            x_img = tf.placeholder(tf.uint8, shape=[dataset['batchsize'], 224, 224, 3])
                            y_ = tf.placeholder(tf.int64, shape=[dataset['batchsize']])
                            inp_th = DataFlowToQueue(dataset_train, [x_img, y_])
                            x_pre, y = inp_th.dequeue()
                            x_pre = tf.cast(x_pre, tf.float32)
                            x = tf.subtract(x_pre, 128)
                        else:
                            raise Exception('invalid dataset: %s' % args.dataset)

                    towers_th.append(inp_th)
                    towers_inp.append((x_img, y_))

                    if conf['model'] == 'alexnet':
                        model = alexnet_model(x, class_size=class_size, convtype=args.conv, model_conf=model_conf, keep_prob=keep_prob)
                    else:
                        raise Exception('invalid model: %s' % conf['model'])

                    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=model))
                    loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope))
                    loss = cross_entropy + loss_reg
                    towers_loss.append(loss)

                    grads = train_step.compute_gradients(loss)
                    towers_grad.append(grads)

                    correct_prediction = tf.equal(tf.argmax(model, 1), y)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    towers_acc.append(accuracy)

                    correct_prediction5 = tf.nn.in_top_k(model, y, k=5)
                    accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))
                    towers_acc5.append(accuracy5)

                    tf.get_variable_scope().reuse_variables()
        pass

    # aggregate all gradients
    grads = average_gradients(towers_grad)
    acc1 = tf.reduce_mean(towers_acc)
    acc5 = tf.reduce_mean(towers_acc5)
    train_step = train_step.apply_gradients(grads, global_step=global_step)
    lss = tf.reduce_mean(towers_loss)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", lss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy-top1", acc1)
    tf.summary.scalar("accuracy-top5", acc5)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(train_step, variables_averages_op)

    logging.info('---- app configuration ---')
    print(json.dumps(conf))
    with open(os.path.join(args.logpath, 'train_conf.json'), 'w') as f:
        f.write(json.dumps(conf, indent=4))
    with open(os.path.join(args.logpath, 'conf.json'), 'w') as f:
        f.write(json.dumps({
            'model': conf['model'],
            'dataset': args.dataset,
            'conv': args.conv,
            'initial_sparsity': conf['model_conf'].get('initial_sparsity', []),
            'dictionary': conf['model_conf'].get('dictionary', [])
        }, indent=4))

    # prepare session
    saver = None
    if not args.cluster:
        saver = tf.train.Saver()
    is_chief = (args.cluster_task == 0)
    hooks = [tf.train.StopAtStepHook(last_step=dataset['iteration'])]

    with tf.Session(config=config) if not args.cluster else \
            tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, checkpoint_dir=args.logpath, hooks=hooks, config=config) as sess:
        logging.info('initialization')

        if not args.cluster:
            sess.run(tf.global_variables_initializer())
        else:
            logging.info('master: %s' % server.target)

        if saver and args.restore:
            saver.restore(sess, os.path.join(args.logpath, args.restore))

        # tensorboard
        file_writer = tf.summary.FileWriter('/date/private/tensorboard/', sess.graph)

        # enqueue thread
        coord = tf.train.Coordinator()
        for th in towers_th:
            th.set_coordinator(coord)
            th.start()

        i = 0
        if args.cluster:
            def stop_condition(): return sess.should_stop()
        else:
            def stop_condition(): return i >= dataset['iteration']

        logging.info('learning start')
        time_started = time.time()
        last_gs_num1 = last_gs_num2 = 0
        while not stop_condition():
            _, gs_num = sess.run([train_op, global_step], feed_dict={keep_prob: dataset['dropkeep']})

            if gs_num - last_gs_num1 >= logstep[args.dataset]['training']:
                train_loss, train_acc1, train_acc5, lr_val, summary = sess.run(
                    [lss, acc1, acc5, learning_rate, merged_summary_op],
                    feed_dict={keep_prob: dataset['dropkeep']}
                )

                # log of training loss / accuracy
                batch_per_sec = (args.gpu if args.gpubatch == 'more' else 1) * i / (time.time() - time_started)
                logging.info('epoch=%.2f step=%d(%d), %0.4f batchstep/sec lr=%f, loss=%g, accuracy(top1)=%.4g, accuracy(top5)=%.4g' % (gs_num / epochstep, gs_num, (i+1), batch_per_sec, lr_val, train_loss, train_acc1, train_acc5))
                last_gs_num1 = gs_num

                file_writer.add_summary(summary, gs_num)

            should_save = (gs_num - last_gs_num2 >= logstep[args.dataset]['validation'] or dataset['iteration'] - gs_num <= 1)

            if is_chief and should_save:
                # validation without batch processing
                MAXPAGE = 200
                if dataset['iteration'] - gs_num <= 1:
                    MAXPAGE = 100000
                total_acc1 = total_acc5 = 0
                total_cnt = 0
                dataset_val.reset_state()
                gen_val = dataset_val.get_data()
                for page in range(MAXPAGE):
                    # log of test accuracy
                    try:
                        images_test, ls = next(gen_val)
                    except StopIteration:
                        break

                    acc1_test, acc5_test = sess.run([accuracy, accuracy5], feed_dict={x_pre: images_test, y: ls, keep_prob: 1.0})
                    total_acc1 += acc1_test * len(ls)
                    total_acc5 += acc5_test * len(ls)
                    total_cnt += len(images_test)

                logging.info('validation(%d) accuracy(top1) %g accuracy(top5) %g' % (total_cnt, total_acc1 / total_cnt, total_acc5 / total_cnt))
                last_gs_num2 = gs_num

                if saver and args.logpath and not args.cluster:
                    saver.save(sess, os.path.join(args.logpath, 'model'), global_step=global_step)

                if args.conv == 'lcnn':
                    # print sparsity
                    gr = tf.get_default_graph()
                    tensors = [gr.get_tensor_by_name('TASK0_TOWER0/layer%d/align_conv/kernel:0' % (convid+1)) for convid in range(7)]
                    kernel_vals = sess.run(tensors)
                    logging.info('lcnn-densities: ' + ', '.join(['%f' % (np.count_nonzero(kernel_val) / kernel_val.size) for kernel_val in kernel_vals]))

            i += 1

        logging.info('optimization finished.')

    logging.info('app finished. %f' % (time.time() - time_started))
