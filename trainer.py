import os
import argparse
import json
import logging
import sys

import tensorflow as tf
import time
import yaml

from models.lenet import lenet_model
from models.alexnet import alexnet_model
import utils
from utils import DataMNIST, DataILSVRC2012, optimizers, logstep, LOG_DIR, DataMNIST224

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Training using LCNN.')
    parser.add_argument('--conf', default='./confs/lenet.yaml', help='configuration file path')
    parser.add_argument('--dataset', default='mnist', help='mnist, mnist224, ilsvrc2012')
    parser.add_argument('--conv', default='normal', help='lcnn, normal')
    parser.add_argument('--path-mnist', default='/data/public/ro/dataset/images/MNIST/_original/')
    parser.add_argument('--path-ilsvrc2012', default='/data/public/ro/dataset/images/imagenet/ILSVRC/2012/object_localization/ILSVRC/')
    parser.add_argument('--model-conf', default='lcnnbest', help='lcnnbest, lcnn0.9, normal')
    parser.add_argument('--logpath', default=LOG_DIR)

    # arguments for multinode
    parser.add_argument('--cluster', default=False, type=bool, help='True, if you train the model with multiple nodes')
    parser.add_argument('--cluster-conf', default='./confs/cluster_cloud_localps.yaml')
    parser.add_argument('--cluster-job', default='ps', help='ps, worker, local')
    parser.add_argument('--cluster-task', default=0, type=int)
    parser.add_argument('--gpu', default=1, type=int)

    args = parser.parse_args()

    # load config
    logging.info('config path : %s' % args.conf)
    with open(args.conf, 'r') as stream:
        conf = yaml.load(stream)
    model_conf = conf['model_conf'][args.model_conf]
    dataset = conf['datasets'][args.dataset]

    # re-calculate iterations using number of gpu towers
    dataset['iteration'] /= args.gpu
    dataset['lrstep'] = [x // args.gpu for x in dataset['lrstep']]

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

    # setting optimizer & learning rate
    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.name_scope('train'):
        optimizer_type = optimizers[dataset['optimizer']]
        if isinstance(dataset['learningrate'], float):
            learning_rate = dataset['learningrate']
        else:
            learning_rate = tf.train.piecewise_constant(global_step, dataset['lrstep'], dataset['learningrate'])
        train_step = optimizer_type(learning_rate)

    # dataset
    g = tf.get_default_graph()
    utils.default_graph = g
    if args.dataset == 'mnist':
        class_size = 10
        data_feeder = DataMNIST(args.path_mnist, dataset['batchsize'], args.gpu)
    elif args.dataset == 'mnist224':
        class_size = 10
        data_feeder = DataMNIST224(args.path_mnist, dataset['batchsize'], args.gpu)
    elif args.dataset == 'ilsvrc2012':
        class_size = 1000
        data_feeder = DataILSVRC2012(args.path_ilsvrc2012, dataset['batchsize'], args.gpu)
        data_feeder.mode = 'http'
    else:
        raise Exception('invalid dataset: %s' % args.dataset)

    # parse model configuration
    towers_grad = []
    towers_acc = []
    with tf.variable_scope(tf.get_variable_scope()):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout prob
        for gpu_id in range(args.gpu):
            logging.info('creating tower for gpu-%d' % (gpu_id + 1))
            with tf.device(('/gpu:%d' % gpu_id) if not args.cluster else tf.train.replica_device_setter(worker_device='/job:{job}/task:{id}/gpu:{gpu_id}'.format(job=args.cluster_job, id=args.cluster_task, gpu_id=gpu_id), cluster=cluster)):
                with tf.name_scope('TASK%d_TOWER%d' % (args.cluster_task, gpu_id)) as scope:
                    if args.dataset == 'mnist':
                        x_img, y_ = data_feeder.dequeue()
                        x = tf.divide(x_img, 255)
                    elif args.dataset == 'mnist224':
                        x_img, y_ = data_feeder.dequeue()
                        x = tf.divide(x_img, 255)
                    elif args.dataset == 'ilsvrc2012':
                        x_img, y_ = data_feeder.dequeue()
                        x = tf.subtract(x_img, 128)
                    else:
                        raise Exception('invalid dataset: %s' % args.dataset)

                    if conf['model'] == 'lenet':
                        model = lenet_model(x, class_size=class_size, conv=args.conv, model_conf=model_conf, keep_prob=keep_prob)
                    elif conf['model'] == 'alexnet':
                        model = alexnet_model(x, class_size=class_size, model_conf=model_conf, keep_prob=keep_prob)
                    else:
                        raise Exception('invalid model: %s' % conf['model'])

                    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=model))

                    grads = train_step.compute_gradients(cross_entropy)
                    towers_grad.append(grads)

                    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    towers_acc.append(accuracy)

                    tf.get_variable_scope().reuse_variables()

                    # train_step = train_step.minimize(cross_entropy, global_step=global_step)
        pass

    grads = average_gradients(towers_grad)
    accs = tf.reduce_mean(towers_acc)
    train_step = train_step.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(train_step, variables_averages_op)

    logging.info('---- app configuration ---')
    print(json.dumps(conf))

    # prepare session
    if not args.cluster:
        saver = tf.train.Saver()
    is_chief = (args.cluster_task == 0)
    hooks = [tf.train.StopAtStepHook(last_step=dataset['iteration'])]
    time_started = time.time()
    with tf.Session(config=config) if not args.cluster else \
            tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, checkpoint_dir=args.logpath, hooks=hooks, config=config) as sess:
        logging.info('initialization')
        tf.train.start_queue_runners(sess=sess)
        data_feeder.start_enqueue_thread(sess, 8)

        if not args.cluster:
            sess.run(tf.global_variables_initializer())
        else:
            logging.info('master: %s' % server.target)

        i = 0
        if args.cluster:
            def stop_condition(): return sess.should_stop()
        else:
            def stop_condition(): return i >= dataset['iteration']

        logging.info('learning start')
        last_gs_num1 = last_gs_num2 = 0
        while not stop_condition():
            _, train_loss, train_accuracy, gs_num, lr_val = sess.run(
                [train_op, cross_entropy, accuracy, global_step, learning_rate],
                feed_dict={keep_prob: dataset['dropkeep']}
            )

            if gs_num - last_gs_num1 >= logstep[args.dataset]['training']:
                # log of training loss / accuracy
                batch_per_sec = i / (time.time() - time_started)
                logging.info('step=%d(%d), %0.4f batchstep/sec lr=%f, loss=%g, accuracy=%.4g' % (gs_num, (i+1), batch_per_sec, lr_val, train_loss, train_accuracy))
                data_feeder.time_image = 0
                data_feeder.time_post_process = 0
                last_gs_num1 = gs_num

                # if args.conv != 'normal':
                #     conv1_kernel_val = gr.get_tensor_by_name('layer1/align_conv/kernel:0').eval(session=sess)
                #     conv2_kernel_val = gr.get_tensor_by_name('layer2/align_conv/kernel:0').eval(session=sess)
                #     logging.info('conv1-density %f, conv2-density %f' %
                #                  (np.count_nonzero(conv1_kernel_val) / conv1_kernel_val.size,
                #                   np.count_nonzero(conv2_kernel_val) / conv2_kernel_val.size))

            if is_chief and gs_num - last_gs_num2 >= logstep[args.dataset]['validation']:
                # validation without batch processing
                page = 0
                total_acc = 0
                total_cnt = 0
                while True:
                    # log of test accuracy
                    images_test, labels_test, num_data, more_batch = data_feeder.validation_set(page)
                    acc = accuracy.eval(feed_dict={x_img: images_test, y_: labels_test, keep_prob: 1.0}, session=sess)
                    total_acc += acc * len(labels_test)
                    total_cnt += num_data
                    page += 1
                    if not more_batch:
                        break
                logging.info('validation(%d) accuracy %g' % (total_cnt, total_acc / total_cnt))
                last_gs_num2 = gs_num

            if not args.cluster and gs_num > 0 and gs_num % 50000 == 0:
                saver.save(sess, os.path.join(args.logpath, 'curtis-model'), global_step=global_step)
            i += 1

        logging.info('optimization finished.')

    logging.info('app finished.')
