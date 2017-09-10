import argparse
import tensorflow as tf


if __name__ == '__main__':
    """
    Remove redundant extra gpu weights, redundant ops, and etcs
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='./models/alexnet/mnist/lcnn-accurate/model', help='model path')
    args = parser.parse_args()

    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(args.model + '.meta', clear_devices=True)
        loader.restore(sess, args.model)

        vvv = [x for x in tf.global_variables() if 'Adagrad' not in x.name and 'MovingAverage' not in x.name]
        saver = tf.train.Saver(vvv)
        saver.save(sess, args.model)
