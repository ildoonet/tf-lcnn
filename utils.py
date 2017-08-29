import time
import abc
import os
import random
import sys
import logging
import hmac
import hashlib
import requests
import base64
import threading
from datetime import datetime
from urllib.parse import urlparse
import concurrent.futures
from queue import Queue

import numpy as np
import multiprocessing
import cv2
import tensorflow as tf
from tensorflow.python.training import queue_runner
import imgaug as ia
from mnist import MNIST
from imgaug import augmenters as iaa

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('requests').setLevel(logging.WARNING)

LOG_DIR = '/data/private/tf-lcnn-logs'
sometimes_1 = lambda aug: iaa.Sometimes(0.1, aug)
sometimes_5 = lambda aug: iaa.Sometimes(0.5, aug)


def get_augseq(dataset):
    auglist = [
            sometimes_5([
            # apply the following augmenters to most images
            sometimes_5(iaa.Crop(percent=(0, 0.15))),  # crop images by 0-10% of their height/width
            iaa.Fliplr(0.0 if dataset == 'mnist' else 0.5),
            sometimes_5(iaa.GaussianBlur(sigma=(0, 3.0))),    # blur images with a sigma of 0 to 3.0
            sometimes_1(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # scale images to 90-110% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10),  # rotate by -10 to +10 degrees
                shear=(-4, 4),  # shear by -4 to +4 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 1),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            sometimes_1(iaa.SomeOf((0, 3), [
                iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                # iaa.OneOf([
                #     iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                #     iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                # ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.2)),  # sharpen images
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
                sometimes_5(iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.2)),
                # move pixels locally around (with random strengths)
                sometimes_5(iaa.PiecewiseAffine(scale=(0.01, 0.03))),  # sometimes move parts of the image around
            ], random_order=True))
            ])
        ]

    augseq = iaa.Sequential(auglist, random_order=True)
    return augseq

optimizers = {
    'adagrad': tf.train.AdagradOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'sgd': tf.train.GradientDescentOptimizer
}

logstep = {
    'mnist': {
        'validation': 1000,
        'training': 500,
    },
    'mnist224': {
        'validation': 2000,
        'training': 1000,
    },
    'ilsvrc2012': {
        'validation': 2000,
        'training': 100,
    }
}

default_graph = None


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


def get_activation_f(name):
    if name == 'tanh':
        return tf.nn.tanh
    elif name == 'relu':
        return tf.nn.relu
    else:
        raise


class DataReader(tf.FIFOQueue):
    def __init__(self, batchsize, gpusize, queue_shapes):
        self.gpusize = gpusize
        self.batchsize = batchsize
        self.queue_shapes = queue_shapes
        self.batch_loader = ia.BatchLoader(self.load_batches)
        self.bg_augmenter = ia.BackgroundAugmenter(batch_loader=self.batch_loader, augseq=self.get_augseq(),
                                                   nb_workers=multiprocessing.cpu_count())
        self.time_image = 0
        self.time_post_process = 0

        super().__init__(capacity=gpusize * 10, dtypes=(tf.uint8, tf.uint8), shapes=queue_shapes)

        self.use_enqueue_thread = False

    def start_enqueue_thread(self, sess, worker_size=2):
        if not self.use_enqueue_thread:
            self.use_enqueue_thread = True
            global default_graph
            with default_graph.as_default():
                self.y_ = tf.placeholder(tf.uint8, shape=self.queue_shapes[1])
                self.x_img = tf.placeholder(tf.uint8, shape=self.queue_shapes[0])
                self.enqueue_op = self.enqueue((self.x_img, self.y_))

        for _ in range(worker_size):
            thread = threading.Thread(target=self._batch_enqueue_job, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()

    def _batch_enqueue_job(self, sess):
        while True:
            batch = self.next_batch()
            sess.run(self.enqueue_op, feed_dict={self.x_img: batch[0], self.y_: batch[1]})
            # while self.size().eval(session=sess) >= self.gpusize * 10:
            time.sleep(0.3)

    @abc.abstractmethod
    def get_augseq(self):
        pass

    @abc.abstractmethod
    def next_image(self):
        # yield np array of image and label
        # yield img, lb_onehot
        pass

    @abc.abstractmethod
    def process_batch(self, images, labels):
        pass

    @abc.abstractmethod
    def validation_set(self, page):
        pass

    def load_batches(self):
        gen = self.next_image()
        while True:
            batch_images = []
            batch_data = []
            t = time.time()
            for _ in range(self.batchsize):
                img, lb = next(gen)
                batch_images.append(img)
                batch_data.append(lb)
            self.time_image += time.time() - t

            try:
                yield ia.Batch(images=np.array(batch_images), data=np.array(batch_data, dtype=np.uint8))
            except Exception as e:
                pass

    def next_batch(self):
        batch = self.bg_augmenter.get_batch()

        t = time.time()
        augmented_batch = self.process_batch(batch.images_aug, batch.data)
        self.time_post_process += time.time() - t

        return augmented_batch


class DataMNIST(DataReader):
    def __init__(self, path, batchsize, gpusize, imgsize=28):
        self.mndata = MNIST(path=path)
        self.images_test, self.labels_test = self.mndata.load_testing()
        self.images_test = np.asarray(self.images_test, dtype=np.uint8)
        self.labels_test = np.asarray(self.labels_test, dtype=np.uint8)
        super().__init__(batchsize, gpusize, queue_shapes=((batchsize, imgsize, imgsize, 1), (batchsize, 10)))

    def get_augseq(self):
        return get_augseq('mnist')

    def next_image(self):
        images, labels = self.mndata.load_training()
        images = np.asarray(images, dtype=np.uint8)
        labels = np.asarray(labels, dtype=np.uint8)
        _cnt = 0
        while True:
            if _cnt == 0:
                # shuffle
                p = np.random.permutation(len(images))
                images = images[p]
                labels = labels[p]

            # label to one hot
            img = images[_cnt].reshape(28, 28, 1)
            lb_onehot = np.zeros(10, dtype=np.uint8)
            lb_onehot[labels[_cnt]] = 1

            yield img, lb_onehot
            _cnt += 1
            _cnt %= len(images)

    def process_batch(self, images, labels):
        return images.reshape((self.batchsize, 28, 28, 1)), labels

    def validation_set(self, page):
        start_idx, end_idx = page * self.batchsize, min((page + 1) * self.batchsize, len(self.images_test))

        images_test = self.images_test[start_idx:end_idx]
        resizes = []
        for image in images_test:
            resizes.append(image)
        for _ in range(end_idx - start_idx, self.batchsize):
            resizes.append(np.zeros(784, dtype=np.uint8))

        labels_test = self.labels_test[start_idx:end_idx]
        labels_test_onehot = np.zeros((self.batchsize, 10), dtype=np.uint8)
        labels_test_onehot[np.arange(len(labels_test)), labels_test] = 1
        return np.array(resizes).reshape(len(resizes), 28, 28, 1), labels_test_onehot, end_idx - start_idx, (page + 1) * self.batchsize < len(self.images_test)


class DataMNIST224(DataMNIST):
    def __init__(self, path, batchsize, gpusize):
        super().__init__(path, batchsize, gpusize, imgsize=224)

    def process_batch(self, images, labels):
        images, labels = super(DataMNIST224, self).process_batch(images, labels)
        resizes = []
        for image in images:
            resizes.append(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA))
        return np.array(resizes).reshape(len(resizes), 224, 224, 1), labels

    def validation_set(self, page):
        images, labels, num_data, do_more = super(DataMNIST224, self).validation_set(page)

        resizes = []
        for image in images:
            resizes.append(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA))

        return np.array(resizes).reshape(len(resizes), 224, 224, 1), labels, num_data, do_more


class DataILSVRC2012(DataReader):
    def __init__(self, path, batchsize, gpusize):
        self.path = path
        self.http_prefix = 'http://'
        self.mode = 'local'

        # read class list
        with open(os.path.join(path, 'imagenet_lsvrc_synsets.txt'), 'r') as f:
            self.cls_list = f.read().splitlines()

        # read image list - training
        with open(os.path.join(path, 'ImageSets/CLS-LOC/train_cls.txt'), 'r') as f:
            self.train_list = np.array([x.split(' ')[0] + '.JPEG' for x in f.read().splitlines()])

        # read image list - validation
        with open(os.path.join(path, 'ImageSets/CLS-LOC/val.txt'), 'r') as f:
            self.valid_list = np.array([x.split(' ')[0] + '.JPEG' for x in f.read().splitlines()])

        with open(os.path.join(path, 'imagenet_validation_synsets.txt'), 'r') as f:
            self.vallb_list = np.array(f.read().splitlines())

        self.image_q = Queue()

        super().__init__(batchsize, gpusize, queue_shapes=((batchsize, 224, 224, 3), (batchsize, 10)))

    def get_augseq(self):
        return get_augseq('ilsvrc2012')

    def _read_images_in_q(self, filepaths=None):
        if filepaths is None:
            filepaths = self.train_list[:self.batchsize]

        def read_image_to_q(filepath):
            img, label = self.read_image(filepath, validation=False)
            self.image_q.put((img, label))
            # return img, label

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            future_to_img = {executor.submit(read_image_to_q, path): path for path in filepaths}
            # for future in concurrent.futures.as_completed(future_to_img):
                # self.image_q.put(future.result())

    def read_image(self, filepath=None, validation=False):
        if filepath:
            if self.mode == 'http':
                host = 'http://twg.kakaocdn.net'
                service_id = 'braincloud'
                read_key = b'r_17b3da3e4f39ef69e08a54ff45c2f6'
                tenth_url = host + '/' + service_id + '/imagenet/ILSVRC/2012/object_localization/ILSVRC/Data/CLS-LOC/' \
                            + ('val/' if validation else 'train/') + filepath

                datetime_str = datetime.strftime(datetime.utcnow(), '%a, %d %b %Y %H:%M:%S GMT')
                query = '\n'.join(['GET', '', '', datetime_str, '', urlparse(tenth_url).path]).encode('utf-8')
                hashv = hmac.new(read_key, query, hashlib.sha1).digest()
                signature = base64.encodebytes(hashv).rstrip()
                headers = {
                    'Authorization': 'TWG %s:%s' % (service_id, signature.decode('utf-8')),
                    'Date': datetime_str,
                }
                resp = requests.get(tenth_url, headers=headers)

                img = cv2.imdecode(np.fromstring(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(os.path.join(self.path, 'Data/CLS-LOC', ('train' if not validation else 'val'), filepath))
            if not validation:
                try:
                    label = self.cls_list.index(filepath.split('/')[0])
                except Exception as e:
                    logging.error('class index error: %s @ %s' % (filepath.split('/')[0], str(e)))
                    sys.exit(-1)
            else:
                label = 0       # dummy data
        else:
            img, label = self.image_q.get()
            return img, label

        # resize based on shorter axis to 256px
        r = 256.0 / min(img.shape[0], img.shape[1])
        dim = (int(img.shape[1] * r), int(img.shape[0] * r))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # random crop - 224x224
        x = random.randint(0, img.shape[1] - 224) if img.shape[1] > 224 else 0
        y = random.randint(0, img.shape[0] - 224) if img.shape[0] > 224 else 0
        img = img[y:y + 224, x:x + 224]

        return img, label

    def next_image(self):
        _cnt = 0
        while True:
            if _cnt == 0:
                # shuffle
                p = np.random.permutation(len(self.train_list))
                self.train_list = self.train_list[p]

            if self.image_q.empty():
                self._read_images_in_q(self.train_list[_cnt:_cnt+self.batchsize])

            # read image & random crop 224x224 inside the image
            img, label = self.read_image()

            lb_onehot = np.zeros(len(self.cls_list), dtype=np.uint8)
            lb_onehot[label] = 1

            yield img, lb_onehot
            _cnt += 1
            _cnt %= len(self.train_list)

    def process_batch(self, images, labels):
        return images, labels

    def validation_set(self, page):
        imgs = []
        labels = []
        start_idx = page * self.batchsize
        end_idx = min((page + 1) * self.batchsize, len(self.valid_list))
        for ridx in range(start_idx, end_idx):
            path = self.valid_list[ridx]
            try:
                img, _ = self.read_image(path, validation=True)
                label = self.cls_list.index(self.vallb_list[ridx])
            except Exception as e:
                logging.warning('validation set error: %s %s' % (path, str(e)))
                continue
            lb_onehot = np.zeros(len(self.cls_list), dtype=np.uint8)
            lb_onehot[label] = 1

            imgs.append(img)
            labels.append(lb_onehot)

        for _ in range(end_idx - start_idx, self.batchsize):
            imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
            labels.append(np.zeros(len(self.cls_list), dtype=np.uint8))

        return imgs, labels, end_idx - start_idx, (page + 1) * self.batchsize < len(self.valid_list)


if __name__ == '__main__':
    data_feeder = DataMNIST('/data/public/ro/dataset/images/MNIST/_original/', batchsize=64, gpusize=1)
    # data_feeder = DataILSVRC2012('/data/public/ro/dataset/images/imagenet/ILSVRC/2012/object_localization/ILSVRC/',
    #                              batchsize=32, gpusize=1)
    data_feeder.mode = 'http'

    # validation test
    print('validation test')
    val_cnt = 0
    page = 0
    while True:
        images_test, labels_test, num_data, more_batch = data_feeder.validation_set(page)
        val_cnt += num_data
        assert len(images_test) == data_feeder.batchsize
        assert len(labels_test) == data_feeder.batchsize
        assert num_data <= data_feeder.batchsize
        assert num_data > 0
        page += 1

        if not more_batch:
            break

    print('validation count=%d' % val_cnt)

    t = time.time()
    for _ in range(100):
        a = time.time()
        batch = data_feeder.next_batch()
        print('total elapsed %.4f, getting img: %.4f, post process: %.4f' % (time.time() - a, data_feeder.time_image, data_feeder.time_post_process))
        data_feeder.time_image = data_feeder.time_post_process = 0

    print()
    print(time.time() - t)
