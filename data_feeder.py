import logging
import multiprocessing
import sys
import time
import threading
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

import requests

import cv2

from tensorpack import imgaug
from tensorpack.dataflow import dataset
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.image import AugmentImageComponent
from tensorpack.dataflow.prefetch import PrefetchData
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated
from tensorpack.dataflow.dataset.ilsvrc import ILSVRC12

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('requests').setLevel(logging.WARNING)


def get_mnist_data(is_train, image_size, batchsize):
    ds = MNISTCh('train' if is_train else 'test', shuffle=True)

    if is_train:
        augs = [
            imgaug.RandomApplyAug(imgaug.RandomResize((0.8, 1.2), (0.8, 1.2)), 0.3),
            imgaug.RandomApplyAug(imgaug.RotationAndCropValid(15), 0.5),
            imgaug.RandomApplyAug(imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01), 0.25),
            imgaug.Resize((224, 224), cv2.INTER_AREA)
        ]
        ds = AugmentImageComponent(ds, augs)
        ds = PrefetchData(ds, 128*10, multiprocessing.cpu_count())
        ds = BatchData(ds, batchsize)
        ds = PrefetchData(ds, 256, 4)
    else:
        # no augmentation, only resizing
        augs = [
            imgaug.Resize((image_size, image_size), cv2.INTER_CUBIC),
        ]
        ds = AugmentImageComponent(ds, augs)
        ds = BatchData(ds, batchsize)
        ds = PrefetchData(ds, 20, 2)
    return ds


def get_ilsvrc_data_alexnet(is_train, image_size, batchsize, directory):
    if is_train:
        if not directory.startswith('/'):
            ds = ILSVRCTTenthTrain(directory)
        else:
            ds = ILSVRC12(directory, 'train')
        augs = [
            imgaug.RandomApplyAug(imgaug.RandomResize((0.9, 1.2), (0.9, 1.2)), 0.7),
            imgaug.RandomApplyAug(imgaug.RotationAndCropValid(15), 0.7),
            imgaug.RandomApplyAug(imgaug.RandomChooseAug([
                imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01),
                imgaug.RandomOrderAug([
                    imgaug.BrightnessScale((0.8, 1.2), clip=False),
                    imgaug.Contrast((0.8, 1.2), clip=False),
                    # imgaug.Saturation(0.4, rgb=True),
                ]),
            ]), 0.7),
            imgaug.Flip(horiz=True),

            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.RandomCrop((224, 224)),
        ]
        ds = AugmentImageComponent(ds, augs)
        ds = PrefetchData(ds, 1000, multiprocessing.cpu_count())
        ds = BatchData(ds, batchsize)
        ds = PrefetchData(ds, 10, 4)
    else:
        if not directory.startswith('/'):
            ds = ILSVRCTenthValid(directory)
        else:
            ds = ILSVRC12(directory, 'val')
        ds = AugmentImageComponent(ds, [
            imgaug.ResizeShortestEdge(224, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ])
        ds = PrefetchData(ds, 100, multiprocessing.cpu_count())
        ds = BatchData(ds, batchsize)

    return ds


class MNISTCh(dataset.Mnist):
    def __init__(self, is_train, shuffle):
        super().__init__(is_train, shuffle)

    def get_data(self):
        gen = super().get_data()
        try:
            while True:
                img, lb = next(gen)
                yield [img.reshape((28, 28, 1)), lb]
        except StopIteration as e:
            pass
        except Exception as e:
            logging.error(str(e))


class ILSVRCTenth(RNGDataFlow):
    def __init__(self, service_code):
        ILSVRCTenth.service_code = service_code
        self.cls_list = [x.decode('utf-8') for x in ILSVRCTenth._read_tenth('imagenet_lsvrc_synsets.txt').splitlines()]
        self.shuffle = True
        self.preload = 32 * 1

    @staticmethod
    def _tenthpath(pathurl):
        tenth_prefix = 'http://twg.kakaocdn.net/%s/imagenet/ILSVRC/2012/object_localization/ILSVRC/' % ILSVRCTenth.service_code
        url = tenth_prefix + pathurl
        return url

    @staticmethod
    def _read_tenth_batch(pathurls):
        import grequests
        urls = [grequests.get(ILSVRCTenth._tenthpath(pathurl)) for pathurl in pathurls]
        resps = grequests.map(urls)
        result_dict = {}
        for url, resp in zip(pathurls, resps):
            if not resp or resp.status_code // 100 != 2:
                continue
            result_dict[url] = resp.content
        return result_dict

    @staticmethod
    def _read_tenth(pathurl):
        url = ILSVRCTenth._tenthpath(pathurl)
        for _ in range(5):
            try:
                resp = requests.get(url)
                if resp.status_code // 100 != 2:
                    logging.warning('request failed code=%d url=%s' % (resp.status_code, url))
                    time.sleep(0.05)
                    continue
                return resp.content
            except Exception as e:
                logging.warning('request failed err=%s' % (str(e)))

        return ''

    def size(self):
        return len(self.train_list)

    def get_data(self):
        idxs = np.arange(len(self.train_list))
        if self.shuffle:
            self.rng.shuffle(idxs)

        caches = {}
        for i, k in enumerate(idxs):
            path = self.train_list[k]
            label = self.lb_list[k]

            if i % self.preload == 0:
                try:
                    caches = ILSVRCTenth._read_tenth_batch(self.train_list[idxs[i:i+self.preload]])
                except Exception as e:
                    logging.warning('tenth local cache failed, err=%s' % str(e))

            content = caches.get(path, '')
            if not content:
                content = ILSVRCTenth._read_tenth(path)

            img = cv2.imdecode(np.fromstring(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield [img, label]


class ILSVRCTTenthTrain(ILSVRCTenth):
    def __init__(self, service_code):
        super().__init__(service_code)

        # read image list - training
        self.train_list = ILSVRCTenth._read_tenth('ImageSets/CLS-LOC/train_cls.txt').splitlines()
        self.train_list = np.asarray(['Data/CLS-LOC/train/' + x.decode('utf-8').split(' ')[0] + '.JPEG' for x in self.train_list])

        self.lb_list = [self.cls_list.index(x.split('/')[3]) for x in self.train_list]

        self.shuffle = True


class ILSVRCTenthValid(ILSVRCTenth):
    def __init__(self, service_code):
        super().__init__(service_code)

        # read image list - validation
        self.train_list = ILSVRCTenth._read_tenth('ImageSets/CLS-LOC/val.txt').splitlines()
        self.train_list = np.asarray(['Data/CLS-LOC/val/' + x.decode('utf-8').split(' ')[0] + '.JPEG' for x in self.valid_list])

        synset_list = ILSVRCTenth._read_tenth('imagenet_validation_synsets.txt').splitlines()
        self.lb_list = [self.cls_list.index(x) for x in synset_list]

        self.shuffle = False


class DataFlowToQueue(threading.Thread):
    def __init__(self, ds, placeholders, queue_size=100):
        super().__init__()
        self.daemon = True

        self.ds = ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(queue_size, [ph.dtype for ph in placeholders], shapes=[ph.get_shape() for ph in placeholders])
        self.op = self.queue.enqueue(placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._coord = None
        self._sess = None

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logging.warning("DataFlowToQueue {} wasn't under a default session!".format(self.name))
            yield

    def start(self):
        self._sess = tf.get_default_session()
        super().start()

    def set_coordinator(self, coord):
        self._coord = coord

    def run(self):
        with self.default_sess():
            try:
                while not self._coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while True:
                            for dp in self.ds.get_data():
                                feed = dict(zip(self.placeholders, dp))
                                self.op.run(feed_dict=feed)
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        pass
                    except Exception as e:
                        if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                            pass
                        else:
                            logging.exception("Exception in {}:{}".format(self.name, str(e)))
            except Exception as e:
                logging.exception("Exception in {}:{}".format(self.name, str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logging.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()


if __name__ == '__main__':
    df = get_mnist_data(is_train=True, image_size=224, batchsize=128)
    # df = get_ilsvrc_data_alexnet(is_train=True, image_size=224, batchsize=32)
    df.reset_state()
    generator = df.get_data()
    t0 = time.time()
    t = time.time()
    for i, dp in enumerate(generator):
        print(i, time.time() - t)
        t = time.time()
        if i == 100:
            break
    print(time.time() - t0)
