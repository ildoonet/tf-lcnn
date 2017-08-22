import numpy as np
import tensorflow as tf
from mnist import MNIST
import imgaug as ia
from imgaug import augmenters as iaa

LOG_DIR = '/data/private/tf-lcnn-logs'
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
augseq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        sometimes(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
        sometimes(iaa.Affine(
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
        iaa.SomeOf((0, 5), [
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
            sometimes(iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.2)),
            # move pixels locally around (with random strengths)
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03)))  # sometimes move parts of the image around
        ], random_order=True)
    ],
    random_order=True
)


def flatten_convolution(tensor_in):
    tendor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
    return tensor_in_flat


def dense_layer(tensor_in, layers, activation_fn=tf.nn.tanh, keep_prob=None):
    if keep_prob is None:
        return tf.contrib.layers.stack(tensor_in, tf.contrib.layers.fully_connected, layers, activation_fn=activation_fn)

    tensor_out = tensor_in
    for layer in layers:
        tensor_out = tf.contrib.layers.fully_connected(tensor_out, layer,
                                                       activation_fn=activation_fn)
        tensor_out = tf.contrib.layers.dropout(tensor_out, keep_prob=keep_prob)

    return tensor_out


def get_activation_f(name):
    if name == 'tanh':
        return tf.nn.tanh
    elif name == 'relu':
        return tf.nn.relu
    else:
        raise


class DataMNIST():
    def __init__(self, path, batchsize):
        self.path = path
        self.mndata = MNIST(path=path)
        self.batchsize = batchsize
        self.batch_loader = ia.BatchLoader(self.load_batches)
        self.bg_augmenter = ia.BackgroundAugmenter(batch_loader=self.batch_loader, augseq=augseq, nb_workers=6)

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
            img = images[_cnt]
            lb_onehot = np.zeros(10)
            lb_onehot[labels[_cnt]] = 1

            yield img, lb_onehot
            _cnt += 1
            _cnt %= len(images)

    def load_batches(self):
        gen = self.next_image()
        while True:
            batch_images = []
            batch_data = []
            for _ in range(self.batchsize):
                img, lb = next(gen)
                batch_images.append(img.reshape(28, 28, 1))
                batch_data.append(lb)

            yield ia.Batch(images=np.array(batch_images, dtype=np.uint8),
                           data=batch_data)

    def next_batch(self):
        batch = self.bg_augmenter.get_batch()

        return batch.images_aug.reshape((self.batchsize, 28*28)) / 255, batch.data

    def test_batch(self):
        images_test, labels_test = self.mndata.load_testing()
        images_test = np.asarray(images_test, dtype=np.uint8) / 255
        labels_test = np.asarray(labels_test, dtype=np.uint8)
        labels_test_onehot = np.zeros((len(labels_test), 10))
        labels_test_onehot[np.arange(len(labels_test)), labels_test] = 1
        return images_test, labels_test_onehot
