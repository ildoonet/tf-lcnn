# tf-lcnn : Predict Faster using Models Trained Fast with Multi-GPUs

Tensorflow implementation for ['LCNN: Lookup-based Convolutional Neural Network'](https://arxiv.org/abs/1611.06473)

This also have an implementations multi-gpu training codes for various models, so you can train your own model faster and predict images faster with Lookup Convolutions.

## Implementations

[x] Achieve MNist, ILSVRC2012 Baseline

[x] Training Imagenet on Multiple node with multiple gpus

[x] Training Code - Lookup-based Convolution Layer

[ ] Same training result as the original paper

[ ] Inference Code - Optimized Dense Matrix Operation

[ ] Fast inference speed as the original paper

## Training Results

### MNIST Dataset on Single/Multi-GPU Cluster

| Model           | GPU                 | Accuracy       | Training Time    | Etc                        |
|:----------------|:--------------------|---------------:|:-----------------|:---------------------------|
| Lenet           | 1 GPU               | 99.44%         | 21m              | Epoch 53.3                 |
| Lenet           | 8 GPU x 1 Machine   | 99.40%         | 2m 26s           | Epoch 53.3                 |
| Alexnet         | 1 GPU               | 99.45%         | 11h 47m          | Epoch 300, Resized 224x224 from 28x28 |

* Performance using Alexnet Architecture might be improved with further tuning of hyper parameters. 

### Imagenet ILSVRC2012 Classification Task

## Inference Tests



## References & Opensource Pakcages

[1] [LCNN: Lookup-based Convolutional Neural Network](https://arxiv.org/abs/1611.06473)

[2] http://openresearch.ai/t/lcnn-lookup-based-convolutional-neural-network

[3] author's code : https://github.com/hessamb/lcnn/blob/master/layers/PooledSpatialConvolution.lua

[4] [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

[] imagenet training on alexnet : https://github.com/dontfollowmeimcrazy/imagenet

[] https://github.com/mouradmourafiq/tensorflow-convolution-models

[] Distributed Tensorflow : https://www.tensorflow.org/deploy/distributed

[] Distributed Tensorflow Example : https://github.com/tensorflow/models/tree/master/inception

[] https://github.com/hpssjellis/easy-tensorflow-on-cloud9/blob/master/aymericdamien-Examples/examples/alexnet.py

[] https://github.com/sugyan/tensorflow-mnist

[] https://github.com/sorki/python-mnist

[] https://github.com/grfiv/MNIST/blob/master/MNIST.pdf

[] imgaug : https://github.com/aleju/imgaug

[] Giridhar Pemmasani, "dispy: Distributed and Parallel Computing with/for Python", http://dispy.sourceforge.net, 2016.

[] https://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow/37774470#37774470