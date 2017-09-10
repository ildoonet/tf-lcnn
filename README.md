# tf-lcnn : Predict Faster using Models Trained Fast with Multi-GPUs

Tensorflow implementation for ['LCNN: Lookup-based Convolutional Neural Network'](https://arxiv.org/abs/1611.06473)

This also have an implementations multi-gpu training codes for various models, so you can train your own model faster and predict images faster with Lookup Convolutions.

![Lookup Convolution](/images/paper_figure1.png)

## Implementations

[x] Achieve MNist, ILSVRC2012 Baseline

[x] Training Imagenet on Multiple node with multiple gpus

[x] Training Code - Lookup-based Convolution Layer

[x] Same training result as the original paper

[x] Inference Code - Optimized Dense Matrix Operation

[x] Fast inference speed as the original paper

## Custom Operation for Sparse Convolutional Layer

### Build

Custom Operation have been implemented for LCNN's lookup convolution.

Source codes in [/ops](/ops), and it should be build before run the inference code.

```
$ cp {tf-lcnn}/ops/* {tensorflow}/tensorflow/core/user_ops/
$ bazel build --config opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/core/user_ops:sparse_conv2d
```

### Performance

As you can see below timeline, this custom lookup convolutional operation has very little weight in the whole time.

![inference timeline](/images/timeline_alexnet.png)

* Alexnet, Total inference time 26ms
* 1 core, single thread

## Training Results

Alexnet's Fully connected layer was replaced with convolutional layer. 

* Codes will be optimized soon and inference times will be updated.

### MNIST Dataset

For LCNN Model, Two versions of networks were trained for experiments.

* LCNN-Fast
  * Dictionary Size : 3, 30, 30, 30, 30, 512, 512
  * Sparsity : 0.083, 0.034, 0.008, 0.013, 0.027, 0.001, 0.002
* LCNN-Accurate
  * Dictionary Size : 3, 500, 500, 500, 30, 1024, 1024
  * Sparsity : 

The original paper was not evaluated on MNIST, but the dataset was suitable for rapid experiments.

| Model           | Conv. Filter         | Inference           | GPU | Training Time | Etc                        |
|:----------------|:---------------------|:--------------------|-----------:|:--------------|:---------------------------|
| Alexnet         | Convolution          | - / 99.98%          | 1 GPU      | 1h 35m        | Epoch 40, Batch 128 |
| Alexnet         | Convolution          | - / 99.22%          | 4 GPU      | 27m (x3.5)    | Epoch 40, Batch 512 |
| | | | | |
| Alexnet         | LCNN-Fast            | 26ms / 99.87%       | 8 GPU      | 23m           | Epoch 40, Batch 128 |
| Alexnet         | LCNN-Accurate        | - /99.43%           | 8 GPU      | 23m           | Epoch 40, Batch 128 |

### Imagenet ILSVRC2012 Classification Task

| Model           | Convolutional Filter | GPU                 | Accuracy(Top1/Top5) | Training Time         | Etc                        |
|:----------------|:---------------------|:--------------------|--------------------:|:----------------------|:---------------------------|
| Alexnet         | Convolution          | 1 GPU               | 59.40% / 81.50%     | 53h                   | Epoch 65, Batch 128        |
| Alexnet         | Convolution          | 4 GPU               | 59.21% / 81.33%     | 14h (x3.78)           | Epoch 65, Batch 128        |
| | | | | |
| Alexnet-LCNN    | LCNN-Fast            | 1 GPU               |                     |                       | Epoch 65, Batch 128        |
| Alexnet-LCNN    | LCNN-Accurate        | 1 GPU               |                     |                       | Epoch 65, Batch 128        |

----

## References & Opensource Pakcages

This code is very experimental and have been helped a lot from various websites. 

### LCNN

[1] [LCNN: Lookup-based Convolutional Neural Network](https://arxiv.org/abs/1611.06473)

[2] http://openresearch.ai/t/lcnn-lookup-based-convolutional-neural-network

[3] author's code : https://github.com/hessamb/lcnn/blob/master/layers/PooledSpatialConvolution.lua

### Base Networks (LENET, Alexnet) & Datasets (MNIST, ImageNet)

[1] [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

[2] imagenet training on alexnet : https://github.com/dontfollowmeimcrazy/imagenet

[3] https://github.com/mouradmourafiq/tensorflow-convolution-models

[4] https://github.com/hpssjellis/easy-tensorflow-on-cloud9/blob/master/aymericdamien-Examples/examples/alexnet.py

### Tensorflow Custom Operation

[1] https://www.tensorflow.org/extend/adding_an_op

[2] http://davidstutz.de/implementing-tensorflow-operations-in-c-including-gradients/

[3] https://github.com/tensorflow/tensorflow/blob/8eaf671025e8cd5358278f91f7e89e2fbbe6a26b/tensorflow/core/kernels/conv_ops.cc#L94

[4] https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/sparse_ops.py

[5] https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc#L503

[6] https://github.com/tensorflow/tensorflow/issues/2412

#### Tensorflow Build for Cmake

[1] https://www.tensorflow.org/install/install_sources

[2] https://github.com/cjweeks/tensorflow-cmake

[3] https://github.com/tensorflow/tensorflow/issues/2412

### Multi GPU / Multi Node Training

[1] Distributed Tensorflow : https://www.tensorflow.org/deploy/distributed

[2] Distributed Tensorflow Example : https://github.com/tensorflow/models/tree/master/inception

[3] https://research.fb.com/publications/imagenet1kin1h/

### Training Techniques

[1] https://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow/37774470#37774470

[2] https://github.com/ppwwyyxx/tensorpack

[3] https://github.com/sorki/python-mnist

[4] imgaug : https://github.com/aleju/imgaug
