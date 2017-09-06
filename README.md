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

| Model           | Convolutional Filter | GPU                 | Validation | Training Time | Etc                        |
|:----------------|:---------------------|:--------------------|-----------:|:--------------|:---------------------------|
| Lenet           | Convolution          | 1 GPU               | 99.99%     | 2m 16s        | Epoch 40                   |
| Lenet           | Convolution          | 4 GPU               | 99.99%     | 53s           | Epoch 40                   |
| Lenet           | Convolution          | 4 GPU               | 99.97%     | 51s *         | Epoch 40                   |
| - | | | | |
| Alexnet         | Convolution          | 1 GPU               | 99.98%     | 1h 35m        | Epoch 40, Batch 128, Resized 224x224  |
| Alexnet         | Convolution          | 2 GPU               | 99.76%     | 52m (x1.8)    | Epoch 40, Batch 256, Resized 224x224  |
| Alexnet         | Convolution          | 4 GPU               | 99.22%     | 27m (x3.5)    | Epoch 40, Batch 512, Resized 224x224  |
| Alexnet         | Convolution          | 8 GPU               | 99.02%     | 21m (x4.4) *  | Epoch 40, Batch 1024, Resized 224x224 |
| - | | | | |
| Alexnet-LCNN    | Lookup-based<br/>Conv. Sparsity(0.66, 0.88, 0.86, 0.86, 0.85)<br/>Conv. Dictionary(10, 100, 100, 100, 100)         | 8 GPU               | 99.95%     | 24m           | Epoch 40, Batch 128, Resized 224x224  |
| Alexnet-LCNN    | Lookup-based<br/>Conv. Sparsity(0.67, 0.56, 0.47, 0.53, 0.51)<br/>Conv. Dictionary(10, 100, 100, 100, 100)         | 8 GPU               | 99.92%     | 23m           | Epoch 40, Batch 128, Resized 224x224  |
| Alexnet-LCNN    | Lookup-based<br/>Conv. Sparsity(0.67, 0.56, 0.47, 0.53, 0.51)<br/>Conv. Dictionary(10, 20, 20, 20, 20)             | 8 GPU               | 99.92%     | 23m           | Epoch 40, Batch 128, Resized 224x224  |

* Batch 128, Warmup-Epoch 10(or 20 for many gpus) for all tests
* Training time is not decreased in Multi-gpu training on Lenet / Alexnet, since the network architecture is super small and data preparation on cpu is the **Bottle-neck**.
* This also includes inference time for full validation set.
* Performance using Alexnet Architecture might be improved with further tuning of hyper parameters.
* Without gradual warm-up introduced in Facebook's recent paper, Multigpu models can not be trained in this experiment.

### Imagenet ILSVRC2012 Classification Task

| Model           | Convolutional Filter | GPU                 | Accuracy(Top1/Top5) | Training Time         | Etc                        |
|:----------------|:---------------------|:--------------------|--------------------:|:----------------------|:---------------------------|
| Alexnet         | Convolution          | 1 GPU               |      |     | Epoch 65, Batch 128        |
| Alexnet         | Convolution          | 4 GPU               | 59.21% / 81.33%     | 14h (11.1 step/sec)   | Epoch 65, Batch 128        |
|- | | | | |
| Alexnet-LCNN    | Lookup-based         | 1 GPU               |                     |                       | Epoch 65, Batch 128        |

----

## References & Opensource Pakcages

### LCNN

[1] [LCNN: Lookup-based Convolutional Neural Network](https://arxiv.org/abs/1611.06473)

[2] http://openresearch.ai/t/lcnn-lookup-based-convolutional-neural-network

[3] author's code : https://github.com/hessamb/lcnn/blob/master/layers/PooledSpatialConvolution.lua

### Base Networks (LENET, Alexnet) & Datasets (MNIST, ImageNet)

[1] [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

[2] imagenet training on alexnet : https://github.com/dontfollowmeimcrazy/imagenet

[3] https://github.com/mouradmourafiq/tensorflow-convolution-models

[4] https://github.com/hpssjellis/easy-tensorflow-on-cloud9/blob/master/aymericdamien-Examples/examples/alexnet.py

### Multi GPU / Multi Node Training

[1] Distributed Tensorflow : https://www.tensorflow.org/deploy/distributed

[2] Distributed Tensorflow Example : https://github.com/tensorflow/models/tree/master/inception

[3] https://research.fb.com/publications/imagenet1kin1h/

### Training Techniques

[1] https://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow/37774470#37774470

[2] https://github.com/ppwwyyxx/tensorpack

[3] https://github.com/sorki/python-mnist

[4] imgaug : https://github.com/aleju/imgaug


