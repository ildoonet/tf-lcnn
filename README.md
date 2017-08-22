# tf-lcnn
Tensorflow implementation for ['LCNN: Lookup-based Convolutional Neural Network'](https://arxiv.org/abs/1611.06473)

## Status

[x] Training Code - Lookup-based Convolution Layer
[ ] Same training result as the original paper
[ ] Inference Code - Optimized Dense Matrix Operation
[ ] Fast inference speed as the original paper

## Test Results

### MNIST DataSet

While imeplementing lookup-convolutional layer in LCNN paper, MNIST dataset and few known network architectures are used to verify.

| Model       | Accuracy | Inference Time |
|:-----------:|---------:|---------------:|
|Lenet        |99.05%    | |
|Alexnet      |99.20%    | |
|Lenet-LCNN   |98.44%    | |
|Alexnet-LCNN | | |

TODO : Accuracy-Speed Trade off table

## References

[1] [LCNN: Lookup-based Convolutional Neural Network](https://arxiv.org/abs/1611.06473)
[2] http://openresearch.ai/t/lcnn-lookup-based-convolutional-neural-network
[3] [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
[4] https://github.com/mouradmourafiq/tensorflow-convolution-models
[5] https://github.com/hpssjellis/easy-tensorflow-on-cloud9/blob/master/aymericdamien-Examples/examples/alexnet.py
[6] imgaug : https://github.com/aleju/imgaug
[7] pyYaml : http://pyyaml.org/wiki/PyYAMLDocumentation
[8] python-mnist : https://github.com/sorki/python-mnist
[9] author's code : https://github.com/hessamb/lcnn/blob/master/layers/PooledSpatialConvolution.lua

[] https://github.com/dontfollowmeimcrazy/imagenet
[] https://github.com/sugyan/tensorflow-mnist
[] https://github.com/grfiv/MNIST/blob/master/MNIST.pdf
[] disco: https://github.com/discoproject/disco/