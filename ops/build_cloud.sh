#! /bin/bash

mv -f /root/repos/tensorflow/tensorflow/core/user_ops/BUILD /root/repos/tensorflow/tensorflow/core/user_ops/BUILD.bak
cp BUILD /root/repos/tensorflow/tensorflow/core/user_ops/
cp sparse_conv2d.cc /root/repos/tensorflow/tensorflow/core/user_ops/

cd /root/repos/tensorflow/

bazel build --config opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/core/user_ops:sparse_conv2d.so
