#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/util/tensor_format.h"
//#include "tensorflow/core/framework/tensor_shape.h"
//#include "tensorflow/core/framework/common_shape_fns.h"

//#include "tensorflow/core/platform/default/logging.h"

using namespace tensorflow;
using namespace shape_inference;


REGISTER_OP("SparseConv2D")
  .Input("input: float")
  .Input("weight_indices: int32")
  .Input("weight_values: float")
  .Attr("dense_shape: list(int)")
  .Attr("strides: list(int)")
  .Output("sparse_conv2d: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

    ShapeHandle weight_indices_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_indices_shape));

    ShapeHandle weight_values_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &weight_values_shape));

    std::vector<int32> dense_shape;
    TF_RETURN_IF_ERROR(c->GetAttr("dense_shape", &dense_shape));

    if (dense_shape.size() != 4) {
        return errors::InvalidArgument("SparseConv2D requires the dense_shape attribute to contain"
                                   " 4 values, but got: ",
                                   dense_shape.size());
    }

    std::vector<int32> strides;
    TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

    if (strides.size() != 2) {
      return errors::InvalidArgument("SparseConv2D requires the stride attribute to contain"
                                   " 2 values, but got: ",
                                   strides.size());
    }

    TensorFormat data_format;
    if (!FormatFromString("NHWC", &data_format)) {
      return errors::InvalidArgument("Invalid data format string: ", "NHWC");
    }

    auto batch_size_dim = c->Dim(input_shape, 0);
    DimensionHandle output_rows;
    TF_RETURN_IF_ERROR(c->Subtract(c->Dim(input_shape, 1), dense_shape[0], &output_rows));
    TF_RETURN_IF_ERROR(c->Add(output_rows, strides[0], &output_rows));
    TF_RETURN_IF_ERROR(c->Divide(output_rows, strides[0], false, &output_rows));

    DimensionHandle output_cols;
    TF_RETURN_IF_ERROR(c->Subtract(c->Dim(input_shape, 2), dense_shape[1], &output_cols));
    TF_RETURN_IF_ERROR(c->Add(output_cols, strides[1], &output_cols));
    TF_RETURN_IF_ERROR(c->Divide(output_cols, strides[1], false, &output_cols));

    auto output_depth_dim = GetTensorDim(dense_shape, data_format, 'C');

    ShapeHandle output_shape;
    output_shape = c->MakeShape({batch_size_dim, output_rows, output_cols, output_depth_dim});
    c->set_output(0, output_shape);

    return Status::OK();
  });

class SparseConv2DOp : public OpKernel {
public:
  explicit SparseConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(context, context->GetAttr("dense_shape", &dense_shape_));

  }

  std::vector<int32> strides_;
  std::vector<int32> dense_shape_;

  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* c) override {
    DCHECK_EQ(3, c->num_inputs());

    // get the input tensor
    const Tensor& input = c->input(0);
    OP_REQUIRES(c, input.dims() == 4, errors::InvalidArgument("input must be 4-dimensional"));

    // get the weight tensor
    const Tensor& weight_indices = c->input(1);
    const Tensor& weight_values = c->input(2);

    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weight_indices.shape();
    DCHECK_EQ(input_shape.dims(), 4);

    // check weights is matrix of correct size
    DCHECK_EQ(weights_shape.dims(), 2);

    // create output shape
    auto out_rows = (input.dim_size(1) - dense_shape_[0] + strides_[0]) / strides_[0];
    auto out_cols = (input.dim_size(2) - dense_shape_[1] + strides_[1]) / strides_[1];
    TensorShape output_shape({input.shape().dim_size(0), out_rows, out_cols, dense_shape_[3]});

    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));

    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.tensor<float, 4>();
    auto index_tensor = weight_indices.matrix<int>();
    auto values_tensor = weight_values.flat<float>();
    auto output_tensor = output->tensor<float, 4>();

    for (int batch_idx = 0; batch_idx < input.shape().dim_size(0); batch_idx ++) {
        for (int sparse_idx = 0; sparse_idx < weight_indices.shape().dim_size(0); sparse_idx ++) {
            int sparse_oc = index_tensor(sparse_idx, 0);
            int sparse_p = index_tensor(sparse_idx, 1);
            int sparse_ic = sparse_p / (dense_shape_[1] * dense_shape_[2]);
            int sparse_ix = (sparse_p % (dense_shape_[1] * dense_shape_[2])) / dense_shape_[1];
            int sparse_iy = (sparse_p % (dense_shape_[1] * dense_shape_[2])) % dense_shape_[1];

            int sparse_v = values_tensor(sparse_idx);

            for (int row = 0; row < input.shape().dim_size(0); row += strides_[0]) {
                int out_row = row / strides_[0];
                for (int col = 0; col < input.shape().dim_size(1); col += strides_[1]) {
                    int out_col = col / strides_[1];

                    output_tensor(batch_idx, out_row, out_col, sparse_oc) += input_tensor(batch_idx, row + sparse_ix, col + sparse_iy, sparse_ic) * sparse_v;
                }
            }

        }
    }

//    for (int i = 0; i < output->shape().dim_size(0); i++) {
//      output_tensor(i, 0) = 0;
//      for (int j = 0; j < weights.shape().dim_size(1); j++) {
//        output_tensor(i, 0) += weights_tensor(i, j)*input_tensor(j, 0);
//      }
//    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SparseConv2DOp);
};

REGISTER_KERNEL_BUILDER(Name("SparseConv2D").Device(DEVICE_CPU), SparseConv2DOp);