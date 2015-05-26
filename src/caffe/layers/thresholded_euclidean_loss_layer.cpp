#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe{

template <typename Dtype>
void ThresholdedEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1)) << "Inputs must have the same dimension.";
  int d = bottom[0]->count() / bottom[0]->num();
  CHECK_EQ(d,1) << "Thresholded Euclidean Loss only supports 1-dimensional input";
  diff_.ReshapeLike(*bottom[0]);
  vector<int> size;
  size.push_back( diff_.num() );
  diff_.Reshape(size);
  distances_.Reshape(size);
  min_y_minus_t_.Reshape(size);
  min_y_hat_minus_t_.Reshape(size);
  weights_.Reshape(size);

  if( this->layer_param_.has_threshold_param() ) {
    threshold_ = this->layer_param_.threshold_param().threshold();
  }
}

template <typename Dtype>
void ThresholdedEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
    count,
    bottom[0]->cpu_data(),
    bottom[1]->cpu_data(),
    diff_.mutable_cpu_data());

  for( int idx=0; idx < bottom[0]->num();idx++ ) {
    Dtype y = bottom[0]->cpu_data()[idx];
    Dtype yhat = bottom[1]->cpu_data()[idx];
    Dtype weight_l = std::min( std::max(Dtype(0.0), y - threshold_), Dtype(1.0));
    Dtype weight_r = std::min( std::max(Dtype(0.0), yhat - threshold_), Dtype(1.0));
    Dtype weight = std::max( weight_l, weight_r );
    weights_.mutable_cpu_data()[idx] = weight;
  }

  caffe_mul(
    count,
    diff_.cpu_data(),
    weights_.cpu_data(),
    diff_.mutable_cpu_data()
  );

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ThresholdedEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
        bottom[i]->count(),              // count
        alpha,                              // alpha
        diff_.cpu_data(),                   // a
        Dtype(0),                           // beta
        bottom[i]->mutable_cpu_diff());  // b
      }
  }
}

INSTANTIATE_CLASS(ThresholdedEuclideanLossLayer);
REGISTER_LAYER_CLASS(ThresholdedEuclideanLoss);

}
