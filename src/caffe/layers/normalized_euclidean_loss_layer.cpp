#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizedEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1)) << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  mean_diff_.ReshapeLike(*bottom[0]);
  vector<int> shape = mean_diff_.shape();
  shape[0] = 1;
  mean_.Reshape( shape );
}

template <typename Dtype>
void NormalizedEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  size_t d = bottom[0]->count() / bottom[0]->num();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype model_loss = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());

  caffe_memset(d, Dtype(0.0), mean_.mutable_cpu_data());

  for( int i=0; i < bottom[0]->num(); i++ ) {
    caffe_add( d, bottom[0]->cpu_data() + i * d, mean_.cpu_data(), mean_.mutable_cpu_data() );
  }
  caffe_scal(d, Dtype(1.0) / Dtype(bottom[0]->count()), mean_.mutable_cpu_data() );
  
  for( int i=0; i < bottom[0]->num(); i++ ) {
    caffe_sub(
      d,
      bottom[0]->cpu_data() + i*d,
      mean_.cpu_data(),
      mean_diff_.mutable_cpu_data() + i*d
    );
  }

  Dtype mean_loss = caffe_cpu_dot(count, mean_diff_.cpu_data(), mean_diff_.cpu_data() );
  //printf("mean loss %f\n", mean_loss );
  //printf("model loss %f\n", model_loss );
  top[0]->mutable_cpu_data()[0] = model_loss / mean_loss;
}

INSTANTIATE_CLASS(NormalizedEuclideanLossLayer);
REGISTER_LAYER_CLASS(NormalizedEuclideanLoss);

}
