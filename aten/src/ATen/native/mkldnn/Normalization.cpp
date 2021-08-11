#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

ideep::tensor mkldnn_secure_batch_norm(
    const ideep::tensor& input,
    const ideep::tensor& weight,
    const ideep::tensor& bias,
    const ideep::tensor& running_mean,
    const ideep::tensor& running_var,
    bool train,
    double momentum,
    double eps,
    void* weight_iv_mac,
    size_t weight_meta_size,
    void* bias_iv_mac,
    size_t bias_meta_size,
    void* mean_iv_mac,
    size_t mean_meta_size,
    void* var_iv_mac,
    size_t var_meta_size,
    void* model_id) {
  //ideep::tensor& x = itensor_from_mkldnn(input);
  //ideep::tensor& w = itensor_from_mkldnn(weight);
  //ideep::tensor& b = itensor_from_mkldnn(bias);
  auto x = input;
  auto w = weight;
  auto b = bias;
  auto m = running_mean;
  auto v = running_var;
  //ideep::tensor m = itensor_from_tensor(running_mean);
  //ideep::tensor v = itensor_from_tensor(running_var);

  ideep::tensor y;

  //TODO:check input dims (ideep::tensor) like this (Tensor):
  //TORCH_CHECK(input.dim() == 4 || input.dim() == 5,
  //           "mkldnn_batch_norm: currently mkldnn only support 2d and 3d batchnorm");

  auto& ctx = at::globalContext();
  sgx_enclave_id_t eid = ctx.getEid();

  ideep::batch_normalization_forward_inference::compute(
      x, m, v, w, b, y, eps, weight_iv_mac, weight_meta_size, bias_iv_mac, bias_meta_size, mean_iv_mac, mean_meta_size, var_iv_mac, var_meta_size, model_id, &eid);

  ctx.setEid(eid);
  return y;
  /*std::make_tuple(
      new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                              input.options().device_opt()),
      new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(input.options().dtype_opt()),
                              input.options().device_opt()),
      new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(input.options().dtype_opt()),
                              input.options().device_opt()));
			      */
  }


std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor& w = itensor_from_mkldnn(weight);
  ideep::tensor& b = itensor_from_mkldnn(bias);
  ideep::tensor& m = itensor_from_mkldnn(running_mean);
  ideep::tensor& v = itensor_from_mkldnn(running_var);

  ideep::tensor y;

  if (train) {
    // TODO: support training
    TORCH_CHECK(false, "mkldnn_batch_norm: mkldnn training is not supported in yet.");

    // ideep::tensor saved_mean;
    // ideep::tensor saved_var;
    // ideep::batch_normalization_forward_training::compute<AllocForMKLDNN>(
    //     x, w, b, y, saved_mean, saved_var, m, v, momentum, eps);
    // return std::make_tuple(
    //     new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
    //                             input.options().device_opt()),
    //     new_with_itensor_mkldnn(std::move(saved_mean), optTypeMetaToScalarType(input.options().dtype_opt()),
    //                             input.options().device_opt()),
    //     new_with_itensor_mkldnn(std::move(saved_var), optTypeMetaToScalarType(input.options().dtype_opt()),
    //                             input.options().device_opt()));
  } else {
    TORCH_CHECK(input.dim() == 4 || input.dim() == 5,
               "mkldnn_batch_norm: currently mkldnn only support 2d and 3d batchnorm");

    //auto& ctx = at::globalContext();
    //sgx_enclave_id_t eid = ctx.getEid();

    ideep::batch_normalization_forward_inference::compute(
        x, m, v, w, b, y, eps);//, &eid);

    //ctx.setEid(eid);
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()),
        new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()),
        new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
  }
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
