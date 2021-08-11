#include <torch/script.h>
#include <limits>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cpu/DepthwiseConvKernel.h>
//#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/ConvUtils.h>
//#include <ATen/native/xnnpack/Engine.h>

#include <ATen/Config.h>
#include <c10/macros/Macros.h>

//#if AT_NNPACK_ENABLED()
#include <nnpack.h>
//#endif

#include <c10/util/ArrayRef.h>
#include <vector>
#include <ideep.hpp>

#include <ATen/native/mkldnn/MKLDNNCommon.cpp>
#include <ATen/native/mkldnn/Utils.h>

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>

#include <ATen/native/mkldnn/Conv.cpp>
#include <ATen/native/mkldnn/Linear.cpp>
#include <ATen/native/mkldnn/Normalization.cpp>

#define IV_MAC_BYTE 28
#define IV_MAC_LENGTH 7
#define META_DATA_BYTE 5
#define RESERVED_LENGTH 25

inline std::vector<int64_t> expand_param_if_needed(
    c10::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

/*ideep::tensor dense_to_mkldnn(const torch::Tensor& cpu_tensor) {
  // TODO: consider to convert non-contiguous tensor to `ideep::tensor` directly.
    auto cpu_tensor_cont = cpu_tensor.contiguous();
    auto data_type = at::ScalarType::Float;
    torch::Tensor mkldnn_tensor = at::native::empty_mkldnn(cpu_tensor_cont.sizes(), data_type,
                                      cpu_tensor_cont.options().layout_opt(), cpu_tensor_cont.options().device_opt(),
                                      cpu_tensor_cont.options().pinned_memory_opt());
    ideep::tensor& dtensor = at::native::itensor_from_mkldnn(mkldnn_tensor);
    return dtensor;
}*/



torch::Tensor secure_linear(
    const torch::Tensor& input_r, const torch::Tensor& weight_r, const torch::Tensor& bias_r) {

    auto input = input_r;
    auto weight = weight_r;
    std::vector<int64_t> weight_dims;
    uint32_t weight_dimension = weight[0].item<int32_t>();
    int weight_amount = 1;
    for (int i=1; i<=weight_dimension; i++) {
        weight_dims.push_back(weight[i].item<int32_t>());
	weight_amount *= weight[i].item<int>();
    }
    
    c10::IntArrayRef weight_size(weight_dims);
    auto weight_empty = torch::empty(weight_size, torch::kFloat);

    auto bias = bias_r;
    std::vector<int64_t> bias_dims;
    uint32_t bias_dimension = bias[0].item<int32_t>();
    int bias_amount = 1;
    for (int i=1; i<=bias_dimension; i++) {
        bias_dims.push_back(bias[i].item<int32_t>());
        bias_amount *= bias[i].item<int>();
    }
    
    c10::IntArrayRef bias_size(bias_dims);
    auto bias_empty = torch::empty(bias_size, torch::kFloat); 

    //int64_t dim = dimension - 2;
    //auto weight;
    auto weight_handle = weight.data_ptr<int>();
    auto bias_handle = bias.data_ptr<int>();
    
    auto data_type = at::ScalarType::Float;
    const ideep::tensor mkldnn_weight = at::native::itensor_from_tensor(weight_empty.contiguous());
    memcpy(mkldnn_weight.get_data_handle(), &weight_handle[weight_dimension+1+IV_MAC_LENGTH+RESERVED_LENGTH], (weight_amount*sizeof(int)));
    const ideep::tensor mkldnn_bias = at::native::itensor_from_tensor(bias_empty.contiguous());
    memcpy(mkldnn_bias.get_data_handle(), &bias_handle[bias_dimension+1+IV_MAC_LENGTH+RESERVED_LENGTH], (bias_amount*sizeof(int)));
    const ideep::tensor mkldnn_input = at::native::itensor_from_tensor(input.contiguous());
    
    uint8_t weight_iv_mac[IV_MAC_BYTE];
    memcpy(weight_iv_mac, &weight_handle[weight_dimension+1+RESERVED_LENGTH], sizeof(weight_iv_mac));
    uint8_t bias_iv_mac[IV_MAC_BYTE];
    memcpy(bias_iv_mac, &bias_handle[bias_dimension+1+RESERVED_LENGTH], sizeof(bias_iv_mac));
    
    uint32_t weight_model_id;
    uint32_t bias_model_id;
    uint32_t model_id;
    memcpy(&weight_model_id, &weight_handle[weight_dimension+2], sizeof(weight_model_id));
    memcpy(&bias_model_id, &bias_handle[bias_dimension+2], sizeof(bias_model_id));

    if (weight_model_id == bias_model_id){
	model_id = weight_model_id;
    }
    else {
        std::ostringstream ss;
        ss << "weight_model_id and bias_model_id not equal ";
        AT_ERROR(ss.str());
    }


    ideep::tensor mkldnn_output = at::native::mkldnn_secure_linear(
      mkldnn_input,
      mkldnn_weight,
      mkldnn_bias,
      (void*)weight_iv_mac,
      IV_MAC_BYTE,
      (void*)bias_iv_mac,
      IV_MAC_BYTE,
      (void*)&model_id);
    /*size_t bytes;
    bytes = mkldnn_output.get_desc().get_size();
    float *output_data = static_cast<float *>(mkldnn_output.get_data_handle());
    for (size_t i = 0; i < bytes/sizeof(float); ++i) {
        printf("###%d### %f. 0x", i, output_data[i]);
        for(int j=0; j<4; j++)
            printf("%X ", *((uint8_t*)&output_data[i] +j) );
    }
    printf("\n");*/
    return  at::native::mkldnn_to_dense(
        at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
}

torch::Tensor secure_conv(
    const torch::Tensor& input_r, const torch::Tensor& weight_r, const torch::Tensor& bias_r,
    c10::IntArrayRef stride_, c10::IntArrayRef padding_, c10::IntArrayRef dilation_,
    bool transposed_, c10::IntArrayRef output_padding_, int64_t groups_) {

    auto input = input_r;
    //std::cout << weight_r << std::endl;
    auto weight = weight_r;
    /*auto a = weight.data_ptr<uint8_t>();
    for(int j=0;j<weight.numel()*4;j++) {
	printf("%X ", a[j] );
	if (j<16 && j%4==0) {
            printf("===%d===", *((uint32_t*)(a+j)) );
	}
    }*/
    
    std::vector<int64_t> weight_dims;
    uint32_t weight_dimension = weight[0].item<int32_t>();
    int weight_amount = 1;
    for (int i = 1; i <= weight_dimension; i++) {
        weight_dims.push_back(weight[i].item<int32_t>());
        weight_amount *= weight[i].item<int>();
    }

    //c10::Scalar dim_sc = c10::Scalar(amount);

    c10::IntArrayRef weight_size(weight_dims);
    auto x = torch::empty(weight_size, torch::kFloat);
    //std::cout << size << std::endl << amount << std::endl;
    //std::cout << weight << std::endl << "================================" << std::endl;
    auto bias = bias_r;

    std::vector<int64_t> bias_dims;
    uint32_t bias_dimension = bias[0].item<int32_t>();
    int bias_amount = 1;
    for (int i=1; i<=bias_dimension; i++) {
        bias_dims.push_back(bias[i].item<int32_t>());
	bias_amount *= bias[i].item<int>();
    }

    c10::IntArrayRef bias_size(bias_dims);
    auto empty_bias = torch::empty(bias_size, torch::kFloat);

    int64_t dim = weight_dimension - 2;
    auto padding = expand_param_if_needed(padding_, "padding", dim);
    auto stride = expand_param_if_needed(stride_, "stride", dim);
    auto dilation = expand_param_if_needed(dilation_, "dilation", dim);
    auto groups = groups_;
    //const ideep::tensor tensor_weight = at::native::itensor_from_tensor(weight.contiguous());
    auto weight_handle = weight.data_ptr<int>();
    auto bias_handle = bias.data_ptr<int>();

    auto data_type = at::ScalarType::Float;
    const ideep::tensor mkldnn_weight = at::native::itensor_from_tensor(x.contiguous());
    memcpy(mkldnn_weight.get_data_handle(), &weight_handle[weight_dimension+1+IV_MAC_LENGTH+RESERVED_LENGTH], (weight_amount*sizeof(int)));
    const ideep::tensor mkldnn_bias = at::native::itensor_from_tensor(empty_bias.contiguous());
    memcpy(mkldnn_bias.get_data_handle(), &bias_handle[bias_dimension+1+IV_MAC_LENGTH+RESERVED_LENGTH], (bias_amount*sizeof(int)));
    const ideep::tensor mkldnn_input = at::native::itensor_from_tensor(input.contiguous());
    //std::cout << "###################2" << std::endl;
    //ideep::tensor mkldnn_bias;
    //if(bias.defined()) {
    //    mkldnn_bias = at::native::itensor_from_tensor(bias);
    //}
    //std::cout << "###################3" << std::endl;

    //size_t bytes;
    //bytes = mkldnn_weight.get_desc().get_size();
    //float *src_data = static_cast<float *>(mkldnn_weight.get_data_handle());
    //for (size_t i = 0; i < bytes/sizeof(float); ++i) {
    //    printf("###%d### %f. 0x", i, src_data[i]);
    //    for(int j=0; j<4; j++)
    //        printf("%X ", *((uint8_t*)&src_data[i] +j) );
    //}
    //printf("\n");

    uint8_t weight_iv_mac[IV_MAC_BYTE];
    memcpy(weight_iv_mac, &weight_handle[weight_dimension+1+RESERVED_LENGTH], sizeof(weight_iv_mac));
    uint8_t bias_iv_mac[IV_MAC_BYTE];
    memcpy(bias_iv_mac, &bias_handle[bias_dimension+1+RESERVED_LENGTH], sizeof(bias_iv_mac));

    uint32_t weight_model_id;
    uint32_t bias_model_id;
    uint32_t model_id;
    memcpy(&weight_model_id, &weight_handle[weight_dimension+2], sizeof(weight_model_id));
    memcpy(&bias_model_id, &bias_handle[bias_dimension+2], sizeof(bias_model_id));
    if (weight_model_id == bias_model_id){
        model_id = weight_model_id;
    }
    else {
        std::ostringstream ss;
        ss << "weight_model_id and bias_model_id not equal ";
        AT_ERROR(ss.str());
    }

    ideep::tensor mkldnn_output = at::native::mkldnn_secure_convolution(
      mkldnn_input,
      mkldnn_weight,
      mkldnn_bias,
      padding,
      stride,
      dilation,
      groups,
      (void*)weight_iv_mac,
      IV_MAC_BYTE,
      (void*)bias_iv_mac,
      IV_MAC_BYTE,
      (void*)&model_id);

    return  at::native::mkldnn_to_dense(
        at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
}

torch::Tensor secure_batch_norm(
//std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> secure_batch_norm(
    const torch::Tensor& input_r,
    const torch::Tensor& weight_r,
    const torch::Tensor& bias_r,
    const torch::Tensor& running_mean_r,
    const torch::Tensor& running_var_r,
    double momentum_r,
    double eps_r) {

//    std::cout << "call in secure batch norm" << std::endl;    
    auto input = input_r;
    auto weight = weight_r;
    std::vector<int64_t> weight_dims;
    uint32_t weight_dimension = weight[0].item<int32_t>();
    int weight_amount = 1;
    for (int i=1; i<=weight_dimension; i++) {
        weight_dims.push_back(weight[i].item<int32_t>());
        weight_amount *= weight[i].item<int>();
    }

    c10::IntArrayRef weight_size(weight_dims);
    auto weight_empty = torch::empty(weight_size, torch::kFloat);


    auto bias = bias_r;
    std::vector<int64_t> bias_dims;
    uint32_t bias_dimension = bias[0].item<int32_t>();
    int bias_amount = 1;
    for (int i=1; i<=bias_dimension; i++) {
        bias_dims.push_back(bias[i].item<int32_t>());
        bias_amount *= bias[i].item<int>();
    }

    c10::IntArrayRef bias_size(bias_dims);
    auto bias_empty = torch::empty(bias_size, torch::kFloat);


    //int64_t dim = dimension - 2;
    //auto weight;
    auto running_mean = running_mean_r;
    std::vector<int64_t> running_mean_dims;
    uint32_t running_mean_dimension = running_mean[0].item<int32_t>();
    int running_mean_amount = 1;
    for (int i=1; i<=running_mean_dimension; i++) {
        running_mean_dims.push_back(running_mean[i].item<int32_t>());
	running_mean_amount *= running_mean[i].item<int>();
    }

    c10::IntArrayRef running_mean_size(running_mean_dims);
    auto running_mean_empty = torch::empty(running_mean_size, torch::kFloat);


    auto running_var = running_var_r;
    std::vector<int64_t> running_var_dims;
    uint32_t running_var_dimension = running_var[0].item<int32_t>();
    int running_var_amount = 1;
    for (int i=1; i<=running_var_dimension; i++) {
        running_var_dims.push_back(running_var[i].item<int32_t>());
	running_var_amount *= running_var[i].item<int>();
    }

    c10::IntArrayRef running_var_size(running_var_dims);
    auto running_var_empty = torch::empty(running_var_size, torch::kFloat);


    auto momentum = momentum_r;
    auto eps = eps_r;

    auto weight_handle = weight.data_ptr<int>();
    auto bias_handle = bias.data_ptr<int>();
    auto running_mean_handle = running_mean.data_ptr<int>();
    auto running_var_handle = running_var.data_ptr<int>();

    auto data_type = at::ScalarType::Float;
    const ideep::tensor mkldnn_weight = at::native::itensor_from_tensor(weight_empty.contiguous());
    memcpy(mkldnn_weight.get_data_handle(), &weight_handle[weight_dimension+1+IV_MAC_LENGTH+RESERVED_LENGTH], (weight_amount*sizeof(int)));
    const ideep::tensor mkldnn_bias = at::native::itensor_from_tensor(bias_empty.contiguous());
    memcpy(mkldnn_bias.get_data_handle(), &bias_handle[bias_dimension+1+IV_MAC_LENGTH+RESERVED_LENGTH], (bias_amount*sizeof(int)));

    const ideep::tensor mkldnn_input = at::native::itensor_from_tensor(input.contiguous());

    const ideep::tensor mkldnn_running_mean = at::native::itensor_from_tensor(running_mean_empty.contiguous());
    memcpy(mkldnn_running_mean.get_data_handle(), &running_mean_handle[running_mean_dimension+1+IV_MAC_LENGTH+RESERVED_LENGTH], (running_mean_amount*sizeof(int)));
    const ideep::tensor mkldnn_running_var = at::native::itensor_from_tensor(running_var_empty.contiguous());
    memcpy(mkldnn_running_var.get_data_handle(), &running_var_handle[running_var_dimension+1+IV_MAC_LENGTH+RESERVED_LENGTH], (running_var_amount*sizeof(int)));

    uint8_t weight_iv_mac[IV_MAC_BYTE];
    memcpy(weight_iv_mac, &weight_handle[weight_dimension+1+RESERVED_LENGTH], sizeof(weight_iv_mac));
    uint8_t bias_iv_mac[IV_MAC_BYTE];
    memcpy(bias_iv_mac, &bias_handle[bias_dimension+1+RESERVED_LENGTH], sizeof(bias_iv_mac));
    uint8_t running_mean_iv_mac[IV_MAC_BYTE];
    memcpy(running_mean_iv_mac, &running_mean_handle[running_mean_dimension+1+RESERVED_LENGTH], sizeof(running_mean_iv_mac));
    uint8_t running_var_iv_mac[IV_MAC_BYTE];
    memcpy(running_var_iv_mac, &running_var_handle[running_var_dimension+1+RESERVED_LENGTH], sizeof(running_var_iv_mac));
    
    uint32_t weight_model_id;
    uint32_t bias_model_id;
    uint32_t model_id;
    memcpy(&weight_model_id, &weight_handle[weight_dimension+2], sizeof(weight_model_id));
    memcpy(&bias_model_id, &bias_handle[bias_dimension+2], sizeof(bias_model_id));

    if (weight_model_id == bias_model_id){
        model_id = weight_model_id;
    }
    else {
        std::ostringstream ss;
        ss << "weight_model_id and bias_model_id not equal ";
        AT_ERROR(ss.str());
    }

    auto mkldnn_output = at::native::mkldnn_secure_batch_norm(
      mkldnn_input,
      mkldnn_weight,
      mkldnn_bias,
      mkldnn_running_mean,
      mkldnn_running_var,
      false,
      momentum,
      eps,
      (void*)weight_iv_mac,
      IV_MAC_BYTE,
      (void*)bias_iv_mac,
      IV_MAC_BYTE,
      (void*)running_mean_iv_mac,
      IV_MAC_BYTE,
      (void*)running_var_iv_mac,
      IV_MAC_BYTE,
      (void*)&model_id);

    /*size_t bytes;
    bytes = mkldnn_output.get_desc().get_size();
    float *output_data = static_cast<float *>(mkldnn_output.get_data_handle());
    for (size_t i = 0; i < bytes/sizeof(float); ++i) {
        printf("###%d### %f. 0x", i, output_data[i]);
        for(int j=0; j<4; j++)
            printf("%X ", *((uint8_t*)&output_data[i] +j) );
    }
    printf("\n");*/

    //return std::make_tuple(
    //    at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
    //                           input.options().device_opt()),
    //    at::native::new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(input.options().dtype_opt()),
    //                            input.options().device_opt()),
    //    at::native::new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(input.options().dtype_opt()),
    //                            input.options().device_opt()));
    //return  mkldnn_output;

    return at::native::mkldnn_to_dense(
        at::native::new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
}

TORCH_LIBRARY(my_ops, m) {
	          m.def("secure_conv", secure_conv);
		  m.def("secure_linear", secure_linear);
		  m.def("secure_batch_norm", secure_batch_norm);
}


