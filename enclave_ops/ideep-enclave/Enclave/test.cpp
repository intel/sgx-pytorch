/*
 * Copyright (C) 2011-2021 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <assert.h>
#include <mbusafecrt.h>

#include <cstring>
#include <iostream>
#include <math.h>
#include <numeric>
#include <string>
#include <sgx_tcrypto.h>

#include "sgx_tseal.h"
#include "sgx_trts.h"

#include "dnnl.hpp"
#include "dnnl_types.h"
#include "dnnl_version.h"
#include "dnnl.h"
#include "dnnl_config.h"
#include "dnnl_debug.h"
#include "mkldnn.hpp"

#include <algorithm>
#include <vector>
#include <map>
#include <Enclave_t.h>

extern "C" void printf(const char *fmt, ...);

typedef struct {
    uint8_t key[16];
} sgx_aes_gcm_128bit_key_struct_t;

std::map<uint32_t, sgx_aes_gcm_128bit_key_struct_t> g_model_keys;

static sgx_status_t get_encryption_key(uint32_t model_id)
{
    /* model key had been retrieved, return directly. */
    if (g_model_keys.find(model_id) != g_model_keys.end())
        return SGX_SUCCESS;

    uint32_t key_blob_len = sgx_calc_sealed_data_size(0, SGX_AESGCM_KEY_SIZE);
    uint8_t key_blob[key_blob_len] = {0};
printf("ocall to get key, key_blob_len is %d.\n", key_blob_len);

    sgx_aes_gcm_128bit_key_struct_t temp_key = {0};

    int ret = -1;
    ocall_get_encryption_key_blob(&ret, model_id, key_blob, key_blob_len);
    if (ret) {
        printf("failed to retrieve the key of model id(%d).\n", model_id);
        return SGX_ERROR_UNEXPECTED;
    }

    uint32_t dec_key_size = sgx_get_encrypt_txt_len((sgx_sealed_data_t *)key_blob);
    if (dec_key_size == UINT32_MAX || dec_key_size != 16) {
        printf("dec_key_size size:%d is not expected: %d.\n", dec_key_size, sizeof(sgx_key_128bit_t));
        return SGX_ERROR_INVALID_PARAMETER;
    }

    sgx_status_t ret2 = sgx_unseal_data((sgx_sealed_data_t *)key_blob, NULL, 0, (uint8_t *)&temp_key, &dec_key_size);
    if (ret2 != SGX_SUCCESS) {
        printf("error(%d) unsealing key.\n", ret2);
        return ret2;
    }

for (int i=0; i<16; i++)
printf("hyhyhyhy: temp_key[%d]=%2d\n", i, temp_key.key[i]);

    g_model_keys.insert(std::pair<uint32_t, sgx_aes_gcm_128bit_key_struct_t>(model_id, temp_key));

    return SGX_SUCCESS;
}

static sgx_status_t AESGCM_decrypt_tensor (void* decrypt_handle,
				       size_t decrypt_bytes,
                                       void* iv_mac,
				       size_t meta_size = 28,
                                       uint32_t model_id = 0)
{
    if (meta_size != 28 || iv_mac == NULL)
        return SGX_ERROR_INVALID_PARAMETER;

    size_t gen_iv_size = 12, gen_mac_size = 16;

    sgx_status_t ret = get_encryption_key(model_id);
    if (ret != SGX_SUCCESS) {
        printf("error(%d) get_encryption_key.\n", ret);
        return ret;
    }

    ret = sgx_rijndael128GCM_decrypt((sgx_aes_gcm_128bit_key_t*)&g_model_keys[model_id],
                                     (uint8_t*)decrypt_handle, decrypt_bytes,
                                     (uint8_t*)decrypt_handle,
                                     (uint8_t*)iv_mac, gen_iv_size,
                                     NULL, 0,
                                     (sgx_aes_gcm_128bit_tag_t*)(iv_mac + gen_iv_size));
    //printf("ret %d/n", ret);
    return ret;
}


extern "C" sgx_status_t ecall_conv_dnnl_function (void* conv_desc, size_t conv_desc_size,
                                        void* conv_attr,  //TODO: change to in/out
                                        void* src_handle, size_t src_data_size,
                                        void* void_src_desc, size_t src_desc_size,
                                        void* weight_handle, size_t weight_data_size,
                                        size_t with_bias,
                                        void* bias, size_t bias_data_size,
                                        void* dst, size_t dst_data_size,
					void* weight_iv_mac, size_t weight_meta_size,
					void* bias_iv_mac, size_t bias_meta_size,
					uint32_t model_id)
{
    dnnl::engine engine2(dnnl::engine::kind::cpu, 0);
    //printf("hyhy call conv in sgx\n");
    // Create dnnl::stream.
    dnnl::stream engine_stream(engine2);

    dnnl::convolution_forward::desc* conv_desc_mem = (dnnl::convolution_forward::desc*)conv_desc;
    dnnl::convolution_forward::desc* conv_desc_pri = (dnnl::convolution_forward::desc*)void_src_desc;

    dnnl::primitive_attr conv_attr_mem;
    dnnl::post_ops* ops = (dnnl::post_ops*)conv_attr;
    conv_attr_mem.set_post_ops(*ops);
    auto conv_pd = dnnl::convolution_forward::primitive_desc(*conv_desc_mem, engine2);
    auto conv_pd_pri = dnnl::convolution_forward::primitive_desc(*conv_desc_pri, engine2);

    auto conv_src_mem = dnnl::memory(conv_pd.src_desc(), engine2);
    auto conv_dst_mem = dnnl::memory(conv_pd.dst_desc(), engine2);
    auto conv_weights_mem = dnnl::memory(conv_pd.weights_desc(), engine2);
    auto conv_bias_mem = dnnl::memory(conv_pd.bias_desc(), engine2);
    auto scratchpad_mem = dnnl::memory(conv_pd.scratchpad_desc(), engine2);


    auto user_src_mem = dnnl::memory(conv_pd_pri.src_desc(), engine2, src_handle);
    auto user_weights_mem = dnnl::memory(conv_pd_pri.weights_desc(), engine2, weight_handle);
    auto user_bias_mem = with_bias ? dnnl::memory(conv_pd_pri.bias_desc(), engine2, bias) : dnnl::memory(conv_pd_pri.bias_desc(), engine2);
    auto user_dst_mem = dnnl::memory(conv_pd_pri.dst_desc(), engine2, dst);

    auto decrypt_weight_handle = user_weights_mem.get_data_handle();
    auto decrypt_weight_bytes = user_weights_mem.get_desc().get_size();
    auto ret = AESGCM_decrypt_tensor(decrypt_weight_handle, decrypt_weight_bytes, weight_iv_mac, weight_meta_size, model_id);
    if (ret != SGX_SUCCESS)
        return ret;

    if (with_bias) {
        auto decrypt_bias_handle = user_bias_mem.get_data_handle();
        auto decrypt_bias_bytes = user_bias_mem.get_desc().get_size();
        ret = AESGCM_decrypt_tensor(decrypt_bias_handle, decrypt_bias_bytes, bias_iv_mac, bias_meta_size, model_id);
        if (ret != SGX_SUCCESS)
            return ret;
    }

    //create src with src handle
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        dnnl::reorder(user_src_mem, conv_src_mem)
                .execute(engine_stream, user_src_mem, conv_src_mem);
    }
    else
        conv_src_mem = user_src_mem;
    
    //create weight with weights handle
    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = dnnl::memory(conv_pd.weights_desc(), engine2);
        dnnl::reorder(user_weights_mem, conv_weights_mem)
                .execute(engine_stream, user_weights_mem, conv_weights_mem);
    }
    else
        conv_weights_mem = user_weights_mem;

    //create bias with bias handle
    if (with_bias) {
        if (conv_pd.bias_desc() != user_bias_mem.get_desc()) {
            dnnl::reorder(user_bias_mem, conv_bias_mem)
                    .execute(engine_stream, user_bias_mem, conv_bias_mem);
        }
        else {
            conv_bias_mem = dnnl::memory(conv_pd.bias_desc(), engine2, (void*)bias);
        }
    }

    auto conv_prim = dnnl::convolution_forward(conv_pd);

    if (with_bias) {
        conv_prim.execute(engine_stream,
                        {{DNNL_ARG_SRC, conv_src_mem},
                         {DNNL_ARG_WEIGHTS, conv_weights_mem},
                         {DNNL_ARG_BIAS, conv_bias_mem},
                         {DNNL_ARG_DST, user_dst_mem},
                         {DNNL_ARG_SCRATCHPAD, scratchpad_mem}
                         });
        engine_stream.wait();
    }
    else {
        conv_prim.execute(engine_stream,
                        {{DNNL_ARG_SRC, conv_src_mem},
                         {DNNL_ARG_WEIGHTS, conv_weights_mem},
                         {DNNL_ARG_DST, user_dst_mem},
                         {DNNL_ARG_SCRATCHPAD, scratchpad_mem}
                         });
        engine_stream.wait();
    }

    if (user_dst_mem.get_desc() != conv_dst_mem.get_desc()) {
            printf("in reorder, dst desc not the same");
            dnnl::reorder(user_dst_mem, conv_dst_mem)
                .execute(engine_stream, user_dst_mem, conv_dst_mem);
    }
    else
        conv_dst_mem = user_dst_mem;

    //TODO: check the size of dst whether equals to input size
    dst = conv_dst_mem.get_data_handle();

    return SGX_SUCCESS;
}


extern "C" sgx_status_t ecall_inner_product_dnnl_function (void* inner_product_desc, size_t inner_product_desc_size,
                                        void* src_handle, size_t src_data_size,
                                        void* inner_product_pri_desc, size_t inner_product_pri_size,
                                        void* weight_handle, size_t weight_data_size,
                                        size_t with_bias,
                                        void* bias, size_t bias_data_size,
                                        void* dst, size_t dst_data_size,
                                        void* weight_iv_mac, size_t weight_meta_size,
                                        void* bias_iv_mac, size_t bias_meta_size,
					uint32_t model_id)
{

    dnnl::engine engine(dnnl::engine::kind::cpu, 0);

    //printf("###############1");
    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    dnnl::inner_product_forward::desc* inner_product_desc_mem = (dnnl::inner_product_forward::desc*)inner_product_desc;
    dnnl::inner_product_forward::desc* inner_product_desc_pri = (dnnl::inner_product_forward::desc*)inner_product_pri_desc;

    auto inner_product_pd
            = dnnl::inner_product_forward::primitive_desc(*inner_product_desc_mem, engine);
    auto inner_product_pd_pri
            = dnnl::inner_product_forward::primitive_desc(*inner_product_desc_pri, engine);

    auto inner_product_src_mem = dnnl::memory(inner_product_pd.src_desc(), engine);
    auto inner_product_dst_mem = dnnl::memory(inner_product_pd.dst_desc(), engine, dst);
    auto inner_product_weights_mem = dnnl::memory(inner_product_pd.weights_desc(), engine);
    auto inner_product_bias_mem = dnnl::memory(inner_product_pd.bias_desc(), engine);


    auto user_src_mem = dnnl::memory(inner_product_pd_pri.src_desc(), engine, src_handle);
    auto user_weights_mem = dnnl::memory(inner_product_pd_pri.weights_desc(), engine, weight_handle);
    auto user_bias_mem = dnnl::memory(inner_product_pd_pri.bias_desc(), engine, bias);

    size_t bytes;
    
    //printf("################2");
    auto decrypt_weight_handle = user_weights_mem.get_data_handle();
    auto decrypt_weight_bytes = user_weights_mem.get_desc().get_size();
    auto ret = AESGCM_decrypt_tensor(decrypt_weight_handle, decrypt_weight_bytes, weight_iv_mac, weight_meta_size, model_id);
    if (ret != SGX_SUCCESS)
        return ret;
    if (with_bias) {
        auto decrypt_bias_handle = user_bias_mem.get_data_handle();
	auto decrypt_bias_bytes = user_bias_mem.get_desc().get_size();
        ret = AESGCM_decrypt_tensor(decrypt_bias_handle, decrypt_bias_bytes, bias_iv_mac, bias_meta_size, model_id);
	if (ret != SGX_SUCCESS)
            return ret;
    }
    //printf("##########3");

    //create src with src handle
    if (inner_product_pd.src_desc() != user_src_mem.get_desc()) {
        printf("src reorder in\n");
        dnnl::reorder(user_src_mem, inner_product_src_mem)
                .execute(engine_stream, user_src_mem, inner_product_src_mem);
    }
    else
        inner_product_src_mem = user_src_mem;

    bytes = inner_product_src_mem.get_desc().get_size();
    float *src_data = static_cast<float *>(inner_product_src_mem.get_data_handle());

    //create weight with weights handle
    if (inner_product_pd.weights_desc() != user_weights_mem.get_desc()) {
        printf("weights reorder in\n");
        inner_product_weights_mem = dnnl::memory(inner_product_pd.weights_desc(), engine);
        dnnl::reorder(user_weights_mem, inner_product_weights_mem)
                .execute(engine_stream, user_weights_mem, inner_product_weights_mem);
    }
    else
        inner_product_weights_mem = user_weights_mem;

    //create bias with bias handle
    if (inner_product_pd.bias_desc() != user_bias_mem.get_desc()) {
        printf("bias reorder in \n");
        dnnl::reorder(user_bias_mem, inner_product_bias_mem)
                .execute(engine_stream, user_bias_mem, inner_product_bias_mem);
    }
    else {
        inner_product_bias_mem = dnnl::memory(inner_product_pd.bias_desc(), engine, (void*)bias);
    }

    auto inner_product_prim = dnnl::inner_product_forward(inner_product_pd);


    inner_product_prim.execute(engine_stream,
                        {{DNNL_ARG_SRC, inner_product_src_mem},
                         {DNNL_ARG_WEIGHTS, inner_product_weights_mem},
                         {DNNL_ARG_BIAS, inner_product_bias_mem},
                         {DNNL_ARG_DST, inner_product_dst_mem}
                         });
    engine_stream.wait();

    return SGX_SUCCESS;
}

extern "C" sgx_status_t ecall_batch_norm_dnnl_function (void* batch_norm_desc, size_t batch_norm_desc_size, //desc of inner_product data
                                        void* src_handle, size_t src_data_size,
                                        void* var_handle, size_t var_data_size,
                                        void* mean_handle, size_t mean_data_size,
                                        void* scale_shift, size_t scale_shift_data_size,
                                        size_t scale_data_size, size_t shift_data_size,
                                        void* dst, size_t dst_data_size,
					void* scale_iv_mac, size_t scale_meta_size,
					void* shift_iv_mac, size_t shift_meta_size,
					void* mean_iv_mac, size_t mean_meta_size,
					void* var_iv_mac, size_t var_meta_size,
					uint32_t model_id)
{
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);

    //printf("call batch norm in sgx\n");
    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    dnnl::batch_normalization_forward::desc* batch_norm_desc_mem = (dnnl::batch_normalization_forward::desc*)batch_norm_desc;

    auto batch_norm_pd
            = dnnl::batch_normalization_forward::primitive_desc(*batch_norm_desc_mem, engine);

    auto batch_norm_src_mem = dnnl::memory(batch_norm_pd.src_desc(), engine, src_handle);
    auto batch_norm_dst_mem = dnnl::memory(batch_norm_pd.dst_desc(), engine, dst);
    auto batch_norm_var_mem = dnnl::memory(batch_norm_pd.variance_desc(), engine, var_handle);
    auto batch_norm_mean_mem = dnnl::memory(batch_norm_pd.mean_desc(), engine, mean_handle);
    auto batch_norm_scale_shift = dnnl::memory(batch_norm_pd.weights_desc(), engine, scale_shift);

    auto scale_handle = batch_norm_scale_shift.get_data_handle();
    auto ret = AESGCM_decrypt_tensor(scale_handle, scale_data_size, scale_iv_mac, scale_meta_size, model_id);
    if (ret != SGX_SUCCESS)
        return ret;
    ret = AESGCM_decrypt_tensor(scale_handle + scale_data_size, shift_data_size, shift_iv_mac, shift_meta_size, model_id);
    if (ret != SGX_SUCCESS)
        return ret;
    ret = AESGCM_decrypt_tensor(mean_handle, mean_data_size, mean_iv_mac, mean_meta_size, model_id);
    if (ret != SGX_SUCCESS)
        return ret;
    ret = AESGCM_decrypt_tensor(var_handle, var_data_size, var_iv_mac, var_meta_size, model_id);
    if (ret != SGX_SUCCESS)
        return ret;


    auto batch_norm_prim = dnnl::batch_normalization_forward(batch_norm_pd);

    batch_norm_prim.execute(engine_stream,
                      {{DNNL_ARG_SRC, batch_norm_src_mem},
                       {DNNL_ARG_SCALE_SHIFT, batch_norm_scale_shift},
                       {DNNL_ARG_VARIANCE, batch_norm_var_mem},
                       {DNNL_ARG_MEAN, batch_norm_mean_mem},
                       {DNNL_ARG_DST, batch_norm_dst_mem}});

    engine_stream.wait();

    return SGX_SUCCESS;
}


