import torch
import time
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import ctypes
from ctypes import *
import struct
import os

LIB_PATH = torch.__path__[0] + "/../enclave_ops/secure_op/build/libsecure_conv.so"

class SecureMkldnnLinear(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype):
        super(SecureMkldnnLinear, self).__init__()
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))
        if dense_module.bias is not None:
            # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
            # we use fp32 dtype.
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            # TODO: Remove this once ScriptModule supports registering None buffer
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())
        self.weight =  dense_module.weight
        self.bias = dense_module.bias
        if dense_module.bias is None:
            self.bias = torch.zeros([dense_module.weight.size(0)], dtype=torch.float)
        self.encrypted_weight = _SecureMkldnnConvNd.construct_tensor(self.weight).detach()
        self.encrypted_bias = _SecureMkldnnConvNd.construct_tensor(self.bias)
        self.weight.requires_grad_(False).zero_()
        self.bias.requires_grad_(False).zero_()

    @torch.jit.script_method
    def __getstate__(self):
        #return (self.weight.to_dense(), self.bias.to_dense(), self.training)
        return (self.encrypted_weight, self.encrypted_bias)

    @torch.jit.script_method
    def __setstate__(self, state):
        #self.weight = state[0].to_mkldnn()
        self.bias = state[1]
        #self.training = state[2]

    #@torch.jit.script_method
    def forward(self, x):
        torch.ops.load_library(LIB_PATH)
        return torch.ops.my_ops.secure_linear(
            x,
            self.encrypted_weight,
            self.encrypted_bias)
        #x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
        #y_mkldnn = torch._C._nn.mkldnn_linear(x_mkldnn, self.weight, self.bias)
        #y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
        #return y


class _SecureMkldnnConvNd(torch.jit.ScriptModule):
    """Common base of MkldnnConv1d and MkldnnConv2d"""
    __constants__ = ['stride', 'padding', 'dilation', 'groups']

    def __init__(self, dense_module):
        super(_SecureMkldnnConvNd, self).__init__()

        self.weight = dense_module.weight
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups
        self.output_padding = dense_module.output_padding
        self.bias = dense_module.bias
        self.encrypted_weight = _SecureMkldnnConvNd.construct_tensor(self.weight).detach()
        if dense_module.bias is None:
            self.bias = torch.zeros([dense_module.weight.size(0)], dtype=torch.float)
        self.encrypted_bias = _SecureMkldnnConvNd.construct_tensor(self.bias)
        self.weight.requires_grad_(False).zero_()
        self.bias.zero_()

    '''
        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
            # we use fp32 dtype.
            # TODO: Remove this once ScriptModule supports registering None buffer
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())
    '''

#    @torch.jit.script_method
#    def __getstate__(self):
#        return (_SecureMkldnnConvNd.construct_tensor(self.weight), self.bias, self.training)

    #TODO:bias need encrypt

    def construct_tensor(tensor, key = bytes(16), model_id = bytes(4)):
        assert isinstance(key, bytes) and len(key) == 16, "Please input bytes(16) as the key"
        assert isinstance(model_id, bytes) and len(model_id) == 4, "Please input bytes(4) as the model_id"
        assert torch.is_tensor(tensor), "Please input torch.tensor"
        mac = bytes(16)
        iv = os.urandom(12)
        aad = b''
        num_in_int32 = int(1 + tensor.dim() + tensor.numel() + (100 + len(iv) + len(mac))/4)
        t = torch.zeros(num_in_int32, dtype=torch.int32)
        t[0] = tensor.dim()
        for i in range (tensor.dim()):
            t[i+1] = tensor.shape[i]
        byte_offset = 4 * (1 + tensor.dim())
        if tensor.dtype == torch.float32:
            dtype_len = 4
            ctypes.memmove(t.storage().data_ptr() + byte_offset, (0).to_bytes(4, 'little'), 4)
        elif tensor.dtype == torch.float16:
            dtype_len = 2
            ctypes.memmove(t.storage().data_ptr() + byte_offset, (1).to_bytes(4, 'little'), 4)
        else:
            assert False, "Only Support float32 or float16"
        ctypes.memmove(t.storage().data_ptr() + byte_offset + 4, model_id, 4)
        byte_offset += 100    
        ctypes.memmove(t.storage().data_ptr() + byte_offset, iv, len(iv))
        byte_offset += len(iv)
        plaintext = ciphertext = bytes(dtype_len * tensor.numel())
        ctypes.memmove(plaintext, tensor.storage().data_ptr(), dtype_len * tensor.numel())
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        encryptor.authenticate_additional_data(aad)
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        mac = encryptor.tag
        ctypes.memmove(t.storage().data_ptr() + byte_offset, mac, len(mac))
        byte_offset += len(mac)
        ctypes.memmove(t.storage().data_ptr() + byte_offset, ciphertext, len(ciphertext))
        return t


    @torch.jit.script_method
    def __getstate__(self):
        return (self.encrypted_weight, self.encrypted_bias)


#    @torch.jit.script_method
    def forward(self, x):
        torch.ops.load_library(LIB_PATH) 
        return torch.ops.my_ops.secure_conv(
            x,
            self.encrypted_weight,
            self.encrypted_bias,
            self.stride,
            self.padding,
            self.dilation,
            False,
            self.output_padding,
            self.groups)


class SecureMkldnnConv1d(_SecureMkldnnConvNd):
    def __init__(self, dense_module, dtype):
        super(SecureMkldnnConv1d, self).__init__(dense_module)

#        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0]
        self.bias = state[1]
#        self.training = state[2]


class SecureMkldnnConv2d(_SecureMkldnnConvNd):
    def __init__(self, dense_module, dtype):
        super(SecureMkldnnConv2d, self).__init__(dense_module)

        pass
    '''        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv2d_weight(
            dense_module.weight.to_mkldnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))
    '''
    @torch.jit.script_method
    def __setstate__(self, state):
        print(state[1])
#        self.weight = state[0]
        self.bias = state[1]
#        self.training = state[2]
    '''
    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.mkldnn_reorder_conv2d_weight(
            state[0].to_mkldnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]'''

class SecureMkldnnConv3d(_SecureMkldnnConvNd):
    def __init__(self, dense_module, dtype):
        super(SecureMkldnnConv3d, self).__init__(dense_module)

        pass

    '''        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv3d_weight(
            dense_module.weight.to_mkldnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))
    '''
    
    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.mkldnn_reorder_conv3d_weight(
            state[0].to_mkldnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]


class SecureMkldnnBatchNorm(torch.jit.ScriptModule):
    __constants__ = ['exponential_average_factor', 'eps']

    def __init__(self, dense_module):
        super(SecureMkldnnBatchNorm, self).__init__()

        assert(not dense_module.training)
        assert(dense_module.track_running_stats)
        assert(dense_module.affine)

        if dense_module.momentum is None:
            self.exponential_average_factor = 0.0
        else:
            self.exponential_average_factor = dense_module.momentum
        self.eps = dense_module.eps

        #self.register_buffer('weight', dense_module.weight.to_mkldnn())
        #self.register_buffer('bias', dense_module.bias.to_mkldnn())
        #self.register_buffer('running_mean', dense_module.running_mean.to_mkldnn())
        #self.register_buffer('running_var', dense_module.running_var.to_mkldnn())
        self.weight = dense_module.weight
        self.bias = dense_module.bias
        self.running_mean = dense_module.running_mean
        self.running_var = dense_module.running_var
        self.encrypted_weight = _SecureMkldnnConvNd.construct_tensor(self.weight).detach()
        self.encrypted_bias = _SecureMkldnnConvNd.construct_tensor(self.bias)
        self.weight.requires_grad_(False).zero_()
        self.bias.requires_grad_(False).zero_()

    @torch.jit.script_method
    def __getstate__(self):
        #weight = self.weight.to_dense()
        #bias = self.bias.to_dense()
        #running_mean = self.running_mean.to_dense()
        #running_var = self.running_var.to_dense()
        #return (weight, bias, running_mean, running_var, self.training)
        return (self.encrypted_weight, self.encrypted_bias, self.running_mean, self.running_var)

    @torch.jit.script_method
    def __setstate__(self, state):
        pass
        #self.weight = state[0].to_mkldnn()
        #self.bias = state[1].to_mkldnn()
        #self.running_mean = state[2].to_mkldnn()
        #self.running_var = state[3].to_mkldnn()
        #self.training = state[4]

    #@torch.jit.script_method
    def forward(self, x):
        torch.ops.load_library(LIB_PATH)
        return torch.ops.my_ops.secure_batch_norm(
            x,
            self.encrypted_weight,
            self.encrypted_bias,
            self.running_mean,
            self.running_var,
            self.exponential_average_factor,
            self.eps)


def to_secure_mkldnn(module, dtype=torch.float):
    assert dtype in [torch.float, torch.bfloat16], \
        "MKLDNN only support float or bfloat16 path now"

    def m_fn(m, d):
        if isinstance(m, torch.nn.Linear):
            return SecureMkldnnLinear(m, d)
        elif isinstance(m, torch.nn.Conv1d):
            return SecureMkldnnConv1d(m, d)
        elif isinstance(m, torch.nn.Conv2d):
            return SecureMkldnnConv2d(m, d)
        elif isinstance(m, torch.nn.Conv3d):
            return SecureMkldnnConv3d(m, d)
        elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
            # For batchnorm bf16 path, OneDNN requires weight and bias need fp32 dtype.
            # so it doesn't need dtype argument.
            return SecureMkldnnBatchNorm(m)
        else:
            return m

    def m_fn_rec(m, d):
        new_m = m_fn(m, d)
        for name, sub_m in m.named_children():
            setattr(new_m, name, m_fn_rec(sub_m, d))
        return new_m

    return m_fn_rec(module, dtype)
