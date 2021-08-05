cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(secure_conv)

find_package(Torch REQUIRED)

set(TORCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)

if (NOT DEFINED ENV{SGX_SDK})
  message(FATAL_ERROR "SGX_SDK is not defined. DiD you source sgx environment?")
endif()


include_directories("${TORCH_ROOT}/aten/src/")
include_directories("${TORCH_ROOT}/torch/include")
include_directories("${TORCH_ROOT}/third_party/ideep/include")
include_directories("${TORCH_ROOT}/enclave_ops/ideep-enclave/App")
include_directories("$ENV{SGX_SDK}/include")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${TORCH_ROOT}/enclave_ops/ideep-enclave/App/Enclave_u.o $ENV{SGX_SDK}/lib64/libsgx_urts.so ")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${TORCH_ROOT}/build/lib/libdnnl.a")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lsgx_urts")
set(CMAKE_CXX_FLAGS "-O0 -Wall -g -ggdb ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENCLAVE_PATH=\\\"${TORCH_ROOT}/enclave_ops/ideep-enclave/libenclave.signed.so\\\"")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TORCH_ROOT}/enclave_ops/ideep-enclave/App/Enclave_u.o")

add_library(secure_conv SHARED op.cpp)
target_compile_features(secure_conv PRIVATE cxx_std_14)
target_link_libraries(secure_conv "${TORCH_LIBRARIES}")
target_link_libraries(torch_cpu) 

