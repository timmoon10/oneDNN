#!/bin/bash
#*******************************************************************************
# Copyright 2019 FUJITSU LIMITED
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# imitations under the License.
# *******************************************************************************/CMakeFiles/
cd tests/gtests

/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_batch_normalization_f32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_batch_normalization_s8
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_binary
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_concat
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_backward_data_f32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_backward_weights_f32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_eltwise_forward_f32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_eltwise_forward_x8s8f32s32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_format_any
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_forward_f32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_forward_u8s8fp
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_forward_u8s8s32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_cross_engine_reorder
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_deconvolution
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_dnnl_threading
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_eltwise
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_bf16bf16bf16
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_bf16bf16f32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_f16
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_f16f16f32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_f32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_s8s8s32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_s8u8s32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_u8s8s32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_u8u8s32
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_global_scratchpad
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_attr
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_handle
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_pd
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_pd_iter
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_primitive_cache
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_runtime_attr
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_runtime_dims
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_stream_attr
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_wino_convolution
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_inner_product_backward_data
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_inner_product_backward_weights
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_inner_product_forward
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_layer_normalization
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_logsoftmax
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_lrn_backward
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_lrn_forward
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_matmul
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_pooling_backward
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_pooling_forward
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_reorder
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_resampling
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_rnn_forward
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_shuffle
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_softmax
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_sum
