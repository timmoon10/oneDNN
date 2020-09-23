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
cd /github/workspace/build/tests/gtests

/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_batch_normalization_f32 | tail -n 1 > check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_batch_normalization_s8 | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_binary | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_concat | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_backward_data_f32 | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_backward_weights_f32 | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_eltwise_forward_f32 | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_eltwise_forward_x8s8f32s32 | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_format_any | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_forward_f32 | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_forward_u8s8fp | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_convolution_forward_u8s8s32 | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_cross_engine_reorder | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_deconvolution | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_dnnl_threading | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_eltwise | tail -n 1 >> check.log
#/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_bf16bf16bf16 | tail -n 1 >> check.log
#/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_bf16bf16f32 | tail -n 1 >> check.log
#/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_f16 | tail -n 1 >> check.log
#/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_f16f16f32 | tail -n 1 >> check.log
#/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_f32 | tail -n 1 >> check.log
#/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_s8s8s32 | tail -n 1 >> check.log
#/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_s8u8s32 | tail -n 1 >> check.log
#/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_u8s8s32 | tail -n 1 >> check.log
#/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_gemm_u8u8s32 | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_global_scratchpad | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_attr | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_handle | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_pd | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_pd_iter | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_primitive_cache | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_runtime_attr | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_runtime_dims | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_stream_attr | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_iface_wino_convolution | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_inner_product_backward_data | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_inner_product_backward_weights | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_inner_product_forward | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_layer_normalization | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_logsoftmax | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_lrn_backward | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_lrn_forward | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_matmul | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_pooling_backward | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_pooling_forward | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_reorder | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_resampling | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_rnn_forward | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_shuffle | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_softmax | tail -n 1 >> check.log
/local_qemu_5.0.0/bin/qemu-aarch64 -cpu max,sve512=on ./test_sum | tail -n 1 >> check.log

NUM_TP=`wc -l check.log | cut -f 1 -d " "`
NUM_OK=`grep PASSED check.log | wc -l | cut -f 1 -d " "`

echo "TP NUM: ${NUM_TP}"
echo "TP OK : ${NUM_OK}"

if [ ${NUM_TP} = ${NUM_OK} ] ; then
    echo "Congratulation!"
    exit 0
else
    echo "Something wrong!"
    exit 1
fi
