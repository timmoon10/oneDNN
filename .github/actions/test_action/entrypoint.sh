#!/bin/bash
#*******************************************************************************
# Copyright 2019-2020 FUJITSU LIMITED
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
# *******************************************************************************/
export DIR_ROOT=`pwd`

# Build QEMU
# QEMU must be build by native compiler
echo "##################################################"
echo "# Download and build QEMU"
echo "# Wait for a few minutes"
echo "##################################################"
.github/automation/env/qemu.sh &> /dev/null
qemu-aarch64 --version

# Set compiler
source .github/automation/env/setenv-gcc-qemu

echo "##################################################"
echo "# Download git submodules"
echo "##################################################"
git submodule sync --recursive
git submodule update --init --recursive

# Build libxed
echo "##################################################"
echo "# Build Intel XED"
echo "# Wait for a few minutes"
echo "##################################################"
cd src/cpu/aarch64/xbyak_translator_aarch64
${DIR_ROOT}/.github/automation/env/xed.sh -q > /dev/null
cd ${DIR_ROOT}


# Build oneDNN
echo "##################################################"
echo "# Build oneDNN"
echo "# Wait for a few minutes"
echo "##################################################"
.github/automation/build.sh --threading omp --mode Release --source-dir $(pwd) --build-dir $(pwd)/build --cmake-opt "-DDNNL_INDIRECT_JIT_AARCH64=ON -DDNNL_TARGET_ARCH=AARCH64 -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=AARCH64 -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu -DDNNL_TARGET_EMULATOR=qemu-aarch64"


# Teste oneDNN
echo "##################################################"
echo "# Test oneDNN"
echo "# Wait for a few minutes"
echo "##################################################"
.github/automation/test.sh --test-kind gtest --build-dir $(pwd)/build --report-dir $(pwd)/report
