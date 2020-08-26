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
# *******************************************************************************/
git submodule sync --recursive
git submodule update --init --recursive

cd src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/
mkdir build_xed_aarch64
cd build_xed_aarch64/
../xed/mfile.py --strip=/usr/bin/aarch64-linux-gnu-strip  --cc=/usr/bin/aarch64-linux-gnu-gcc --cxx=/usr/bin/aarch64-linux-gnu-g++ --host-cpu=aarch64 --shared examples install
cd kits/
XED=`ls | grep install`
ln -sf $XED xed
cd xed/bin/
CI_XED_PATH=`pwd`
cd ../../../../../
source dot.zshrc.xbyak
cd ../../../../../
mkdir build
cd build
export LD_LIBRARY_PATH=/github/workspace/src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/build_xed_aarch64/kits/xed/lib:${LD_LIBRARY_PATH} && \
export XED_ROOT_DIR=/github/workspace/src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/build_xed_aarch64/kits/xed && \
cmake -DCMAKE_BUILD_TYPE=Debug -DDNNL_INDIRECT_JIT_AARCH64=ON -DDNNL_TARGET_ARCH=AARCH64 -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=AARCH64 -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++  -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu ..
make -j2
cd /
./gtest_all.sh


