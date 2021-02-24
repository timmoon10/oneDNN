#! /bin/bash
#===============================================================================
# Copyright 2020 FUJITSU LIMITED
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
# limitations under the License.
#===============================================================================
#*******************************************************************************
# Function definition
#*******************************************************************************
usage_exit() {
    echo "Usage: $0 (-n|-q)"
    echo "build libxed"
    echo "  -n build libxed for native environment"
    echo "  -q build libxed for qemu-aarch64 environment"
    exit 1
}

# Build xed
DIR_BUILD=src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/build_xed_aarch64
OPT_CHECK=0
USAGE_DONE=0
mkdir -p ${DIR_BUILD}
cd ${DIR_BUILD}

while getopts nqhH OPT
do
    case $OPT in
        q)
            ../xed/mfile.py \
                --strip=/usr/bin/aarch64-linux-gnu-strip  \
                --cc=/usr/bin/aarch64-linux-gnu-gcc \
                --cxx=/usr/bin/aarch64-linux-gnu-g++ \
                --host-cpu=aarch64 \
                --shared examples install
            OPT_CHECK=1
            ;;
        n)
            ../xed/mfile.py \
            --shared examples install
            OPT_CHECK=1
            ;;
        h|H)
            usage_exit
            USAGE_DONE=1
            exit 1
            ;;
   esac
done
shift $((OPTIND - 1))

if [ ${OPT_CHECK} != 1 ] ; then
    if [ ${USAGE_DONE} != 1 ] ; then
        usage_exit
    fi
    exit 1
fi

cd kits
ln -sf xed-install* xed
