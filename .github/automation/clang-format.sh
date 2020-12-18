#! /bin/bash

#===============================================================================
# Copyright 2019-2020 Intel Corporation
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

echo "Using clang-format version: $(clang-format --version)"
echo "Starting format check..."

TMPFILE=$(mktemp)
find "$(pwd)" -type f | grep -P ".*\.(c|cpp|h|hpp|cl)$" > ${TMPFILE}
NUM_LINE=`wc -l ${TMPFILE} | cut -f 1 -d " "`
TOTAL_I=0

if [ "$(uname)" == "Linux" ]; then
    NUM_CPU="$(grep -c processor /proc/cpuinfo)"
else
    NUM_CPU="$(sysctl -n hw.physicalcpu)"
fi

# Run clang-format in parallel
while [ ${TOTAL_I} -lt ${NUM_LINE} ]
do
    LOCAL_I=0
    ARRAY=()
    while [ ${LOCAL_I} -lt ${NUM_CPU} ]
    do
# Debug
# echo "clang-format `sed -n $((${TOTAL_I}+1))p ${TMPFILE}`"
        nohup clang-format -i -style=file `sed -n $((${TOTAL_I}+1))p ${TMPFILE}` &> /dev/null
        ARRAY+=($!)
        TOTAL_I=$((${TOTAL_I}+1))
        LOCAL_I=$((${LOCAL_I}+1))

        if [ ${TOTAL_I} -ge ${NUM_LINE} ] ; then
            break;
        fi
    done
    wait ${ARRAY[@]}
done

RETURN_CODE=0
echo $(git status) | grep "nothing to commit" > /dev/null

if [ $? -eq 1 ]; then
    echo "Clang-format check FAILED! Found not formatted files!"
    echo "$(git status)"
    RETURN_CODE=3
else
    echo "Clang-format check PASSED! Not formatted files not found..."
fi

exit ${RETURN_CODE}
