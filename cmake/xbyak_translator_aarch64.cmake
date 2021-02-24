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

find_package(XED REQUIRED)
find_package(XBYAK_AARCH64 REQUIRED)
find_package(XBYAK_TRANSLATOR_AARCH64 REQUIRED)

if(XED_FOUND)
    list(APPEND EXTRA_SHARED_LIBS ${XED_LIBRARIES})
    include_directories(${XED_INCLUDE_DIRS})
    message(STATUS "Xed Library: ${XED_LIBRARIES}")
endif()

if(XBYAK_AARCH64_FOUND)
    list(APPEND EXTRA_STATIC_LIBS ${XBYAK_AARCH64_LIBRARIES})
    include_directories(${XBYAK_AARCH64_INCLUDE_DIRS})
    message(STATUS "Xbyak_aarch64 Library: ${XBYAK_AARCH64_LIBRARIES}")
endif()

if(XBYAK_TRANSLATOR_AARCH64_FOUND)
    list(APPEND EXTRA_STATIC_LIBS ${XBYAK_TRANSLATOR_AARCH64_LIBRARIES})
    include_directories(${XBYAK_TRANSLATOR_AARCH64_INCLUDE_DIRS})
    message(STATUS "Xbyak_translator_aarch64 Library: ${XBYAK_TRANSLATOR_AARCH64_LIBRARIES}")
    add_definitions(-DDNNL_AARCH64_USE_XBYAK_TRANSLATOR_AARCH64)
endif()
