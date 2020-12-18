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

#===============================================================================
# CMake - Cross Platform Makefile Generator
# Copyright 2000-2020 Kitware, Inc. and Contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of Kitware, Inc. nor the names of Contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ------------------------------------------------------------------------------

# ----------
# FindXED
# ----------
#
# Finds Intel X86 Encoder Decoder (Intel XED)
# https://github.com/intelxed/xed
#
# This module defines the following variables:
#
#   XED_FOUND          - True if Intel XED was found
#   XED_INCLUDE_DIRS   - include directories for Intel XED
#   XED_LIBRARIES      - link against this library to use Intel XED
#
# The module will also define two cache variables:
#
#   XED_INCLUDE_DIR    - the the Intel XED include directory
#   XED_LIBRARY        - the path to the Intel XED library
#

# Use XED_ROOT_DIR environment variable to find the library and headers
find_path(XED_DIR
  NAMES include/xed/xed-init.h
  PATHS "src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/build_xed_aarch64/kits/xed"
  NO_DEFAULT_PATH
  )

get_filename_component(XED_INCLUDE_DIR "${XED_DIR}/include" ABSOLUTE)

find_library(XED_LIBRARY
  NAMES xed
  PATHS "src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/build_xed_aarch64/kits/xed"
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XED DEFAULT_MSG
  XED_INCLUDE_DIR
  XED_LIBRARY
)

mark_as_advanced(
  XED_LIBRARY
  XED_INCLUDE_DIR
  )

# Find the extra libraries and include dirs
if(XED_FOUND)
  list(APPEND XED_INCLUDE_DIRS ${XED_INCLUDE_DIR})
  list(APPEND XED_LIBRARIES ${XED_LIBRARY})
endif()


