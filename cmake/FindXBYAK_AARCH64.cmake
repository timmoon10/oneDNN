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

# ----------
# FindXbyak_aarch64
# ----------
#
# This module defines the following variables:
#
#   XBYAK_AARCH64_FOUND          - True if Xbyak_aarch64 was found
#   XBYAK_AARCH64_INCLUDE_DIRS   - include directories for Xbyak_aarch64
#   XBYAK_AARCH64_LIBRARIES      - link against this library to use Xbyak_aarch64

find_path(XBYAK_AARCH64_DIR
  NAMES xbyak_aarch64/xbyak_aarch64.h
  PATHS "src/cpu/aarch64"
  NO_DEFAULT_PATH
  )

get_filename_component(XBYAK_AARCH64_INCLUDE_DIR "${XBYAK_AARCH64_DIR}" ABSOLUTE)

find_library(XBYAK_AARCH64_LIBRARY
  NAMES xbyak_aarch64
  PATHS "src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/xbyak_aarch64"
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XBYAK_AARCH64 DEFAULT_MSG
  XBYAK_AARCH64_INCLUDE_DIR
  XBYAK_AARCH64_LIBRARY
)

if(XBYAK_AARCH64_FOUND)
  list(APPEND XBYAK_AARCH64_INCLUDE_DIRS ${XBYAK_AARCH64_INCLUDE_DIR})
  list(APPEND XBYAK_AARCH64_LIBRARIES ${XBYAK_AARCH64_LIBRARY})
endif()


