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

if(xed_cmake_included)
  return()
endif()
set(xed_cmake_included true)
include("cmake/options.cmake")

if(NOT DNNL_TARGET_ARCH STREQUAL "AARCH64")
  return()
endif()

find_package(XED)

if(NOT XED_FOUND)
  message(FATAL_ERROR, "libxed not found!")
endif()

if(XED_FOUND)
  list(APPEND EXTRA_SHARED_LIBS ${XED_LIBRARIES})

  include_directories(${XED_INCLUDE_DIRS})

  message(STATUS "xed libraries: ${XED_LIBRARIES}")
  message(STATUS "xed headers: ${XED_INCLUDE_DIRS}")
endif()

