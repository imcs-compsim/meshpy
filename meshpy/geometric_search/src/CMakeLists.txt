# The MIT License (MIT)
#
# Copyright (c) 2018-2025 MeshPy Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.0)

# Project name
project(arborxpy)

# Find python packages
find_package(Python COMPONENTS Interpreter Development)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Add the local pybind11 directory
add_subdirectory(../../../external/pybind11 build)
find_package(pybind11 CONFIG)
include_directories(../../../external/pybind11/include)

# Get ArborX from github
include(FetchContent)
fetchcontent_declare(
  arborx
  GIT_REPOSITORY https://github.com/arborx/ArborX.git
  GIT_TAG 4c015c70f3927b55e730515ce8db235e68c13480
  )
fetchcontent_makeavailable(arborx)

# Compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -fvisibility=hidden")

# Setup arborxpy
set(module_name "geometric_search_arborx_lib")
set(target_name "${module_name}")
add_definitions("-DPYTHON_MODULE_NAME")
set(PYTHON_MODULE_NAME "${target_name}")
add_library(
  ${target_name}
  MODULE
  geometric_search.cpp
  find_close_points.cpp
)
set_target_properties(
  ${target_name}
  PROPERTIES
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
target_link_libraries(
  ${target_name}
  PUBLIC
  ArborX::ArborX
)
add_custom_command(
  TARGET ${target_name}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${target_name}> ${CMAKE_SOURCE_DIR}/../
)
