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
  GIT_TAG aefe78fd6e8126a17e3b73998377aa8c205a13dd #v1.3
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
