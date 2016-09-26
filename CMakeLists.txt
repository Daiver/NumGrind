cmake_minimum_required(VERSION 3.0)
project(NumGrind)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#include_directories("./3rdparty")
add_subdirectory(src)
add_subdirectory(tests)

