cmake_minimum_required(VERSION 3.0)
project(NumGrind)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

#include_directories("./3rdparty")
add_subdirectory(src)
add_subdirectory(tests)

