cmake_minimum_required(VERSION 3.0)
project(NumGrind)

#find_package(PkgConfig)
#pkg_search_module(Eigen3 REQUIRED eigen3)
#find_package(Eigen3 REQUIRED)
#set(CMAKE_CXX_FLAGS "-O3  -msse4 -fopenmp")
#set(CMAKE_CXX_FLAGS "-O3  -msse2 -fopenmp")
#set(CMAKE_CXX_FLAGS "-O3  -msse4 ")
#set(CMAKE_CXX_FLAGS "-O3  -msse2 ")
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")


set(EIGEN3_INCLUDE_DIR 3rdparty)
include_directories(${EIGEN3_INCLUDE_DIR})

#include_directories("./3rdparty")
add_subdirectory(NumGrind)
add_subdirectory(examples)
add_subdirectory(tests)

