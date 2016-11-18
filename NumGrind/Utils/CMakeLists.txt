cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})

set(NUMGRIND_UTILS_FILES
        utils.h utils.cpp
        eigenimport.h
        eigenimport.cpp
        randomutils.h
        randomutils.cpp
        Normalizer.h
        Normalizer.cpp
        )

add_library(Utils ${NUMGRIND_UTILS_FILES})
