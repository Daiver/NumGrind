cmake_minimum_required(VERSION 3.0)

include_directories(${EIGEN3_INCLUDE_DIR})

set(NUMGRIND_DEEPGRIND_FILES
        deepgrind.cpp deepgrind.h
        ActivationFunctions.h ActivationFunctions.cpp
        Utils.cpp Utils.h
        CGConv2DNode.cpp CGConv2DNode.h
        Conv2DParams.cpp Conv2DParams.h)

add_library(DeepGrind ${NUMGRIND_DEEPGRIND_FILES})
target_link_libraries(DeepGrind NumGrindSymbolicGraph NumGrindCompGraph)
