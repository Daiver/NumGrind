cmake_minimum_required(VERSION 3.0)

set(REPO ../)

project(Test)
project(Example)

include(CTest)
enable_testing()

#set(gtest_disable_pthreads on) #needed in MinGW
include(${REPO}/3rdparty/DownloadProject/DownloadProject.cmake)
download_project(
        PROJ                googletest
        GIT_REPOSITORY      https://github.com/google/googletest.git
        GIT_TAG             master
        UPDATE_DISCONNECTED 1
        )

add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
include_directories("../3rdparty")
include_directories("../NumGrind/")
#include_directories("../src/CompGraph/")
set(test_SRCS
        helpers.cpp helpers.h
        compgraphtests.cpp compgraphtests.h
        symbolicgraphtests.cpp symbolicgraphtests.h
        gradientchecktests.cpp gradientchecktests.h)

add_executable(runUnitTests gtest.cpp ${test_SRCS})
target_link_libraries(runUnitTests gtest gmock numgrind)
add_test(runUnitTests runUnitTests) #included in all tutorials but I don't know what it actually does
