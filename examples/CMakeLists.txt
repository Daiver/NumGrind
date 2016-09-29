project(Examples)

#add_subdirectory(../NumGrind)
include_directories(../NumGrind)
add_executable(examples
        main.cpp
        mnist.cpp mnist.h
        )
target_link_libraries(examples numgrind)
