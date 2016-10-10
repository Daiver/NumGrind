project(Examples)

#add_subdirectory(../NumGrind)
include_directories(../NumGrind)
add_executable(examples
        main.cpp
        mnist.cpp mnist.h)
target_link_libraries(examples numgrind)

add_executable(operators_example
        operators_example.cpp
        mnist.cpp mnist.h)
target_link_libraries(operators_example numgrind)

add_executable(mnist_example
        mnist_example.cpp
        mnist.cpp mnist.h)
target_link_libraries(mnist_example numgrind)
