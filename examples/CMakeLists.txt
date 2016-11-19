project(Examples)

include_directories(../NumGrind)
include_directories(..)

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

add_executable(houseprice_example houseprice_example.cpp)
target_link_libraries(houseprice_example numgrind)

add_executable(cppoptlib_example cppoptlib_example.cpp)
target_link_libraries(cppoptlib_example numgrind)