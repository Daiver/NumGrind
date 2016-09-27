project(Examples)

#add_subdirectory(../NumGrind)
include_directories(../NumGrind)
add_executable(examples main.cpp )
target_link_libraries(examples core)
