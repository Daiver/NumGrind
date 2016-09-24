# NumGrind
Simple computational graph with reverse mode autodiff framework

Not for production

#TODO

#Common
 - Split NumGrind and utility code (main.cpp/utils.h/etc)
 - Add NumGrin namespace
 - Add nice interface for GraphNodes
 - Switch from raw pointers to smart pointers

##Performance
 - Sdd pre-allocated arrays for intermediate results
 - (long term issue) Do not compute backwardPass for Constants

##Tests
 - Split tests into several files
 - Change tests names

##Solved
- Nothing
