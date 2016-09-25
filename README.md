# NumGrind
Simple computational graph with reverse mode autodiff framework. Inspired by TensorFlow/Theano and other computational graph tools

Not for production

Pull-requests are welcomed

#TODO

#Common
 - Make work with variables values more explicit
 - Split NumGrind and utility code (main.cpp/utils.h/etc)
 - Add NumGrin namespace
 - Add nice interface for GraphNodes
 - Switch from raw pointers to smart pointers
 - Implement toString() for all nodes
 - Improve string formating
 - Refactor all nodes
 - Remove copy-paste from nodes. Add intermediate classes for binary operators

##Performance
 - Create big problem for performance tests
 - Sdd pre-allocated arrays for intermediate results
 - (long term issue) Do not compute backwardPass for Constants
 - (long term issue) Split mutable part of the graph (caches) into separate structure

##Tests
 - Split tests into several files
 - Change tests names

##Solved
- Nothing
