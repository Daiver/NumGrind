# NumGrind
Simple computational graph with reverse mode autodiff framework. Inspired by TensorFlow/Theano and other computational graph tools

[![Build Status](https://travis-ci.org/Daiver/NumGrind.svg?branch=master)](https://travis-ci.org/Daiver/NumGrind)

Not for production

Pull-requests are welcomed


#Dependencies
 - Eigen3 for basic linear algebra

#Installation
1. Download Eigen from official site. Copy Eigen dir into 3rdparty dir
2. mkdir build ; cd build ; cmake .. && make
3. Profit?

#TODO

#Common
 - Avoid copy-paste in Symbolic nodes operators
 - Split symbolic graph from reverse autodiff graph
 - Make work with variables values more explicit
 - Split NumGrind and utility code (main.cpp/utils.h/etc)
 - Add NumGrin namespace
 - Add nice interface for GraphNodes
 - Implement toString() for all nodes
 - Improve string formating
 - Refactor all nodes
 - Remove copy-paste from nodes. Add intermediate classes for binary operators

##Optimization (Numerical)
 - Add gradient check test for complex case

##Performance
 - Create big problem for performance tests
 - Add pre-allocated arrays for intermediate results
 - (long term issue) Do not compute backwardPass for Constants
 - (long term issue) Split mutable part of the graph (caches) into separate structure

##Tests
 - Split tests into several files
 - Change tests names

##3rdparty
 - Find a way to automaticly include Eigen without adding it inside repository

##CI
 - Make travis happy #1

##Solved
 - Switch from raw pointers to smart pointers - currently not needed because of GraphManager. Maybe will be implemented later
