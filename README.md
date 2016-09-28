# NumGrind

Platform | Build status
---------|-------------:
Linux | [![Build Status](https://travis-ci.org/Daiver/NumGrind.svg?branch=master)](https://travis-ci.org/Daiver/NumGrind)

Simple computational graph with reverse mode autodiff framework. Inspired by TensorFlow/Theano and other computational graph tools

I created it for educational purposes


Currently NumGrind in active development. Not for production now.

Pull-requests are welcomed

#Examples

##Multilayer perceptron
```cpp

    using namespace NumGrind::SymbolicGraph;
    NumGrind::GraphManager gm;

    Eigen::MatrixXf data(4, 2);
    Eigen::VectorXf targets(4);
    data << 0, 0,
            0, 1,
            1, 0,
            1, 1;
    targets << 0, 1, 1, 0;

    std::default_random_engine generator;
    generator.seed(42);

    auto X = gm.constant(data);
    auto y = gm.constant(targets);

    auto W1 = gm.variable(NumGrind::utils::gaussf(2, 2, 0.0, 0.5, generator));
    auto b1 = gm.variable(NumGrind::utils::gaussf(1, 2, 0.0, 0.5, generator));
    auto W2 = gm.variable(NumGrind::utils::gaussf(2, 1, 0.0, 0.01, generator));
    auto b2 = gm.variable(NumGrind::utils::gaussf(0.0f, 0.01f, generator));
    auto f1 = apply<sigmoid, sigmoidDer>(matmult(X, W1) + b1);
    auto f2 = apply<sigmoid, sigmoidDer>(matmult(f1, W2) + b2);
    auto residual = f2 - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();

    NumGrind::solvers::SolverSettings settings;
    settings.nMaxIterations = 40;

    std::cout << "is gradient ok? " << NumGrind::solvers::isGradientOk(gm.funcFromNode(&err), gm.gradFromNode(&err), vars) << std::endl;

    NumGrind::solvers::gradientDescent(settings, 2.0, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    f2.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl << f2.value() << std::endl;
    std::cout << "W1:" << std::endl << W1.value() << std::endl;
    std::cout << "W2:" << std::endl << W2.value() << std::endl;
    std::cout << "b2:" << std::endl << b2.value() << std::endl;

```

See examples/main.cpp for more examples

#Dependencies
 - cmake - build tool
 - Eigen3 - basic linear algebra
 - Download Project - download google test in compile time
 - Google Test - for testing

#Installation
Current version tested only under Ubuntu 16.04. But it should be ok on any platform with modern C++11 compiler, Eigen, cmake and DownloadProject

1. Install CMake. In case of Ubuntu
```
sudo apt-get install cmake
```
2. Download Eigen from official site. Copy Eigen dir into 3rdparty dir. Ubuntu:
```
sudo apt-get install libeigen3-dev
```
3. 
```
mkdir build 
cd build 
cmake .. 
make
```
4. Profit?

#TODO

##Common
 - Improve + operator for matrices (numpy like). Change - operator according to +
 - Add reshape node
 - Add transpose node
 - Add some code style
 - Add MNIST test
 - Switch to session model
 - Avoid copy-paste in Symbolic nodes operators
 - Make work with variables values more explicit
 - Refactor all nodes
 - Remove copy-paste from nodes. Add intermediate classes for binary operators

##Optimization (Numerical)
 - Add own settings class for every solver
 - Add and test basic SGD with momentum
 - Add and test complex SGD solvers as Adam/AdaGrad
 - Add common interface for solvers

##Performance
 - Create big problem for performance tests
 - Add pre-allocated arrays for intermediate results
 - (long term issue) Do not compute backwardPass for Constants
 - (long term issue) Split mutable part of the graph (caches) into separate structure
 - (long term issue) Add GPU/OpenCL support

##Tests
 - Own namespaces/classes for tests?
 - Change tests names

##3rdparty
 - Find a way to automaticly include Eigen without adding it inside repository

##CI
 - Add Windows support
 - Add OSX support

##Solved
 - Switch from raw pointers to smart pointers - currently not needed because of GraphManager. Maybe will be implemented later
 - Make travis happy #1
 - Add nice interface for GraphNodes - done by symbolic graph
 - Split symbolic graph from reverse autodiff graph
 - Add NumGrind namespace
 - Implement toString() for all nodes - i just remove toString
 - Add runinig tests inside travis
 - Split tests into several files
 - Split NumGrind and utility code (main.cpp/utils.h/etc)
 - Add gradient check test for complex case - i hope that simple mlp is enough

