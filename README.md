# NumGrind
Simple computational graph with reverse mode autodiff framework. Inspired by TensorFlow/Theano and other computational graph tools

[![Build Status](https://travis-ci.org/Daiver/NumGrind.svg?branch=master)](https://travis-ci.org/Daiver/NumGrind)

Not for production

Pull-requests are welcomed

#Examples
##Logistic regression

```cpp
    using namespace SymbolicNodeOps;

    GraphManager man; 

    //Data for "AND" operator
    Eigen::MatrixXf data(4, 2);
    Eigen::VectorXf targets(4);
    data << 0, 0,
            0, 1,
            1, 0,
            1, 1;
    targets << 0, 0, 0, 1;

    auto X = man.constant(data);   
    auto y = man.constant(targets);
    auto w = man.variable(2, 1, 0);
    auto b = man.variable(0);
    auto f = apply<sigmoid, sigmoidDer>(matmult(X, w) + b);//sigmoid and sigmoidDer are simple float (float) functions
    auto residual = f - y;
    auto err = dot(residual, residual);

    auto vars = man.initializeVariables();
    auto grad = man.initializeGradient(vars);

	solvers::gradientDescent(20, 0.1, *err.node(), vars);
    f.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl;
    std::cout << f.value() << std::endl;
    std::cout << "W:" << std::endl;
    std::cout << w.value() << std::endl;
    std::cout << "b:" << std::endl;
    std::cout << b.value() << std::endl;

```

##Multilayer perceptron
```cpp

	using namespace SymbolicNodeOps;
    GraphManager gm;

    Eigen::MatrixXf data(4, 3);
    Eigen::VectorXf targets(4);
    data << 0, 0, 1,
            0, 1, 1,
            1, 0, 1,
            1, 1, 1;
    targets << 0, 1, 1, 0;

    std::default_random_engine generator;

    auto X = gm.constant(data);
    auto y = gm.constant(targets);

    auto W1 = gm.variable(utils::gaussf(3, 2, 0.0, 1.0, generator));
    auto W2 = gm.variable(utils::gaussf(2, 1, 0.0, 0.01, generator));
    auto b2 = gm.variable(utils::gaussf(0.0f, 0.01f, generator));
    auto f1 = apply<sigmoid, sigmoidDer>(matmult(X, W1));
    auto f2 = apply<sigmoid, sigmoidDer>(matmult(f1, W2) + b2);
    auto residual = f2 - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();
    auto grad = gm.initializeGradient(vars);

    solvers::gradientDescent(50, 1.3, *err.node(), vars);
    f2.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl;
    std::cout << f2.value() << std::endl;
    std::cout << "W1:" << std::endl;
    std::cout << W1.value() << std::endl;

    std::cout << "W2:" << std::endl;
    std::cout << W2.value() << std::endl;
    std::cout << "b2:" << std::endl;
    std::cout << b2.value() << std::endl;

```

#Dependencies
 - cmake - build tool
 - Eigen3 - basic linear algebra
 - Download Project - download google test in compile time
 - Google Test - for testing

#Installation
1. Install CMake
2. Download Eigen from official site. Copy Eigen dir into 3rdparty dir
3. mkdir build ; cd build ; cmake .. && make
4. Profit?

#TODO

##Common
 - Switch to session model
 - Avoid copy-paste in Symbolic nodes operators
 - Split symbolic graph from reverse autodiff graph
 - Make work with variables values more explicit
 - Split NumGrind and utility code (main.cpp/utils.h/etc)
 - Add NumGrin namespace
 - Implement toString() for all nodes
 - Improve string formating
 - Refactor all nodes
 - Remove copy-paste from nodes. Add intermediate classes for binary operators

##Optimization (Numerical)
 - Add gradient check test for complex case
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
 - Split tests into several files
 - Change tests names

##3rdparty
 - Find a way to automaticly include Eigen without adding it inside repository

##CI
 - Add runinig tests inside travis

##Solved
 - Switch from raw pointers to smart pointers - currently not needed because of GraphManager. Maybe will be implemented later
 - Make travis happy #1
 - Add nice interface for GraphNodes - done by symbolic graph
