# NumGrind

Platform | Build status
---------|-------------:
Linux <br> OSX| [![Build Status](https://travis-ci.org/Daiver/NumGrind.svg?branch=master)](https://travis-ci.org/Daiver/NumGrind)

Simple computational graph with reverse mode autodiff framework. Inspired by TensorFlow/Theano and other computational graph tools

I created it for educational purposes

Currently NumGrind in active development. It is not for production now.

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
    auto err = sumOfSquares(f2 - y);

    auto vars = gm.initializeVariables();

    NumGrind::solvers::SolverSettings settings;
    settings.nMaxIterations = 500;
    NumGrind::solvers::StochasticGradientDescentSolver solver(settings, 4.0);

    std::cout << "is gradient ok? "
              << NumGrind::solvers::isGradientOk(gm.funcFromNode(&err), gm.gradFromNode(&err), vars)
              << std::endl;

    const int nIters = 100;
    std::uniform_int_distribution<int> dist(0, data.rows() - 1);
    for (int iter = 0; iter < nIters; ++iter) {
        const int index = dist(generator);
        const Eigen::MatrixXf sample = data.row(index);
        const Eigen::MatrixXf label = targets.row(index);
        X.setValue(data);
        y.setValue(targets);
        solver.makeStep(gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    }
    
    f2.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl << f2.value() << std::endl;
    std::cout << "W1:" << std::endl << W1.value() << std::endl;
    std::cout << "W2:" << std::endl << W2.value() << std::endl;
    std::cout << "b2:" << std::endl << b2.value() << std::endl;
```

##MNIST Example
```cpp

    const std::string fnameMNISTDir = "/home/daiver/coding/data/mnist/";
    const std::string fnameImagesTrain = fnameMNISTDir + "train-images-idx3-ubyte";
    const std::string fnameLabelsTrain = fnameMNISTDir + "train-labels-idx1-ubyte";
    const std::string fnameImagesTest  = fnameMNISTDir + "t10k-images-idx3-ubyte";
    const std::string fnameLabelsTest  = fnameMNISTDir + "t10k-labels-idx1-ubyte";

    const Eigen::MatrixXf trainData   = mnist::readMNISTImages(fnameImagesTrain)/255.0;
    const Eigen::VectorXi trainLabelsPure = mnist::readMNISTLabels(fnameLabelsTrain);

    const Eigen::MatrixXf testData   = mnist::readMNISTImages(fnameImagesTest)/255.0;
    const Eigen::VectorXi testLabelsPure = mnist::readMNISTLabels(fnameLabelsTest);

    Eigen::MatrixXf trainLabels = labelsToMatrix(trainLabelsPure, 10);

    std::default_random_engine generator;
    generator.seed(42);

    using namespace NumGrind::SymbolicGraph;
    NumGrind::GraphManager gm;

    auto X = gm.constant(trainData);
    auto y = gm.constant(trainLabels);

    auto W1 = gm.variable(NumGrind::utils::gaussf(trainData.cols(), 800, 0.0, 0.02, generator));
    auto b1 = gm.variable(NumGrind::utils::gaussf(1, 800, 0.0, 0.02, generator));
    auto W2 = gm.variable(NumGrind::utils::gaussf(800, 10, 0.0, 0.01, generator));
    auto b2 = gm.variable(NumGrind::utils::gaussf(1, 10, 0.0f, 0.01f, generator));
    auto f1 = apply<relu, reluDer>(matmult(X, W1) + b1);
    auto f2 = apply<sigmoid, sigmoidDer>(matmult(f1, W2) + b2);

    auto output = f2;
    const int batchSize = 50;
    auto err = sumOfSquares(output - y);

    auto vars = gm.initializeVariables();

    NumGrind::solvers::SolverSettings settings;
    settings.nMaxIterations = 20;
    settings.verbose = false;

    float bestAcc = 0.0;

    X.setValue(trainData.block(0, 0, batchSize, 28*28));
    y.setValue(trainLabels.block(0, 0, batchSize, 10));
    NumGrind::solvers::gradientDescent(settings, 0.0003, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    settings.nMaxIterations = 2;
    for(int i = 0; i < 50001; ++i){
        X.setValue(trainData.block((i*batchSize) % trainData.rows(), 0, batchSize, 28*28));
        y.setValue(trainLabels.block((i*batchSize) % trainData.rows(), 0, batchSize, 10));
        NumGrind::solvers::gradientDescent(settings, 0.0015, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
        std::cout << "Epoch " << i << " err " << err.node()->value() << std::endl;
        if(i%100 == 0){
            X.setValue(testData);
            output.node()->forwardPass(vars);
            auto res = f2.value();
            const auto colwiseMax = argmaxRowwise(res);
            int nErr = 0;
            for(int j = 0; j < colwiseMax.rows(); ++j){
                if(colwiseMax[j] != testLabelsPure[j])
                    nErr += 1;
            }
            const float fErr = (float)nErr/testLabelsPure.rows();
            const float acc = (1.0 - fErr) * 100;
            if(acc > bestAcc)
                bestAcc = acc;
            std::cout 
                      << std::endl 
                      << "Test error " << fErr << ", "
                      << "acc " << (1.0 - fErr) * 100 << "%, "
                      << "n errors " << (float)nErr << ", " 
                      << "best " << bestAcc << "% "
                      << std::endl << std::endl;
        }
    }


```

See examples/main.cpp for more examples

#Dependencies
 - cmake - build tool
 - Eigen3 - basic linear algebra
 - Download Project - download google test in compile time
 - Google Test - for testing

#Installation
Current version tested only under Ubuntu 16.04. But it should be ok on any platform with modern C++11 compiler, Eigen, cmake and DownloadProject

##Manual install for any platform
1. Install CMake.
2. Download Eigen from official site. Or install it by any another way. Copy Eigen dir into 3rdparty dir. Or make link
3. Just compile project with cmake: mkdir build ; cd build ; cmake .. && make
4. Run tests <path_to_cloned_repo>/build/tests/runUnitTests.
5. It's all. Project is builded. I really don't know why you need NumGrind
6. Profit?

##For Ubuntu/Linux:
Just run it
```
#Install Eigen and CMake
#In case of non-Ubuntu system just replace this by your package manager commands
sudo apt-get install cmake libeigen3-dev

git clone https://github.com/Daiver/NumGrind

cd NumGrind

#Replace it by your Eigen folder if it needed
ln -s /usr/include/eigen3/Eigen 3rdparty/Eigen

mkdir build 
cd build 
cmake .. 
make
./tests/runUnitTests

```

#TODO

##Common
 - Refactor examples
 - Refactor utils
 - Add shape checks into graph to make debugging more easy
 - Improve + operator for matrices (numpy like). Change - operator according to +
 - Add reshape node
 - Add transpose node
 - Add some code style
 - Switch to session model
 - Avoid copy-paste in Symbolic nodes operators
 - Make work with variables values more explicit
 - Refactor all nodes
 - Remove copy-paste from nodes. Add intermediate classes for binary operators

##Example
 - Split examples into several files

##Optimization (Numerical)
 - Add own settings class for every solver
 - Add and test complex SGD solvers as Adam/AdaGrad
 - Add and test basic SGD with momentum
 - Add common interface for solvers

##Deep learning
 - Add batch normalization
 - Add dropout
 - Add convolution network

##Performance
 - Add pre-allocated arrays for intermediate results
 - (long term issue) Do not compute backwardPass for Constants
 - (long term issue) Split mutable part of the graph (caches) into separate structure
 - (long term issue) Add GPU/OpenCL support

##Tests
 - Proper gradient check for MLP 
 - Own namespaces/classes for tests?
 - Change tests names

##3rdparty
 - Find a way to automaticly include Eigen without adding it inside repository

##CI
 - Add Windows support

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
 - Add node for sum of squares
 - Add MNIST test - Done! 97.47% acc!
 - Prepare valgrind or another profiler (gprof?) - currently done by console run
 - Create big problem for performance tests - MNIST is ok, bottleneck is matrices multiplication
 - Add OSX support
 - Travis show multiple warnings. I should fix it

