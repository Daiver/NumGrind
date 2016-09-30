#include <iostream>

#include "utils.h"
#include "numgrind.h"
#include "solvers/GradientDescentSolver.h"
#include "solvers/StochasticGradientDescentSolver.h"
#include "solvers/checkgradient.h"
#include "mnist.h"

float sigmoid(float z) {
    return (float) (1.0f / (1.0f + exp(-z)));
}

float sigmoidDer(float z) {
    const float sigZ = sigmoid(z);
    return sigZ * (1.0 - sigZ);
}

void logisticRegressionOperatorAndExample02() {
    using namespace NumGrind;
    using namespace NumGrind::SymbolicGraph;

    GraphManager gm;

    Eigen::MatrixXf data(4, 2);
    Eigen::VectorXf targets(4);
    data << 0, 0,
            0, 1,
            1, 0,
            1, 1;
    targets << 0, 0, 0, 1;

    auto X = gm.constant(data);
    auto y = gm.constant(targets);
    auto w = gm.variable(2, 1, 0);
    auto b = gm.variable(0);
    auto f = apply<sigmoid, sigmoidDer>(matmult(X, w) + b);
    auto residual = f - y;
    auto err = dot(residual, residual);

    auto vars = gm.initializeVariables();

    solvers::SolverSettings settings;
    settings.nMaxIterations = 20;
    solvers::gradientDescent(settings, 0.1, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    f.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl;
    std::cout << f.value() << std::endl;
    std::cout << "W:" << std::endl;
    std::cout << w.value() << std::endl;
    std::cout << "b:" << std::endl;
    std::cout << b.value() << std::endl;
}

void mlpOperatorOrExample01() {
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

    std::cout << "is gradient ok? "
              << NumGrind::solvers::isGradientOk(gm.funcFromNode(&err), gm.gradFromNode(&err), vars) << std::endl;

    NumGrind::solvers::gradientDescent(settings, 2.0, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    f2.node()->forwardPass(vars);
    std::cout << "Function result" << std::endl << f2.value() << std::endl;
    std::cout << "W1:" << std::endl << W1.value() << std::endl;
    std::cout << "W2:" << std::endl << W2.value() << std::endl;
    std::cout << "b2:" << std::endl << b2.value() << std::endl;
}


void mlpOperatorOrExample02() {
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
}

void mlpOperatorOrAndExample03() {
    using namespace NumGrind::SymbolicGraph;
    NumGrind::GraphManager gm;

    Eigen::MatrixXf data(4, 2);
    Eigen::MatrixXf targets(4, 2);
    data << 0, 0,
            0, 1,
            1, 0,
            1, 1;
    targets << 0, 0,
               1, 0,
               1, 0,
               0, 1;

    std::default_random_engine generator;
    generator.seed(42);

    auto X = gm.constant(data);
    auto y = gm.constant(targets);

    auto W1 = gm.variable(NumGrind::utils::gaussf(2, 2, 0.0, 0.5, generator));
    auto b1 = gm.variable(NumGrind::utils::gaussf(1, 2, 0.0, 0.5, generator));
    auto W2 = gm.variable(NumGrind::utils::gaussf(2, 2, 0.0, 0.01, generator));
    auto b2 = gm.variable(NumGrind::utils::gaussf(1, 2, 0.0f, 0.01f, generator));
    //auto b2 = gm.variable(NumGrind::utils::gaussf(0.0f, 0.01f, generator));
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
}

void mnistTest01() {
    const std::string fnameMNISTDir = "/home/daiver/coding/data/mnist/";
    const std::string fnameImagesTrain = fnameMNISTDir + "train-images-idx3-ubyte";
    const std::string fnameLabelsTrain = fnameMNISTDir + "train-labels-idx1-ubyte";
    const std::string fnameImagesTest  = fnameMNISTDir + "t10k-images-idx3-ubyte";
    const std::string fnameLabelsTest  = fnameMNISTDir + "t10k-labels-idx1-ubyte";

    const Eigen::MatrixXf trainData   = mnist::readMNISTImages(fnameImagesTrain);
    const Eigen::VectorXi trainLabels = mnist::readMNISTLabels(fnameLabelsTrain);

    Eigen::MatrixXf labels(trainLabels.size(), 1);

}

int main() {
//    logisticRegressionOperatorAndExample02();
//    mlpOperatorOrExample01();
//    mlpOperatorOrExample02();
    mlpOperatorOrAndExample03();
    //mnistTest01();
    return 0;
}

