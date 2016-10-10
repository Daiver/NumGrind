#include <iostream>

#include "utils.h"
#include "numgrind.h"
#include "solvers/GradientDescentSolver.h"
#include "solvers/StochasticGradientDescentSolver.h"
#include "solvers/checkgradient.h"
#include "mnist.h"

float sigmoid(float z)
{
    return (float) (1.0f / (1.0f + exp(-z)));
}

float sigmoidDer(float z)
{
    const float sigZ = sigmoid(z);
    return sigZ * (1.0f - sigZ);
}

float relu(float x)
{
    return std::max(0.0f, x);
}

float reluDer(float x)
{
    if(x < 0.0f)
        return 0.0f;
    return x;
}

void mnistTest01() {
    const std::string fnameMNISTDir = "/home/daiver/coding/data/mnist/";
    const std::string fnameImagesTrain = fnameMNISTDir + "train-images-idx3-ubyte";
    const std::string fnameLabelsTrain = fnameMNISTDir + "train-labels-idx1-ubyte";
    const std::string fnameImagesTest  = fnameMNISTDir + "t10k-images-idx3-ubyte";
    const std::string fnameLabelsTest  = fnameMNISTDir + "t10k-labels-idx1-ubyte";

    const Eigen::MatrixXf trainData   = mnist::readMNISTImages(fnameImagesTrain)/255.0;
    const Eigen::VectorXi trainLabelsPure = mnist::readMNISTLabels(fnameLabelsTrain);

    const Eigen::MatrixXf testData   = mnist::readMNISTImages(fnameImagesTest)/255.0;
    const Eigen::VectorXi testLabelsPure = mnist::readMNISTLabels(fnameLabelsTest);

    Eigen::MatrixXf trainLabels = NumGrind::utils::labelsToMatrix(trainLabelsPure, 10);
    //Eigen::MatrixXf testLabels = labelsToMatrix(testLabelsPure, 10);


    std::default_random_engine generator;
    generator.seed(42);

    using namespace NumGrind::SymbolicGraph;
    NumGrind::GraphManager gm;

    auto X = gm.constant(trainData);
    auto y = gm.constant(trainLabels);

//    auto b1 = gm.variable(1, 10, 0);
    auto W1 = gm.variable(NumGrind::utils::gaussf(trainData.cols(), 300, 0.0, 0.02, generator));
    auto b1 = gm.variable(NumGrind::utils::gaussf(1, 300, 0.0, 0.02, generator));
    auto W2 = gm.variable(NumGrind::utils::gaussf(300, 10, 0.0, 0.01, generator));
    auto b2 = gm.variable(NumGrind::utils::gaussf(1, 10, 0.0f, 0.01f, generator));
    //auto f1 = apply<sigmoid, sigmoidDer>(matmult(X, W1) + b1);
    auto f1 = apply<relu, reluDer>(matmult(X, W1) + b1);
    auto f2 = apply<sigmoid, sigmoidDer>(matmult(f1, W2) + b2);

    auto output = f2;
//    auto residual = f1 - y;
//    auto err = dot(residual, residual);
    //auto tmp = residual * residual;
    //auto err = reduceSum(residual);
    const int batchSize = 32;
    auto err = sumOfSquares(output - y);

    auto vars = gm.initializeVariables();

    NumGrind::solvers::SolverSettings settings;
    settings.nMaxIterations = 20;
    settings.verbose = false;

    float bestAcc = 0.0;

    X.setValue(trainData.block(0, 0, batchSize, 28*28));
    y.setValue(trainLabels.block(0, 0, batchSize, 10));
    NumGrind::solvers::gradientDescent(settings, 0.0003, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    settings.nMaxIterations = 1;
    //for(int i = 0; i < 2001; ++i){
    for(int iterInd = 0; iterInd < 200001; ++iterInd){
        X.setValue(trainData.block((iterInd*batchSize) % trainData.rows(), 0, batchSize, 28*28));
        y.setValue(trainLabels.block((iterInd*batchSize) % trainData.rows(), 0, batchSize, 10));
        NumGrind::solvers::gradientDescent(settings, 0.0030, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
        if(iterInd % 10 == 0)
            std::cout << "Epoch " << iterInd << " err " << err.node()->value() << std::endl;
        if(iterInd%100 == 0){
            X.setValue(testData);
            output.node()->forwardPass(vars);
            auto res = f2.value();
            const auto colwiseMax = NumGrind::utils::argmaxRowwise(res);
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
}

int main() {
    mnistTest01();
    return 0;
}

