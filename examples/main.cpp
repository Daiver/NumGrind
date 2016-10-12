#include <iostream>

#include "Utils/utils.h"
#include "numgrind.h"
#include "Solvers/GradientDescentSolver.h"
#include "Solvers/SGDSolver.h"
#include "Solvers/SGDWithMomentumSolver.h"
#include "Solvers/checkgradient.h"
#include "DeepGrind/ActivationFunctions.h"
#include "mnist.h"


void mnistTest01() {
    const std::string fnameMNISTDir = "/home/daiver/coding/data/mnist/";
    const std::string fnameImagesTrain = fnameMNISTDir + "train-images-idx3-ubyte";
    const std::string fnameLabelsTrain = fnameMNISTDir + "train-labels-idx1-ubyte";
    const std::string fnameImagesTest = fnameMNISTDir + "t10k-images-idx3-ubyte";
    const std::string fnameLabelsTest = fnameMNISTDir + "t10k-labels-idx1-ubyte";

    const Eigen::MatrixXf trainData = mnist::readMNISTImages(fnameImagesTrain) / 255.0;
    const Eigen::VectorXi trainLabelsPure = mnist::readMNISTLabels(fnameLabelsTrain);

    const Eigen::MatrixXf testData = mnist::readMNISTImages(fnameImagesTest) / 255.0;
    const Eigen::VectorXi testLabelsPure = mnist::readMNISTLabels(fnameLabelsTest);

    Eigen::MatrixXf trainLabels = NumGrind::Utils::labelsToMatrix(trainLabelsPure, 10);

    std::default_random_engine generator;
    generator.seed(42);

    using namespace NumGrind::SymbolicGraph;
    NumGrind::GraphManager gm;

    auto X = gm.constant(trainData);
    auto y = gm.constant(trainLabels);

    auto W1 = gm.variable(NumGrind::Utils::gaussf(trainData.cols(), 500, 0.0, 0.02, generator));
    auto b1 = gm.variable(NumGrind::Utils::gaussf(1, 500, 0.0, 0.02, generator));
    auto W2 = gm.variable(NumGrind::Utils::gaussf(500, 10, 0.0, 0.01, generator));
    auto b2 = gm.variable(NumGrind::Utils::gaussf(1, 10, 0.0f, 0.01f, generator));
    auto f1 = apply<NumGrind::DeepGrind::relu, NumGrind::DeepGrind::reluDer>(matmult(X, W1) + b1);
    auto f2 = apply<NumGrind::DeepGrind::sigmoid, NumGrind::DeepGrind::sigmoidDer>(matmult(f1, W2) + b2);

    auto output = f2;
    const int batchSize = 32;
    auto err = sumOfSquares(output - y);

    auto vars = gm.initializeVariables();

    NumGrind::Solvers::SolverSettings settings;
    settings.nMaxIterations = 20;
    settings.verbose = false;

    float bestAcc = 0.0;

    X.setValue(trainData.block(0, 0, batchSize, 28 * 28));
    y.setValue(trainLabels.block(0, 0, batchSize, 10));
    NumGrind::Solvers::gradientDescent(settings, 0.0003, gm.funcFromNode(&err), gm.gradFromNode(&err), vars);
    settings.nMaxIterations = 1;
//    NumGrind::Solvers::SGDSolver solver(settings, 0.002, vars);
    NumGrind::Solvers::SGDWithMomentumSolver solver(settings, 0.0025, 0.9, vars);

    Eigen::MatrixXf trainDataSamples(trainData.rows(), trainData.cols());
    Eigen::MatrixXf trainLabelsSamples(trainData.rows(), trainLabels.cols());
    std::vector<int> shuffledIndices = NumGrind::Utils::range(0, trainData.rows());

//    for(int iterInd = 0; iterInd < 10001; ++iterInd){
    for (int epochInd = 0; epochInd < 10000; ++epochInd) {
        std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), generator);
        NumGrind::Utils::sampleRowsByIndices(shuffledIndices, trainData, trainDataSamples);
        NumGrind::Utils::sampleRowsByIndices(shuffledIndices, trainLabels, trainLabelsSamples);
        solver.setStep(0.003/(0.05*epochInd + 1));//step decay
        solver.setMomentumCoeff(0.9/(0.05*epochInd + 1));
        for (int iterInd = 0; iterInd < trainData.rows() / batchSize + 1; ++iterInd) {
//            NumGrind::Utils::rowDataTargetRandomSampling<float, float>(trainData, trainLabels, generator, trainDataSamples, trainLabelsSamples);
//            X.setValue(trainDataSamples);
//            y.setValue(trainLabelsSamples);
            X.setValue(trainDataSamples.block((iterInd * batchSize) % (trainData.rows() - batchSize), 0, batchSize, 28 * 28));
            y.setValue(trainLabelsSamples.block((iterInd * batchSize) % (trainData.rows() - batchSize), 0, batchSize, 10));
            solver.makeStep(gm.funcFromNode(&err),
                            gm.gradFromNode(&err));

            if (iterInd % 10 == 0)
                std::cout << "Epoch: " << epochInd << " iter: " << iterInd << " err " << err.node()->value() << std::endl;
            if (iterInd % 100 == 0) {
                X.setValue(testData);
                output.node()->forwardPass(solver.vars());
                auto res = f2.value();
                const auto colwiseMax = NumGrind::Utils::argmaxRowwise(res);
                int nErr = 0;
                for (int j = 0; j < colwiseMax.rows(); ++j) {
                    if (colwiseMax[j] != testLabelsPure[j])
                        nErr += 1;
                }
                const float fErr = (float) nErr / testLabelsPure.rows();
                const float acc = (1.0 - fErr) * 100;
                if (acc > bestAcc)
                    bestAcc = acc;
                std::cout
                        << std::endl
                        << "Test error " << fErr << ", "
                        << "acc " << (1.0 - fErr) * 100 << "%, "
                        << "n errors " << (float) nErr << ", "
                        << "best " << bestAcc << "% "
                        << std::endl << std::endl;
            }
        }
    }
}

int main() {
    mnistTest01();
    return 0;
}

