#include "utils.h"

Eigen::MatrixXf NumGrind::Utils::labelsToMatrix(const Eigen::VectorXi &labels, const int nClasses)
{
    Eigen::MatrixXf trainLabels = Eigen::MatrixXf::Zero(labels.rows(), nClasses);
    for(int i = 0; i < trainLabels.rows(); ++i)
        trainLabels(i, labels[i]) = 1.0;
    return trainLabels;
}

Eigen::VectorXi NumGrind::Utils::argmaxRowwise(const Eigen::MatrixXf &mat){
    Eigen::VectorXi res = Eigen::VectorXi::Zero(mat.rows());
    for(int row = 0; row < mat.rows(); ++row){
        float maxVal = -10000;
        int maxInd = 0;
        for(int col = 0; col < mat.cols(); ++col){
            if(mat(row, col) > maxVal){
                maxVal = mat(row, col);
                maxInd = col;
            }
        }
        res[row] = maxInd;
    }
    return res;
}