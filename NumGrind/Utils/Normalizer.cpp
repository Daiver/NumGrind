#include "Normalizer.h"

#include <cfloat>

NumGrind::Utils::Normalizer::Normalizer(const Eigen::MatrixXf &data)
{
    assert(false);//Not implemented yet
    const int nSamples  = data.rows();
    const int nFeatures = data.cols();
    this->mins = Eigen::VectorXf::Zero(nFeatures);
    Eigen::VectorXf maxs = Eigen::VectorXf::Zero(nFeatures);

    mins.fill(FLT_MAX);

    for(int sampleInd = 0; sampleInd < nSamples; ++sampleInd) {
        for(int featInd = 0; featInd < nFeatures; ++featInd) {
            const float val = data(sampleInd, featInd);
            if(val > maxs[featInd]) maxs[featInd] = val;
            if(val < mins[featInd]) mins[featInd] = val;
        }
    }

    deltas = maxs - mins;
}

Eigen::MatrixXf NumGrind::Utils::Normalizer::transform(const Eigen::MatrixXf &data)
{
    const int nSamples  = data.rows();
    const int nFeatures = data.cols();
    assert(nFeatures == mins.size());

    Eigen::MatrixXf res = Eigen::MatrixXf::Zero(data.rows(), nFeatures);
    for(int i = 0; i < nSamples; ++i) {
        res.row(i) = ((data.row(i).transpose() - mins)).array()/deltas.array();//.transpose();
    }
    return res;
}


