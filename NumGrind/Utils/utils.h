#ifndef NUMGRINDTEST01_UTILS_H
#define NUMGRINDTEST01_UTILS_H

#include <random>
#include <vector>
#include "Eigen/Core"

namespace NumGrind {
    namespace Utils {

        inline Eigen::VectorXf vec2EVecf(const std::vector<float> &vec) {
            Eigen::VectorXf res(vec.size());
            for (int i = 0; i < vec.size(); ++i)
                res[i] = vec[i];
            return res;
        }

        template<typename T>
        T gauss(const T mean, const T std, std::default_random_engine &generator);

        float gaussf(const float mean, const float std, std::default_random_engine &generator);

        template<typename T>
        Eigen::Matrix<T, -1, -1>
        gauss(const int rows, const int cols, const T mean, const T std, std::default_random_engine &generator);

        Eigen::MatrixXf gaussf(const int rows, const int cols, const float mean, const float std,
                               std::default_random_engine &generator);

        Eigen::MatrixXf labelsToMatrix(const Eigen::VectorXi &labels, const int nClasses);

        Eigen::VectorXi argmaxRowwise(const Eigen::MatrixXf &mat);

        template <typename T1, typename T2>
        void rowDataTargetRandomSampling(const Eigen::Matrix<T1, -1, -1> &sourceData,
                                         const Eigen::Matrix<T2, -1, -1> &sourceTargets,
                                         std::default_random_engine &generator, Eigen::Matrix<T1, -1, -1> &resData,
                                         Eigen::Matrix<T2, -1, -1> &resTargets);

        std::vector<int> range(const int start, const int end);

        template <typename T1, typename T2>
        void sampleRowsByIndices(const std::vector<int> &indices, const Eigen::Matrix<T1, -1, -1> &mat, Eigen::Matrix<T2, -1, -1> &res);

    }

}




















//IMPLEMENTATIONS

template <typename T>
inline T NumGrind::Utils::gauss(const T mean, const T std, std::default_random_engine &generator)
{
    std::normal_distribution<T> distribution(mean, std);
    return distribution(generator);
}

inline float NumGrind::Utils::gaussf(const float mean, const float std, std::default_random_engine &generator)
{
    std::normal_distribution<float> distribution(mean, std);
    return distribution(generator);
}

template <typename T>
inline Eigen::Matrix<T, -1, -1> NumGrind::Utils::gauss(const int rows, const int cols, const T mean, const T std, std::default_random_engine &generator)
{
    std::normal_distribution<T> distribution(mean, std);
    Eigen::Matrix<T, -1, -1> res(rows, cols);
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            res(i, j) = distribution(generator);
    return res;
};

inline Eigen::MatrixXf NumGrind::Utils::gaussf(const int rows, const int cols, const float mean, const float std, std::default_random_engine &generator)
{
    return gauss<float>(rows, cols, mean, std, generator);
}

template<typename T1, typename T2>
inline void NumGrind::Utils::rowDataTargetRandomSampling(const Eigen::Matrix<T1, -1, -1> &sourceData,
                                                         const Eigen::Matrix<T2, -1, -1> &sourceTargets,
                                                         std::default_random_engine &generator,
                                                         Eigen::Matrix<T1, -1, -1> &resData,
                                                         Eigen::Matrix<T2, -1, -1> &resTargets)
{
    const int nTargetRows = resData.rows();
    const int nDataCols = sourceData.cols();
    const int nTargetCols = sourceTargets.cols();
    assert(sourceData.rows() == sourceTargets.rows());
    assert(nDataCols == resData.cols());
    assert(nTargetCols == resTargets.cols());
    assert(nTargetRows == resTargets.rows());
    std::uniform_int_distribution<> dist(0, sourceData.rows() - 1);
    for(int i = 0; i < nTargetRows; ++i){
        const int index = dist(generator);
        resData.row(i) = sourceData.row(index);
        resTargets.row(i) = sourceTargets.row(index);
    }
}

template<typename T1, typename T2>
inline void ::NumGrind::Utils::sampleRowsByIndices(const std::vector<int> &indices, const Eigen::Matrix<T1, -1, -1> &mat,
                                            Eigen::Matrix<T2, -1, -1> &res) {
    const int nRows = indices.size();
    assert(nRows == res.rows());
    assert(mat.cols() == res.cols());
    for(int i = 0; i < nRows; ++i){
        const int index = indices[i];
        res.row(i) = mat.row(index);
    }
}


#endif //NUMGRINDTEST01_UTILS_H
