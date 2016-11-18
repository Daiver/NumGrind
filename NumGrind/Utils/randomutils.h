#ifndef NUMGRIND_RANDOMUTILS_H
#define NUMGRIND_RANDOMUTILS_H

#include <random>
#include "Eigen/Core"

namespace NumGrind {
    namespace Utils {
        template<typename T>
        T gauss(const T mean, const T std, std::default_random_engine &generator);

        float gaussf(const float mean, const float std, std::default_random_engine &generator);

        template<typename T>
        Eigen::Matrix<T, -1, -1>
        gauss(const int rows, const int cols, const T mean, const T std, std::default_random_engine &generator);

        Eigen::MatrixXf gaussf(const int rows, const int cols, const float mean, const float std,
                               std::default_random_engine &generator);
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

#endif //NUMGRIND_RANDOMUTILS_H
