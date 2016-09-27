#ifndef NUMGRINDTEST01_UTILS_H
#define NUMGRINDTEST01_UTILS_H

#include <random>
#include <vector>
#include "Eigen/Core"

namespace NumGrind {
    namespace utils {

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
    }
}




















//IMPLEMENTATIONS

template <typename T>
inline T NumGrind::utils::gauss(const T mean, const T std, std::default_random_engine &generator)
{
    std::normal_distribution<T> distribution(mean, std);
    return distribution(generator);
}

inline float NumGrind::utils::gaussf(const float mean, const float std, std::default_random_engine &generator)
{
    std::normal_distribution<float> distribution(mean, std);
    return distribution(generator);
}

template <typename T>
inline Eigen::Matrix<T, -1, -1> NumGrind::utils::gauss(const int rows, const int cols, const T mean, const T std, std::default_random_engine &generator)
{
    std::normal_distribution<T> distribution(mean, std);
    Eigen::Matrix<T, -1, -1> res(rows, cols);
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            res(i, j) = distribution(generator);
    return res;
};

inline Eigen::MatrixXf NumGrind::utils::gaussf(const int rows, const int cols, const float mean, const float std, std::default_random_engine &generator)
{
    return gauss<float>(rows, cols, mean, std, generator);
};


#endif //NUMGRINDTEST01_UTILS_H
