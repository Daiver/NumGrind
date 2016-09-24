#ifndef NUMGRINDTEST01_UTILS_H
#define NUMGRINDTEST01_UTILS_H

#include <vector>
#include "Eigen/Core"

namespace utils {

    inline Eigen::VectorXf vec2EVecf(const std::vector<float> &vec)
    {
        Eigen::VectorXf res(vec.size());
        for(int i = 0; i < vec.size(); ++i)
            res[i] = vec[i];
        return res;
    }
}

#endif //NUMGRINDTEST01_UTILS_H
