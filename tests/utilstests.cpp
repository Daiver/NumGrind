#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "Utils/utils.h"
#include "Utils/Normalizer.h"

TEST(NumGrindUtilsSuit, testRowSampling01) {
    auto indices = {1, 3, 0};
    Eigen::MatrixXf mat(5, 2);
    mat <<  10, -1,
            20, -2,
            30, -3,
            40, -4,
            50, -5;
    Eigen::MatrixXf res(3, 2);
    NumGrind::Utils::sampleRowsByIndices(indices, mat, res);
    ASSERT_FLOAT_EQ(res(0, 0), 20);
    ASSERT_FLOAT_EQ(res(0, 1), -2);
    ASSERT_FLOAT_EQ(res(1, 0), 40);
    ASSERT_FLOAT_EQ(res(1, 1), -4);
    ASSERT_FLOAT_EQ(res(2, 0), 10);
    ASSERT_FLOAT_EQ(res(2, 1), -1);
}



