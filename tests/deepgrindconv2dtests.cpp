#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "numgrind.h"
#include "DeepGrind/deepgrind.h"

TEST(DeepGrindConv2DTests, test01) {

    DeepGrind::Conv2DParams params(2, 3, 10, 6);
    const int nParamsFor1Filter = params.nParams1FilterBiased();
    const int nParams = params.nParamsBiased();
    ASSERT_EQ(nParamsFor1Filter, 61);
    ASSERT_FLOAT_EQ(nParams, 61*6);

}
