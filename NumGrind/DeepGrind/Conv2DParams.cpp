//
// Created by daiver on 16.10.16.
//

#include "Conv2DParams.h"

DeepGrind::Conv2DParams::Conv2DParams() {

}

DeepGrind::Conv2DParams::Conv2DParams(const int xShape, const int yShape, const int zShape, const int nFilters):
    mXShape(xShape), mYShape(yShape), mZShape(zShape), mNFilters(nFilters)
{

}
