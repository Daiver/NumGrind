//
// Created by daiver on 16.10.16.
//

#include "Conv2DFilterShape.h"

DeepGrind::Conv2DFilterShape::Conv2DFilterShape() {

}

DeepGrind::Conv2DFilterShape::Conv2DFilterShape(const int xShape, const int yShape, const int zShape, const int nFilters):
    mXShape(xShape), mYShape(yShape), mZShape(zShape), mNFilters(nFilters)
{

}
