//
// Created by daiver on 28.09.16.
//

#ifndef NUMGRIND_HELPERS_H
#define NUMGRIND_HELPERS_H

#include <cmath>

namespace helpers {


    inline float sigmoid(float z)
    {
        return 1.0/(1.0 + exp(-z));
    }

    inline float sigmoidDer(float z)
    {
        return sigmoid(z) * sigmoid(1.0 - z);
    }

};


#endif //NUMGRIND_HELPERS_H
