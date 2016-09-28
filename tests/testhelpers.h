//
// Created by daiver on 28.09.16.
//

#ifndef NUMGRIND_HELPERS_H
#define NUMGRIND_HELPERS_H

#include <cmath>

namespace testhelpers {


    inline float sigmoid(float z)
    {
        return static_cast<float>(1.0/(1.0 + exp(-z)));
    }

    inline float sigmoidDer(float z)
    {
        return sigmoid(z) * (1.0 - sigmoid(z));
    }

    inline float square(float x) { return x*x; }
    inline float squareDer(float x) { return 2*x; }
};


#endif //NUMGRIND_HELPERS_H
