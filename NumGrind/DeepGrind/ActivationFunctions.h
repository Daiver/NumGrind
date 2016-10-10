#ifndef NUMGRIND_ACTIVATIONFUNCTIONS_H
#define NUMGRIND_ACTIVATIONFUNCTIONS_H

#include <cmath>

namespace NumGrind {
    namespace DeepGrind {

        float sigmoid(float z) {
            return (float) (1.0f / (1.0f + exp(-z)));
        }

        float sigmoidDer(float z) {
            const float sigZ = sigmoid(z);
            return sigZ * (1.0f - sigZ);
        }

        float relu(float x) {
            return std::max(0.0f, x);
        }

        float reluDer(float x) {
            if (x < 0.0f)
                return 0.0f;
            return x;
        }

    }
}

#endif //NUMGRIND_ACTIVATIONFUNCTIONS_H
