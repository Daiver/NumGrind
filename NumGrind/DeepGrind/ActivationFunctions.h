#ifndef NUMGRIND_ACTIVATIONFUNCTIONS_H
#define NUMGRIND_ACTIVATIONFUNCTIONS_H

#include <cmath>

namespace DeepGrind {

    float sigmoid(const float z);

    float sigmoidDer(const float z);

    float relu(const float x);

    float reluDer(const float x);

}
















//IMPLEMENTATIONS

inline float DeepGrind::sigmoid(const float z) 
{
    return (float) (1.0f / (1.0f + exp(-z)));
}

inline float DeepGrind::sigmoidDer(const float z) 
{
    const float sigZ = sigmoid(z);
    return sigZ * (1.0f - sigZ);
}

inline float DeepGrind::relu(const float x) 
{
    return std::max(0.0f, x);
}

inline float DeepGrind::reluDer(const float x) 
{
    if (x < 0.0f)
        return 0.0f;
    return x;
}



#endif //NUMGRIND_ACTIVATIONFUNCTIONS_H
