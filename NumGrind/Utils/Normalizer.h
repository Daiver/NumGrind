#ifndef NUMGRIND_NORMALIZER_H
#define NUMGRIND_NORMALIZER_H

#include "Eigen/Core"

namespace NumGrind {
namespace Utils {

class Normalizer {
public:
    Normalizer(const Eigen::MatrixXf &data);

    Eigen::MatrixXf transform(const Eigen::MatrixXf &data);

private:
    Eigen::VectorXf mins;
    Eigen::VectorXf deltas;

};

}
}

#endif
