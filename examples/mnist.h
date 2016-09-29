#ifndef NUMGRIND_MNIST_H
#define NUMGRIND_MNIST_H

#include <string>
#include "Eigen/Core"

namespace mnist {

    Eigen::MatrixXf readMNISTImages(const std::string &fileName);
    Eigen::VectorXi readMNISTLabels(const std::string &fileName);

};


#endif //NUMGRIND_MNIST_H
