#include <iostream>

#include "utils.h"

#include "numgrind.h"
#include "Eigen/Core"

int main() {
    std::cout << "Hello, World!" << std::endl;

    auto n1 = GNVectorVariable({0, 1});
    auto n2 = GNVectorVariable({2, 3});

    auto n3 = GNVectorElementWiseProduct(&n1, &n2);
    auto n4 = GNDotProduct(&n3, &n1);

    auto graph = n4;

    Eigen::VectorXf vars = utils::vec2EVecf({1, 2, 3, 4});
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(vars.size());
    graph.forwardPass(vars);
    graph.backwardPass(1, grad);

    std::cout << graph.toString() << std::endl;
    for(int i = 0; i < grad.size(); ++i)
        std::cout << i << ":" << grad[i] << " ";
    std::cout << std::endl;

    return 0;
}