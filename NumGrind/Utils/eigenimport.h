#ifndef EIGEXPORT_H
#define EIGEXPORT_H

#include <string>
#include <fstream>
#include <iostream>

#include "Eigen/Core"

namespace NumGrind {
    namespace Utils {
        Eigen::MatrixXf readMatFromTxt(const std::string &fname);
    }
}

#endif
