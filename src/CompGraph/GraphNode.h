#ifndef NUMGRINDTEST01_GRAPHNODE_H
#define NUMGRINDTEST01_GRAPHNODE_H

#include <vector>
#include <string>

#include "Eigen/Core"

namespace NumGrind {
    namespace CompGraph {
        class GraphNode {
        public:
            virtual void forwardPass(const Eigen::VectorXf &vars) = 0;

            virtual std::string toString() const = 0;
        };
    }
}
#endif //NUMGRINDTEST01_GRAPHNODE_H
