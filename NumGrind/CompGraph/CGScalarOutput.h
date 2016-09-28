#ifndef NUMGRINDTEST01_GRAPHNODESCALAROUTPUT_H
#define NUMGRINDTEST01_GRAPHNODESCALAROUTPUT_H

#include "Eigen/Core"
#include "CompGraphNode.h"

namespace NumGrind {
    namespace CompGraph {
        class CGScalarOutput : public CompGraphNode {
        public:
            virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) = 0;

            virtual float value() const = 0;
        };
    }
}

#endif //NUMGRINDTEST01_GRAPHNODESCALAROUTPUT_H
