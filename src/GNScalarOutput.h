#ifndef NUMGRINDTEST01_GRAPHNODESCALAROUTPUT_H
#define NUMGRINDTEST01_GRAPHNODESCALAROUTPUT_H

#include "Eigen/Core"
#include "GraphNode.h"

class GNScalarOutput : public GraphNode {
public:
    virtual void backwardPass(const float sensitivity, Eigen::VectorXf &grad) = 0;
    virtual float value() const = 0;
};


#endif //NUMGRINDTEST01_GRAPHNODESCALAROUTPUT_H
