#ifndef NUMGRIND_SOLVERSETTINGS_H
#define NUMGRIND_SOLVERSETTINGS_H

class SolverSettings {
public:

    int nMaxIterations = 0;
    double minGradL2 = 1e-6;
    double minDErr = 1e-9;
    bool verbose = true;
};


#endif //NUMGRIND_SOLVERSETTINGS_H
