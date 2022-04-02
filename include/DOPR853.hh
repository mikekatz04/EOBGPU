#ifndef __DOPR853_HH__
#define __DOPR853_HH__

#include "EOB.hh"

void dormandPrinceSteps_wrap(
        double* x,
        double* xCurrent,
        double* solOld,
        double* h,
        double* arg,
        double* additionalArgs,
        double* k1,
        double* k2,
        double* k3,
        double* k4,
        double* k5,
        double* k6,
        double* k7,
        double* k8,
        double* k9,
        double* k10,
        double* ak_term_buffer,
        int nargs, 
        int numEq,
        int num_add_args,
        SEOBNRv5 *ode_class
);

void error_wrap(
    double* err,
    double* solOld,
    double* solNew,
    double* h,
    double* k1,
    double* k2,
    double* k3,
    double* k6,
    double* k7,
    double* k8,
    double* k9,
    double* k10,
    int numEq,
    int nargs
);

void controllerSuccess_wrap(
    bool* flagSuccess, 
    double* err, 
    double* errOld, 
    bool* previousReject, 
    double* h, 
    double* x,
    int numEq,
    int nargs
);

#endif // __DOPR853_HH__