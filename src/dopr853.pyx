import numpy as np
cimport numpy as np
from libcpp cimport bool

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "EOB.hh":
    ctypedef void* SEOBNRv5 'SEOBNRv5'

cdef extern from "DOPR853.hh":
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
    )

    void controllerSuccess_wrap(
        bool* flagSuccess, 
        double* err, 
        double* errOld, 
        bool* previousReject, 
        double* h, 
        double* x,
        int numEq,
        int nargs
    )

@pointer_adjust
def dormandPrinceSteps(
        x,
        xCurrent,
        solOld,
        h,
        arg,
        additionalArgs,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
        k9,
        k10,
        ak_term_buffer,
        nargs, 
        numEq,
        num_add_args,
        ode_class
    ):

    cdef size_t x_in = x
    cdef size_t xCurrent_in = xCurrent
    cdef size_t solOld_in = solOld
    cdef size_t h_in = h
    cdef size_t arg_in = arg
    cdef size_t additionalArgs_in = additionalArgs
    cdef size_t k1_in = k1
    cdef size_t k2_in = k2
    cdef size_t k3_in = k3
    cdef size_t k4_in = k4
    cdef size_t k5_in = k5
    cdef size_t k6_in = k6
    cdef size_t k7_in = k7
    cdef size_t k8_in = k8
    cdef size_t k9_in = k9
    cdef size_t k10_in = k10
    cdef size_t ak_term_buffer_in = ak_term_buffer
    cdef size_t ode_class_in = ode_class.conceiled_ptr

    dormandPrinceSteps_wrap(
        <double*> x_in,
        <double*> xCurrent_in,
        <double*> solOld_in,
        <double*> h_in,
        <double*> arg_in,
        <double*> additionalArgs_in,
        <double*> k1_in,
        <double*> k2_in,
        <double*> k3_in,
        <double*> k4_in,
        <double*> k5_in,
        <double*> k6_in,
        <double*> k7_in,
        <double*> k8_in,
        <double*> k9_in,
        <double*> k10_in,
        <double*> ak_term_buffer_in,
        nargs, 
        numEq,
        num_add_args,
        <SEOBNRv5 *>ode_class_in,
    )


@pointer_adjust
def error(
        err,
        solOld,
        solNew,
        h,
        k1,
        k2,
        k3,
        k6,
        k7,
        k8,
        k9,
        k10,
        nargs, 
        numEq
    ):

    cdef size_t err_in = err
    cdef size_t solOld_in = solOld
    cdef size_t solNew_in = solNew
    cdef size_t h_in = h
    cdef size_t k1_in = k1
    cdef size_t k2_in = k2
    cdef size_t k3_in = k3
    cdef size_t k6_in = k6
    cdef size_t k7_in = k7
    cdef size_t k8_in = k8
    cdef size_t k9_in = k9
    cdef size_t k10_in = k10
    
    error_wrap(
        <double*> err_in,
        <double*> solOld_in,
        <double*> solNew_in,
        <double*> h_in,
        <double*> k1_in,
        <double*> k2_in,
        <double*> k3_in,
        <double*> k6_in,
        <double*> k7_in,
        <double*> k8_in,
        <double*> k9_in,
        <double*> k10_in,
        nargs, 
        numEq
    )

@pointer_adjust
def controllerSuccess(
        flagSuccess,
        err,
        errOld,
        previousReject,
        h,
        x,
        numEq,
        nargs
    ):

    cdef size_t x_in = x
    cdef size_t err_in = err
    cdef size_t errOld_in = errOld
    cdef size_t h_in = h
    cdef size_t flagSuccess_in = flagSuccess
    cdef size_t previousReject_in = previousReject

    controllerSuccess_wrap(
        <bool*> flagSuccess_in,
        <double*> err_in,
        <double*> errOld_in,
        <bool*> previousReject_in,
        <double*> h_in,
        <double*> x_in, 
        numEq,
        nargs,
    )
