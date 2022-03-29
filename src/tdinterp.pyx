import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "interpolate.hh":
    ctypedef void* cmplx 'cmplx'
    void TDInterp(long* templateChannels_ptrs, double* dataTime, long* tsAll, long* propArraysAll, long* c1All, long* c2All, long* c3All, int* old_lengths, int data_length, int numBinAll, int numModes, int* ls, int* ms, long* inds_ptrs, int* inds_start, int* ind_lengths);
    void TDInterp2(cmplx* templateChannels, double* dataTime, double* tsAll, double* propArraysAll, double* c1All, double* c2All, double* c3All, int old_length, int* old_lengths, int data_length, int numBinAll, int numModes, int* ls, int* ms, int* inds, int* lengths, int max_length);

    void interpolate_TD(double* t_arr, double* propArrays,
                     double* B, double* upper_diag, double* diag, double* lower_diag,
                     int* lengths, int numBinAll, int nsubs);

    void all_in_one_likelihood_wrap(cmplx*temp_sum, cmplx*hp_fd, cmplx*hc_fd, cmplx*data, double*psd, double*Fplus, double*Fcross, double*time_shift, double df, int num_bin_all, int nchannels, int data_length)


@pointer_adjust
def interpolate_TD_wrap(t_arr, propArrays,
                B, upper_diag, diag, lower_diag,
                lengths, numBinAll, nsubs):

    cdef size_t t_arr_in = t_arr
    cdef size_t propArrays_in = propArrays
    cdef size_t B_in = B
    cdef size_t upper_diag_in = upper_diag
    cdef size_t diag_in = diag
    cdef size_t lower_diag_in = lower_diag
    cdef size_t lengths_in = lengths

    interpolate_TD(<double*> t_arr_in, <double*> propArrays_in,
                    <double*> B_in, <double*> upper_diag_in, <double*> diag_in, <double*> lower_diag_in,
                    <int*> lengths_in, numBinAll, nsubs)



@pointer_adjust
def TDInterp_wrap(templateChannels_ptrs, dataTime, ts, propArrays, c1, c2, c3, old_lengths, data_length, numBinAll, numModes, ls, ms, inds_ptrs, inds_start, ind_lengths):

    cdef size_t ts_in = ts
    cdef size_t propArrays_in = propArrays
    cdef size_t templateChannels_ptrs_in = templateChannels_ptrs
    cdef size_t dataTime_in = dataTime
    cdef size_t c1_in = c1
    cdef size_t c2_in = c2
    cdef size_t c3_in = c3
    cdef size_t inds_ptrs_in = inds_ptrs
    cdef size_t inds_start_in = inds_start
    cdef size_t ind_lengths_in = ind_lengths
    cdef size_t old_lengths_in = old_lengths
    cdef size_t ls_in = ls
    cdef size_t ms_in = ms

    TDInterp(<long*> templateChannels_ptrs_in, <double*> dataTime_in, <long*> ts_in, <long*> propArrays_in, <long*> c1_in, <long*> c2_in, <long*> c3_in, <int*> old_lengths_in, data_length, numBinAll, numModes, <int*> ls_in, <int*> ms_in, <long*> inds_ptrs_in, <int*> inds_start_in, <int*> ind_lengths_in)


@pointer_adjust
def TDInterp_wrap2(templateChannels, dataTime, ts, propArrays, c1, c2, c3, old_length, old_lengths, data_length, numBinAll, numModes, ls, ms, inds, lengths, max_length):

    cdef size_t ts_in = ts
    cdef size_t propArrays_in = propArrays
    cdef size_t templateChannels_in = templateChannels
    cdef size_t dataTime_in = dataTime
    cdef size_t c1_in = c1
    cdef size_t c2_in = c2
    cdef size_t c3_in = c3
    cdef size_t inds_in = inds
    cdef size_t old_lengths_in = old_lengths
    cdef size_t lengths_in = lengths
    cdef size_t ls_in = ls
    cdef size_t ms_in = ms

    TDInterp2(<cmplx*> templateChannels_in, <double*> dataTime_in, <double*> ts_in, <double*> propArrays_in, <double*> c1_in, <double*> c2_in, <double*> c3_in, old_length, <int*> old_lengths_in, data_length, numBinAll, numModes, <int*> ls_in, <int*> ms_in, <int*> inds_in, <int*> lengths_in, max_length)

@pointer_adjust
def all_in_one_likelihood(temp_sum, hp_fd, hc_fd, data, psd, Fplus, Fcross, time_shift, df, num_bin_all, nchannels, data_length):

    cdef size_t temp_sum_in = temp_sum
    cdef size_t hp_fd_in = hp_fd
    cdef size_t hc_fd_in = hc_fd
    cdef size_t data_in = data
    cdef size_t psd_in = psd
    cdef size_t Fplus_in = Fplus
    cdef size_t Fcross_in = Fcross
    cdef size_t time_shift_in = time_shift

    all_in_one_likelihood_wrap(<cmplx*> temp_sum_in, <cmplx*>hp_fd_in, <cmplx*>hc_fd_in, <cmplx*>data_in, <double*> psd_in, <double*> Fplus_in, <double*> Fcross_in, <double*> time_shift_in, df, num_bin_all, nchannels, data_length)