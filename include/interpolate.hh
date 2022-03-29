#ifndef __INTERPOLATE_HH__
#define __INTERPOLATE_HH__

#include "cuda_complex.hpp"
#include "global.h"
#include "constants.h"
#include "stdio.h"

void TDInterp(long *templateChannels_ptrs, double *dataTime, long *tsAll, long *propArraysAll, long *c1All, long *c2All, long *c3All, int *old_lengths, int data_length, int numBinAll, int numModes, int *ls, int *ms, long *inds_ptrs, int *inds_start, int *ind_lengths);

void TDInterp2(cmplx *templateChannels, double *dataTime, double *tsAll, double *propArraysAll, double *c1All, double *c2All, double *c3All, int old_length, int *old_lengths, int data_length, int numBinAll, int numModes, int *ls, int *ms, int *inds, int *lengths, int max_length);

void interpolate_TD(double *t_arr, double *propArrays,
                    double *B, double *upper_diag, double *diag, double *lower_diag,
                    int *lengths, int numBinAll, int nsubs);

void all_in_one_likelihood_wrap(cmplx *temp_sum, cmplx *hp_fd, cmplx *hc_fd, cmplx *data, double *psd, double *Fplus, double *Fcross, double *time_shift, double df, int num_bin_all, int nchannels, int data_length);

#endif // __INTERPOLATE_HH__