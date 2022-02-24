#ifndef __INTERPOLATE_HH__
#define __INTERPOLATE_HH__

#include "cuda_complex.hpp"
#include "global.h"
#include "constants.h"
#include "stdio.h"


void TDInterp(long* templateChannels_ptrs, double* dataTime, long* tsAll, long* propArraysAll, long* c1All, long* c2All, long* c3All, double* Fplus_in, double* Fcross_in, int* old_lengths, int data_length, int numBinAll, int numModes, int* ls, int* ms, long* inds_ptrs, int* inds_start, int* ind_lengths, int numChannels);

void TDInterp2(cmplx* templateChannels, double* dataTime, double* tsAll, double* propArraysAll, double* c1All, double* c2All, double* c3All, double* Fplus_in, double* Fcross_in, int old_length, int* old_lengths, int data_length, int numBinAll, int numModes, int* ls, int* ms, int* inds, int* lengths, int max_length, int numChannels);

void interpolate_TD(double* t_arr, double* propArrays,
                                   double* B, double* upper_diag, double* diag, double* lower_diag,
                                   int* lengths, int numBinAll, int nsubs);

#endif // __INTERPOLATE_HH__