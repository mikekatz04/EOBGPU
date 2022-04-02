#ifndef __EOB_HH__
#define __EOB_HH__

#include "global.h"

void compute_hlms_wrap(cmplx *hlms, double *r_arr, double *phi_arr, double *pr_arr, double *L_arr,
                       double *m1_arr, double *m2_arr, double *chi1_arr, double *chi2_arr,
                       int *num_steps, int num_steps_max, int *ell_arr_in, int *mm_arr_in, int num_modes, int num_bin_all);

void root_find_all_wrap(double *xOut, double *x0In, double *argsIn, double *additionalArgsIn, int max_iter, double err, int numBinAll, int n, int num_args, int num_add_args);
void root_find_scalar_all_wrap(double *pr_res, double *start_bounds, double *argsIn, double *additionalArgsIn, int max_iter, double err, int numBinAll, int num_args, int num_add_args);

void grad_Ham_align_AD_wrap(double *arg, double *grad_out, double *additionalArgs, int numSys);
void hessian_Ham_align_AD_wrap(double *arg, double *hessian_out, double *additionalArgs, int numSys);
void ODE_Ham_align_AD_wrap(double *x, double *arg, double *k, double *additionalArgs, int numSys);
void evaluate_Ham_align_AD_wrap(double *out, double r, double phi, double pr, double pphi, double m_1, double m_2, double chi_1, double chi_2, double K, double d5, double dSO, double dSS);

void RR_force_wrap(double *force_out, double *grad_out, double *args, double *additionalArgs, int numSys);
void IC_cons_wrap(double *res, double *x, double *args, double *additionalArgs, double *grad_out, int numSys);
void IC_diss_wrap(double* out, double *pr, double *args, double *additionalArgs, double *grad_out, double *grad_temp_force, double *hess_out, double *force_out, int numSys);
#endif // __EOB_HH__
