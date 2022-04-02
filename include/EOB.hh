#ifndef __EOB_HH__
#define __EOB_HH__

#include "global.h"

void compute_hlms_wrap(cmplx *hlms, double *r_arr, double *phi_arr, double *pr_arr, double *L_arr,
                       double *m1_arr, double *m2_arr, double *chi1_arr, double *chi2_arr,
                       int *num_steps, int num_steps_max, int *ell_arr_in, int *mm_arr_in, int num_modes, int num_bin_all);

void root_find_all_wrap(double *xOut, double *x0In, double *argsIn, double *additionalArgsIn, int max_iter, double err, int numBinAll, int n, int num_args, int num_add_args);

void grad_Ham_align_AD_wrap(double *arg, double *grad_out, double *additionalArgs, int numSys);
void hessian_Ham_align_AD_wrap(double *arg, double *hessian_out, double *additionalArgs, int numSys);

void evaluate_Ham_align_AD_wrap(double *out, double r, double phi, double pr, double pphi, double m_1, double m_2, double chi_1, double chi_2, double K, double d5, double dSO, double dSS);

void IC_cons_wrap(double *res, double *x, double *args, double *additionalArgs, double *grad_out, int numSys);


typedef struct FacWaveformCoeffsTag
{
    double delta22vh3;
    double delta22vh6;
    double delta22vh6S;
    double delta22v8;
    double delta22v8S;
    double delta22vh9;
    double delta22v5;
    double delta22v6;
    double delta22v6S;

    double rho22v2;
    double rho22v3;
    double rho22v3S;
    double rho22v4;
    double rho22v4S;
    double rho22v5;
    double rho22v5S;
    double rho22v6;
    double rho22v6S;
    double rho22v6l;
    double rho22v7;
    double rho22v7S;
    double rho22v8;
    double rho22v8S;
    double rho22v8l;
    double rho22v10;
    double rho22v10l;

    double delta21vh3;
    double delta21vh6;
    double delta21vh6S;
    double delta21vh7;
    double delta21vh7S;
    double delta21vh9;
    double delta21v5;
    double delta21v7;

    double rho21v1;
    double rho21v2;
    double rho21v2S;
    double rho21v3;
    double rho21v3S;
    double rho21v4;
    double rho21v4S;
    double rho21v5;
    double rho21v5S;
    double rho21v6;
    double rho21v6S;
    double rho21v6l;
    double rho21v7;
    double rho21v7S;
    double rho21v7l;
    double rho21v7lS;
    double rho21v8;
    double rho21v8l;
    double rho21v10;
    double rho21v10l;

    double f21v1;
    double f21v1S;
    double f21v3;
    double f21v3S;
    double f21v4;
    double f21v5;
    double f21v6;
    double f21v7c;

    double delta33vh3;
    double delta33vh6;
    double delta33vh6S;
    double delta33vh9;
    double delta33v5;
    double delta33v7;

    double rho33v2;
    double rho33v3;
    double rho33v4;
    double rho33v4S;
    double rho33v5;
    double rho33v5S;
    double rho33v6;
    double rho33v6S;
    double rho33v6l;
    double rho33v7;
    double rho33v7S;
    double rho33v8;
    double rho33v8l;
    double rho33v10;
    double rho33v10l;

    double f33v3;
    double f33v4;
    double f33v5;
    double f33v6;
    double f33v3S;
    double f33vh6;

    double delta32vh3;
    double delta32vh4;
    double delta32vh4S;
    double delta32vh6;
    double delta32vh6S;
    double delta32vh9;

    double rho32v;
    double rho32vS;
    double rho32v2;
    double rho32v2S;
    double rho32v3;
    double rho32v3S;
    double rho32v4;
    double rho32v4S;
    double rho32v5;
    double rho32v5S;
    double rho32v6;
    double rho32v6S;
    double rho32v6l;
    double rho32v8;
    double rho32v8l;

    double delta31vh3;
    double delta31vh6;
    double delta31vh6S;
    double delta31vh7;
    double delta31vh7S;
    double delta31vh9;
    double delta31v5;

    double rho31v2;
    double rho31v3;
    double rho31v4;
    double rho31v4S;
    double rho31v5;
    double rho31v5S;
    double rho31v6;
    double rho31v6S;
    double rho31v6l;
    double rho31v7;
    double rho31v7S;
    double rho31v8;
    double rho31v8l;

    double f31v3;
    double f31v3S;

    double delta44vh3;
    double delta44vh6;
    double delta44vh6S;
    double delta44v5;
    double delta44vh9;

    double rho44v2;
    double rho44v3;
    double rho44v3S;
    double rho44v4;
    double rho44v4S;
    double rho44v5;
    double rho44v5S;
    double rho44v6;
    double rho44v6S;
    double rho44v6l;
    double rho44v8;
    double rho44v8l;
    double rho44v10;
    double rho44v10l;

    double delta43vh3;
    double delta43vh4;
    double delta43vh4S;
    double delta43vh6;

    double rho43v;
    double rho43v2;
    double rho43v4;
    double rho43v4S;
    double rho43v5;
    double rho43v5S;
    double rho43v6;
    double rho43v6l;

    double f43v;
    double f43vS;

    double delta42vh3;
    double delta42vh6;
    double delta42vh6S;

    double rho42v2;
    double rho42v3;
    double rho42v3S;
    double rho42v4;
    double rho42v4S;
    double rho42v5;
    double rho42v5S;
    double rho42v6;
    double rho42v6S;
    double rho42v6l;

    double delta41vh3;
    double delta41vh4;
    double delta41vh4S;
    double delta41vh6;

    double rho41v;
    double rho41v2;
    double rho41v4;
    double rho41v4S;
    double rho41v5;
    double rho41v5S;
    double rho41v6;
    double rho41v6l;

    double f41v;
    double f41vS;

    double delta55vh3;
    double delta55vh6;
    double delta55vh9;

    double delta55v5;
    double rho55v2;
    double rho55v3;
    double rho55v3S;
    double rho55v4;
    double rho55v4S;
    double rho55v5;
    double rho55v5S;
    double rho55v6;
    double rho55v6l;
    double rho55v8;
    double rho55v8l;
    double rho55v10;
    double rho55v10l;
    double f55v3;
    double f55v4;
    double f55v5c;

    double delta54vh3;
    double delta54vh4;
    double delta54vh4S;
    double rho54v2;
    double rho54v3;
    double rho54v3S;
    double rho54v4;
    double rho54v4S;

    double delta53vh3;
    double rho53v2;
    double rho53v3;
    double rho53v3S;
    double rho53v4;
    double rho53v4S;
    double rho53v5;
    double rho53v5S;

    double delta52vh3;
    double delta52vh4;
    double delta52vh4S;
    double rho52v2;
    double rho52v3;
    double rho52v3S;
    double rho52v4;
    double rho52v4S;

    double delta51vh3;
    double rho51v2;
    double rho51v3;
    double rho51v3S;
    double rho51v4;
    double rho51v4S;
    double rho51v5;
    double rho51v5S;

    double delta66vh3;
    double rho66v2;
    double rho66v3;
    double rho66v3S;
    double rho66v4;
    double rho66v4S;

    double delta65vh3;
    double rho65v2;
    double rho65v3;
    double rho65v3S;

    double delta64vh3;
    double rho64v2;
    double rho64v3;
    double rho64v3S;
    double rho64v4;
    double rho64v4S;

    double delta63vh3;
    double rho63v2;
    double rho63v3;
    double rho63v3S;

    double delta62vh3;
    double rho62v2;
    double rho62v3;
    double rho62v3S;
    double rho62v4;
    double rho62v4S;

    double delta61vh3;
    double rho61v2;
    double rho61v3;
    double rho61v3S;

    double delta77vh3;
    double rho77v2;
    double rho77v3;
    double rho77v3S;

    double rho76v2;

    double delta75vh3;
    double rho75v2;
    double rho75v3;
    double rho75v3S;

    double rho74v2;

    double delta73vh3;
    double rho73v2;
    double rho73v3;
    double rho73v3S;

    double rho72v2;

    double delta71vh3;
    double rho71v2;
    double rho71v3;
    double rho71v3S;

    double rho88v2;
    double rho87v2;
    double rho86v2;
    double rho85v2;
    double rho84v2;
    double rho83v2;
    double rho82v2;
    double rho81v2;
} FacWaveformCoeffs;


class SEOBNRv5{
    public:
        FacWaveformCoeffs * hCoeffs;
        // TODO: add prefixes
        int numSys;
        bool allocated;

    SEOBNRv5();

    ~SEOBNRv5();

    void deallocate_information();
    void update_information(double *m1_, double *m2_, double *eta_, double *tplspin_, double *chi_S_, double *chi_A, int use_hm, int numSys_);
    void RR_force_wrap(double *force_out, double *grad_out, double *args, double *additionalArgs, int numSys);
    void root_find_scalar_all_wrap(double *pr_res, double *start_bounds, double *argsIn, double *additionalArgsIn, int max_iter, double err, int numBinAll, int num_args, int num_add_args);
    void ODE_Ham_align_AD_wrap(double *x, double *arg, double *k, double *additionalArgs, int numSys);
    void IC_diss_wrap(double* out, double *pr, double *args, double *additionalArgs, double *grad_out, double *grad_temp_force, double *hess_out, double *force_out, int numSys);

};
#endif // __EOB_HH__
