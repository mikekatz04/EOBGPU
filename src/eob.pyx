import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "EOB.hh":
    ctypedef void* cmplx 'cmplx'

    void root_find_all_wrap(double* xOut, double* x0In, double*argsIn, double* additionalArgsIn, int max_iter, double err, int numBinAll, int n, int num_args, int num_add_args);

    void grad_Ham_align_AD_wrap(double *arg, double *grad_out, double *additionalArgs, int numSys);
    void hessian_Ham_align_AD_wrap(double *arg, double *hessian_out, double *additionalArgs, int numSys);
    void evaluate_Ham_align_AD_wrap(double *out, double *r, double *phi, double *pr, double *pphi, double *m_1, double *m_2, double *chi_1, double *chi_2, double K, double d5, double dSO, double dSS, int numSys);

    void EOBComputeNewtonMultipolePrefixes_wrap(cmplx *prefixes, double *m1, double *m2, int ell_max, int numSys);
    void IC_cons_wrap(double *res, double *x, double *args, double *additionalArgs, double *grad_out, int numSys);

    cppclass SEOBNRv5:
        SEOBNRv5();
        void deallocate_information();
        void update_information(double *m1_, double *m2_, double *eta_, double *tplspin_, double *chi_S_, double *chi_A, int use_hm, int numSys_, cmplx *newtonian_prefixes_hlms);
        void RR_force_wrap(double *force_out, double *grad_out, double *args, double *additionalArgs, int numSys, int *hCoeffs_index);
        void root_find_scalar_all_wrap(double *pr_res, double *start_bounds, double *argsIn, double *additionalArgsIn, int max_iter, double err, int numBinAll, int num_args, int num_add_args);
        void ODE_Ham_align_AD_wrap(double *x, double *arg, double *k, double *additionalArgs, int numSys, int *hCoeffs_ind);
        void IC_diss_wrap(double* out, double *pr, double *args, double *additionalArgs, double *grad_out, double *grad_temp_force, double *hess_out, double *force_out, int numSys);
        void compute_hlms_wrap(cmplx *hlms, double *r_arr, double *phi_arr, double *pr_arr, double *L_arr,
                       double *m1_arr, double *m2_arr, double *chi1_arr, double *chi2_arr,
                       int *num_steps, int num_steps_max, int *ell_arr_in, int *mm_arr_in, int num_modes, int num_bin_all, double* additionalArgsIn, int num_add_args)
        void EOBGetSpinFactorizedWaveform_wrap(
            cmplx *out,
            double *r,
            double *phi,
            double *pr,
            double *pphi,
            double *v,
            double *Hreal,
            double *eta,
            double *vPhiInput,
            int l,
            int m,
            cmplx *newtonian_prefix,
            double h22_calib,
            int numSys,
            int traj_length,
            int *hCoeffs_ind
        );

@pointer_adjust
def root_find_all(xOut, x0In, argsIn, additionalArgsIn, max_iter, err, numBinAll, n, num_args, num_add_args):

    cdef size_t xOut_in = xOut
    cdef size_t x0In_in = x0In
    cdef size_t argsIn_in = argsIn
    cdef size_t additionalArgsIn_in = additionalArgsIn

    root_find_all_wrap(<double*> xOut_in, <double*> x0In_in, <double*> argsIn_in, <double*> additionalArgsIn_in, max_iter, err, numBinAll, n, num_args, num_add_args)

@pointer_adjust
def grad_Ham_align_AD(arg, grad_out, additionalArgs, numSys):

    cdef size_t arg_in = arg
    cdef size_t grad_out_in = grad_out
    cdef size_t additionalArgs_in = additionalArgs

    grad_Ham_align_AD_wrap(<double*> arg_in, <double*> grad_out_in, <double*> additionalArgs_in, numSys)

@pointer_adjust
def hessian_Ham_align_AD(arg, hessian_out, additionalArgs, numSys):

    cdef size_t arg_in = arg
    cdef size_t hessian_out_in = hessian_out
    cdef size_t additionalArgs_in = additionalArgs

    hessian_Ham_align_AD_wrap(<double*> arg_in, <double*> hessian_out_in, <double*> additionalArgs_in, numSys)

@pointer_adjust
def evaluate_Ham_align_AD(out, r, phi, pr, pphi, m_1, m_2, chi_1, chi_2, K, d5, dSO, dSS, numSys):
    
    cdef size_t out_in = out
    cdef size_t r_in = r
    cdef size_t phi_in = phi
    cdef size_t pr_in = pr
    cdef size_t pphi_in = pphi
    cdef size_t m_1_in = m_1
    cdef size_t m_2_in = m_2
    cdef size_t chi_1_in = chi_1
    cdef size_t chi_2_in = chi_2

    evaluate_Ham_align_AD_wrap(<double *>out_in, <double *>r_in, <double *>phi_in, <double *>pr_in, <double *>pphi_in, <double *>m_1_in, <double *>m_2_in, <double *>chi_1_in, <double *>chi_2_in, K, d5, dSO, dSS, numSys)

@pointer_adjust
def IC_cons(res, x, args, additionalArgs, grad_out, numSys):

    cdef size_t res_in = res
    cdef size_t x_in = x
    cdef size_t args_in = args
    cdef size_t additionalArgs_in = additionalArgs
    cdef size_t grad_out_in = grad_out

    IC_cons_wrap(<double *>res_in, <double *>x_in, <double *>args_in, <double *>additionalArgs_in, <double *>grad_out_in, numSys)



cdef class SEOBNRv5Class:
    cdef SEOBNRv5* cobj

    def __init__(self):
        self.cobj = new SEOBNRv5()
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')

    def __del__(self):
        del self.cobj
    
    @pointer_adjust
    def update_information(self, m1, m2, eta, tplspin, chi_S, chi_A, use_hm, numSys, newtonian_prefixes_hlms):
        cdef size_t m1_in = m1
        cdef size_t m2_in = m2
        cdef size_t eta_in = eta
        cdef size_t tplspin_in = tplspin
        cdef size_t chi_S_in = chi_S
        cdef size_t chi_A_in = chi_A
        cdef size_t newtonian_prefixes_hlms_in = newtonian_prefixes_hlms

        self.cobj.update_information(<double *>m1_in, <double *>m2_in, <double *>eta_in, <double *>tplspin_in, <double *>chi_S_in, <double *>chi_A_in, use_hm, numSys, <cmplx *>newtonian_prefixes_hlms_in)

    def deallocate_information(self):
        self.cobj.deallocate_information()

    @pointer_adjust
    def IC_diss(self, out, pr, args, additionalArgs, grad_out, grad_temp_force, hess_out, force_out, numSys):
        
        cdef size_t out_in = out
        cdef size_t pr_in = pr
        cdef size_t args_in = args
        cdef size_t additionalArgs_in = additionalArgs
        cdef size_t grad_out_in = grad_out
        cdef size_t grad_temp_force_in = grad_temp_force
        cdef size_t hess_out_in = hess_out
        cdef size_t force_out_in = force_out

        self.cobj.IC_diss_wrap(<double *>out_in, <double *>pr_in, <double *>args_in, <double *>additionalArgs_in, <double *>grad_out_in, <double *>grad_temp_force_in, <double *>hess_out_in, <double *>force_out_in, numSys)

    @pointer_adjust
    def RR_force(self, force_out, grad_out, args, additionalArgs, numSys, hCoeffs_index):

        cdef size_t force_out_in = force_out
        cdef size_t grad_out_in = grad_out
        cdef size_t args_in = args
        cdef size_t additionalArgs_in = additionalArgs
        cdef size_t hCoeffs_index_in = hCoeffs_index

        self.cobj.RR_force_wrap(<double *>force_out_in, <double *>grad_out_in, <double *>args_in, <double *>additionalArgs_in, numSys, <int *>hCoeffs_index_in)

    @pointer_adjust
    def ODE_Ham_align_AD(self, x, arg, k, additionalArgs, numSys, hCoeffs_index):

        cdef size_t x_in = x
        cdef size_t arg_in = arg
        cdef size_t k_in = k
        cdef size_t additionalArgs_in = additionalArgs
        cdef size_t hCoeffs_index_in = hCoeffs_index

        self.cobj.ODE_Ham_align_AD_wrap(<double*> x_in, <double*> arg_in, <double*> k_in, <double*> additionalArgs_in, numSys, <int *>hCoeffs_index_in)

    @pointer_adjust
    def root_find_scalar_all(self, pr_res, start_bounds, argsIn, additionalArgsIn, max_iter, err, numBinAll, num_args, num_add_args):

        cdef size_t pr_res_in = pr_res
        cdef size_t start_bounds_in = start_bounds
        cdef size_t argsIn_in = argsIn
        cdef size_t additionalArgsIn_in = additionalArgsIn

        self.cobj.root_find_scalar_all_wrap(<double*> pr_res_in, <double*> start_bounds_in, <double*> argsIn_in, <double*> additionalArgsIn_in, max_iter, err, numBinAll, num_args, num_add_args)

    @pointer_adjust
    def compute_hlms(self, hlms, r_arr, phi_arr, pr_arr, L_arr,
                    m1_arr, m2_arr, chi1_arr, chi2_arr,
                    num_steps, num_steps_max, ell_arr_in, mm_arr_in, num_modes, num_bin_all, additionalArgsIn, num_add_args):

        cdef size_t hlms_in = hlms
        cdef size_t r_arr_in = r_arr
        cdef size_t phi_arr_in = phi_arr
        cdef size_t pr_arr_in = pr_arr
        cdef size_t L_arr_in = L_arr
        cdef size_t m1_arr_in = m1_arr
        cdef size_t m2_arr_in = m2_arr
        cdef size_t chi1_arr_in = chi1_arr
        cdef size_t chi2_arr_in = chi2_arr
        cdef size_t num_steps_in = num_steps
        cdef size_t ell_arr_in_in = ell_arr_in
        cdef size_t mm_arr_in_in = mm_arr_in
        cdef size_t additionalArgsIn_in = additionalArgsIn

        self.cobj.compute_hlms_wrap(<cmplx*> hlms_in, <double*> r_arr_in, <double*> phi_arr_in, <double*> pr_arr_in, <double*> L_arr_in,
                        <double*> m1_arr_in, <double*> m2_arr_in, <double*> chi1_arr_in, <double*> chi2_arr_in,
                        <int*> num_steps_in, num_steps_max, <int*> ell_arr_in_in, <int*> mm_arr_in_in, num_modes, num_bin_all,
                        <double*>additionalArgsIn_in, num_add_args)

    @pointer_adjust
    def EOBGetSpinFactorizedWaveform(
            self,
            out,
            r,
            phi,
            pr,
            pphi,
            v,
            Hreal,
            eta,
            vPhiInput,
            l,
            m,
            newtonian_prefix,
            h22_calib,
            numSys,
            traj_length,
            hCoeffs_index 
        ):
    
        cdef size_t out_in = out
        cdef size_t r_in = r
        cdef size_t phi_in = phi
        cdef size_t pr_in = pr
        cdef size_t pphi_in = pphi
        cdef size_t v_in = v
        cdef size_t Hreal_in = Hreal
        cdef size_t eta_in = eta
        cdef size_t vPhiInput_in = vPhiInput
        cdef size_t newtonian_prefix_in = newtonian_prefix
        cdef size_t hCoeffs_index_in = hCoeffs_index

        self.cobj.EOBGetSpinFactorizedWaveform_wrap(
            <cmplx *>out_in,
            <double *>r_in,
            <double *>phi_in,
            <double *>pr_in,
            <double *>pphi_in,
            <double *>v_in,
            <double *>Hreal_in,
            <double *>eta_in,
            <double *>vPhiInput_in,
            l,
            m,
            <cmplx *>newtonian_prefix_in,
            h22_calib,
            numSys,
            traj_length,
            <int *>hCoeffs_index_in
        )

    property conceiled_ptr:
        def __get__(self):
            return <long int>self.cobj