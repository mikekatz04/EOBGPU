#include "interpolate.hh"
#include "cuda_complex.hpp"
#include "global.h"
#include "constants.h"
#include "stdio.h"

#define NUM_THREADS_INTERPOLATE 256
#define NUM_THREADS_BUILD 256

CUDA_CALLABLE_MEMBER
void prep_splines_TD(int i, int length, int interp_i, int ninterp, double *b, double *ud, double *diag, double *ld, double *x, double *y, int numBinAll, int bin_i)
{
    double dx1, dx2, d, slope1, slope2;
    int ind0x, ind1x, ind2x, ind0y, ind1y, ind2y, ind_out;

    double xval0, xval1, xval2, yval1;

    // int numFreqarrs = int(ninterps / num_intermediates);

    // if ((threadIdx.x == 10) && (blockIdx.x == 1)) printf("numFreqarrs %d %d %d %d %d\n", ninterps, interp_i, num_intermediates, numFreqarrs, tArr_i);
    if (i == length - 1)
    {
        ind0y = (length - 3) * ninterp + interp_i;
        ind1y = (length - 2) * ninterp + interp_i;
        ind2y = (length - 1) * ninterp + interp_i;

        ind0x = (length - 3) * numBinAll + bin_i;
        ind1x = (length - 2) * numBinAll + bin_i;
        ind2x = (length - 1) * numBinAll + bin_i;

        ind_out = (length - 1) * ninterp + interp_i;

        xval0 = x[ind0x];
        xval1 = x[ind1x];
        xval2 = x[ind2x];

        dx1 = xval1 - xval0;
        dx2 = xval2 - xval1;
        d = xval2 - xval0;

        yval1 = y[ind1y];

        slope1 = (yval1 - y[ind0y]) / dx1;
        slope2 = (y[ind2y] - yval1) / dx2;

        b[ind_out] = ((dx2 * dx2 * slope1 +
                       (2 * d + dx2) * dx1 * slope2) /
                      d);
        diag[ind_out] = dx1;
        ld[ind_out] = d;
        ud[ind_out] = 0.0;
    }
    else if (i == 0)
    {

        ind0y = 0 * ninterp + interp_i;
        ind1y = 1 * ninterp + interp_i;
        ind2y = 2 * ninterp + interp_i;

        ind0x = 0 * numBinAll + bin_i;
        ind1x = 1 * numBinAll + bin_i;
        ind2x = 2 * numBinAll + bin_i;

        ind_out = 0 * ninterp + interp_i;

        xval0 = x[ind0x];
        xval1 = x[ind1x];
        xval2 = x[ind2x];

        dx1 = xval1 - xval0;
        dx2 = xval2 - xval1;
        d = xval2 - xval0;

        yval1 = y[ind1y];

        // amp
        slope1 = (yval1 - y[ind0y]) / dx1;
        slope2 = (y[ind2y] - yval1) / dx2;

        b[ind_out] = ((dx1 + 2 * d) * dx2 * slope1 +
                      dx1 * dx1 * slope2) /
                     d;
        ud[ind_out] = d;
        ld[ind_out] = 0.0;
        diag[ind_out] = dx2;
    }
    else
    {

        ind0y = (i - 1) * ninterp + interp_i;
        ind1y = (i + 0) * ninterp + interp_i;
        ind2y = (i + 1) * ninterp + interp_i;

        ind0x = (i - 1) * numBinAll + bin_i;
        ind1x = (i + 0) * numBinAll + bin_i;
        ind2x = (i + 1) * numBinAll + bin_i;

        ind_out = i * ninterp + interp_i;

        xval0 = x[ind0x];
        xval1 = x[ind1x];
        xval2 = x[ind2x];

        dx1 = xval1 - xval0;
        dx2 = xval2 - xval1;

        yval1 = y[ind1y];

        // amp
        slope1 = (yval1 - y[ind0y]) / dx1;
        slope2 = (y[ind2y] - yval1) / dx2;

        b[ind_out] = 3.0 * (dx2 * slope1 + dx1 * slope2);
        diag[ind_out] = 2 * (dx1 + dx2);
        ud[ind_out] = dx1;
        ld[ind_out] = dx2;
    }

    // if ((param < 3) && (i == 10) && ((sub_i == 0) || (sub_i == 6))) printf("%d %d %d %e %e %e %e\n", param, sub_i, freqArr_i, b[ind_out], xval1, xval2, yval1);
}

CUDA_KERNEL
void fill_B_TD(double *t_arr, double *y_all, double *B, double *upper_diag, double *diag, double *lower_diag,
               int ninterps, int numBinAll, int *lengths, int nsubs)
{

#ifdef __CUDACC__

    int start1 = blockIdx.x * blockDim.x + threadIdx.x;
    int end1 = ninterps;
    int diff1 = gridDim.x * blockDim.x;

#else

    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;

#endif
    for (int interp_i = start1;
         interp_i < end1; // 2 for re and im
         interp_i += diff1)
    {

        int bin_i = interp_i % numBinAll;
        int length = lengths[bin_i];

        int start2 = 0;
        int diff2 = 1;
        for (int i = start2;
             i < length;
             i += diff2)
        {
            if (i == 0) printf("%d %d\n", bin_i, length);
            prep_splines_TD(i, length, interp_i, ninterps, B, upper_diag, diag, lower_diag, t_arr, y_all, numBinAll, bin_i);
        }
    }
}

CUDA_KERNEL
void interpolate_kern_TD(int *mAll, int n, double *a, double *b, double *c, double *d, int nsubs, int numBinAll)
{
#ifdef __CUDACC__

    int start1 = threadIdx.x + blockDim.x * blockIdx.x;
    ;
    int diff1 = blockDim.x * gridDim.x;

#else

    int start1 = 0;
    int diff1 = 1;

#endif
    for (int interp_i = start1;
         interp_i < n; // 2 for re and im
         interp_i += diff1)
    {
        int bin_i = interp_i % numBinAll;

        int m = mAll[bin_i];
        int ind_i, ind_im1, ind_ip1;
        double w = 0.0;
        for (int i = 1; i < m; i += 1)
        {
            ind_i = i * n + interp_i;
            ind_im1 = (i - 1) * n + interp_i;

            w = a[ind_i] / b[ind_im1];
            b[ind_i] = b[ind_i] - w * c[ind_im1];
            d[ind_i] = d[ind_i] - w * d[ind_im1];
        }

        ind_i = (m - 1) * n + interp_i;

        d[ind_i] = d[ind_i] / b[ind_i];
        for (int i = m - 2; i >= 0; i -= 1)
        {
            ind_i = i * n + interp_i;
            ind_ip1 = (i + 1) * n + interp_i;

            d[ind_i] = (d[ind_i] - c[ind_i] * d[ind_ip1]) / b[ind_i];
        }
    }
}

CUDA_CALLABLE_MEMBER
void fill_coefficients_td(int i, int interp_i, int ninterps, double *dydx, double dx, double *y, double *coeff1, double *coeff2, double *coeff3)
{
    double slope, t, dydx_i;

    int ind_i = i * ninterps + interp_i;
    int ind_ip1 = (i + 1) * ninterps + interp_i;

    slope = (y[ind_ip1] - y[ind_i]) / dx;

    dydx_i = dydx[ind_i];

    t = (dydx_i + dydx[ind_ip1] - 2 * slope) / dx;

    coeff1[ind_i] = dydx_i;
    coeff2[ind_i] = (slope - dydx_i) / dx - t;
    coeff3[ind_i] = t / dx;

    // if ((param == 1) && (i == length - 3) && (sub_i == 0)) printf("freq check: %d %d %d %d %d\n", i, dydx[ind_i], dydx[ind_ip1]);
}

CUDA_KERNEL
void set_spline_constants_TD(double *t_arr, double *y, double *c1, double *c2, double *c3, double *B,
                             int ninterps, int *lengths, int numBinAll, int nsubs)
{

#ifdef __CUDACC__

    int start1 = blockIdx.x * blockDim.x + threadIdx.x;
    int end1 = ninterps;
    int diff1 = gridDim.x * blockDim.x;

#else

    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;

#endif
    for (int interp_i = start1;
         interp_i < ninterps; // 2 for re and im
         interp_i += diff1)
    {

        int bin_i = interp_i % numBinAll;
        int length = lengths[bin_i];

        int start2 = 0;
        int diff2 = 1;
        for (int i = start2;
             i < length - 1;
             i += diff2)
        {
            if (i == 0) printf("fill coeffs: %d %d\n", bin_i, length);
            // TODO: check if there is faster way to do this
            double dt = t_arr[(i + 1) * numBinAll + bin_i] - t_arr[(i)*numBinAll + bin_i];

            fill_coefficients_td(i, interp_i, ninterps, B, dt, y, c1, c2, c3);
        }
    }
}

void interpolate_TD(double *t_arr, double *propArrays,
                    double *B, double *upper_diag, double *diag, double *lower_diag,
                    int *lengths, int numBinAll, int nsubs)
{

    int ninterps = numBinAll * nsubs;

    int nblocks = std::ceil((ninterps + NUM_THREADS_INTERPOLATE - 1) / NUM_THREADS_INTERPOLATE);

    double *c1 = upper_diag; //&interp_array[0 * numInterpParams * amp_phase_size];
    double *c2 = diag;       //&interp_array[1 * numInterpParams * amp_phase_size];
    double *c3 = lower_diag; //&interp_array[2 * numInterpParams * amp_phase_size];

    // printf("%d after response, %d\n", jj, nblocks2);

#ifdef __CUDACC__
    fill_B_TD<<<nblocks, NUM_THREADS_INTERPOLATE>>>(t_arr, propArrays, B, upper_diag, diag, lower_diag, ninterps, numBinAll, lengths, nsubs);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // printf("%d after fill b\n", jj);
    interpolate_kern_TD<<<nblocks, NUM_THREADS_INTERPOLATE>>>(lengths, ninterps, lower_diag, diag, upper_diag, B, nsubs, numBinAll);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_spline_constants_TD<<<nblocks, NUM_THREADS_INTERPOLATE>>>(t_arr, propArrays, c1, c2, c3, B,
                                                                  ninterps, lengths, numBinAll, nsubs);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    fill_B_TD(t_arr, propArrays, B, upper_diag, diag, lower_diag, ninterps, numBinAll, lengths, nsubs);

    // printf("%d after fill b\n", jj);
    interpolate_kern_TD(lengths, ninterps, lower_diag, diag, upper_diag, B, nsubs, numBinAll);

    set_spline_constants_TD(t_arr, propArrays, c1, c2, c3, B,
                            ninterps, lengths, numBinAll, nsubs);
#endif
    // printf("%d after set spline\n", jj);
}

CUDA_CALLABLE_MEMBER
cmplx LIGO_combine_information(double re, double im, int l, int m, double phase_orb)
{
    cmplx h = cmplx(re, im) * gcmplx::exp(cmplx(0.0, -(m * phase_orb)));
    return h;
}

#define NUM_TERMS 4

#define MAX_NUM_COEFF_TERMS 1200
#define MAX_CHANNELS 5
#define MAX_EOB_MODES 20
CUDA_KERNEL
void TD(cmplx *templateChannels, double *dataTimeIn, double *timeOld, double *propArrays, double *c1In, double *c2In, double *c3In, int old_length, int data_length, int numBinAll, int numModes, int *ls_in, int *ms_in, int *inds, int ind_start, int ind_length, int bin_i)
{

    int start, increment;

    CUDA_SHARED int ls[MAX_EOB_MODES];
    CUDA_SHARED int ms[MAX_EOB_MODES];

#ifdef __CUDACC__
    start = threadIdx.x;
    increment = blockDim.x;
#else
    start = 0;
    increment = 1;
#pragma omp parallel for
#endif
    for (int i = start; i < numModes; i += increment)
    {
        ls[i] = ls_in[i];
        ms[i] = ms_in[i];
    }
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    increment = blockDim.x * gridDim.x;
#else
    start = 0;
    increment = 1;
#pragma omp parallel for
#endif
    for (int i = start; i < ind_length; i += increment)
    {
        double t = dataTimeIn[i + ind_start];

        int ind_here = inds[i];

        double t_old = timeOld[ind_here];

        double x = t - t_old;
        double x2 = x * x;
        double x3 = x * x2;

        // printf("CHECK0\n");
        cmplx temp_channels(0.0, 0.0);

        int int_shared = (2 * numModes) * old_length + ind_here;
        // printf("CHECK1 %d %d %d %d\n", numModes, old_length, ind_here, int_shared);
        double phi_orb = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;
        // printf("CHECK2\n");
        for (int mode_i = 0; mode_i < numModes; mode_i += 1)
        {

            int l = ls[mode_i];
            int m = ms[mode_i];

            int int_shared = (mode_i)*old_length + ind_here;
            double re = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            // if ((i == 100) || (i == 101)) printf("%d %d %d %e %e %e %e %e %e\n", window_i, mode_i, i, amp, f, f_old, y[int_shared], c1[int_shared], c2[int_shared]);

            int_shared = (numModes + mode_i) * old_length + ind_here;
            double imag = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            temp_channels += LIGO_combine_information(re, imag, l, m, phi_orb);
        }

        templateChannels[data_length + i] = temp_channels;
    }
}

void TDInterp(long *templateChannels_ptrs, double *dataTime, long *tsAll, long *propArraysAll, long *c1All, long *c2All, long *c3All, int *old_lengths, int data_length, int numBinAll, int numModes, int *ls, int *ms, long *inds_ptrs, int *inds_start, int *ind_lengths)
{
#ifdef __CUDACC__
    cudaStream_t streams[numBinAll];
#endif

#pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        int length_bin_i = ind_lengths[bin_i];
        int ind_start = inds_start[bin_i];
        int *inds = (int *)inds_ptrs[bin_i];
        int old_length = old_lengths[bin_i];

        double *ts = (double *)tsAll[bin_i];
        double *propArrays = (double *)propArraysAll[bin_i];
        double *c1 = (double *)c1All[bin_i];
        double *c2 = (double *)c2All[bin_i];
        double *c3 = (double *)c3All[bin_i];

        cmplx *templateChannels = (cmplx *)templateChannels_ptrs[bin_i];

        int nblocks3 = std::ceil((length_bin_i + NUM_THREADS_BUILD - 1) / NUM_THREADS_BUILD);

#ifdef __CUDACC__
        dim3 gridDim(nblocks3, 1);
        cudaStreamCreate(&streams[bin_i]);
        TD<<<gridDim, NUM_THREADS_BUILD, 0, streams[bin_i]>>>(templateChannels, dataTime, ts, propArrays, c1, c2, c3, old_length, data_length, numBinAll, numModes, ls, ms, inds, ind_start, length_bin_i, bin_i);
#else
        TD(templateChannels, dataTime, ts, propArrays, c1, c2, c3, old_length, data_length, numBinAll, numModes, ls, ms, inds, ind_start, length_bin_i, bin_i);
#endif
    }

#ifdef __CUDACC__
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

#pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        // destroy the streams
        cudaStreamDestroy(streams[bin_i]);
    }
#endif
}

CUDA_KERNEL
void TD2(cmplx *templateChannels, double *dataTimeIn, double *timeOld, double *propArrays, double *c1In, double *c2In, double *c3In, int old_length, int *old_lengths, int data_length, int numBinAll, int numModes, int *ls_in, int *ms_in, int *inds, int *lengths, int max_length)
{

    int start, increment;

    CUDA_SHARED int ls[MAX_EOB_MODES];
    CUDA_SHARED int ms[MAX_EOB_MODES];

#ifdef __CUDACC__
    start = threadIdx.x;
    increment = blockDim.x;
#else
    start = 0;
    increment = 1;
#pragma omp parallel for
#endif
    for (int i = start; i < numModes; i += increment)
    {
        ls[i] = ls_in[i];
        ms[i] = ms_in[i];
    }
    CUDA_SYNC_THREADS;

    int start1, increment1;
#ifdef __CUDACC__
    start1 = blockIdx.y;
    increment1 = gridDim.y;
#else
    start1 = 0;
    increment1 = 1;
#pragma omp parallel for
#endif
    for (int bin_i = start1; bin_i < numBinAll; bin_i += increment1)
    {
        int length_here = lengths[bin_i];

#ifdef __CUDACC__
        start = threadIdx.x;
        increment = blockDim.x;
#else
        start = 0;
        increment = 1;
#pragma omp parallel for
#endif

#ifdef __CUDACC__
        start = blockIdx.x * blockDim.x + threadIdx.x;
        increment = blockDim.x * gridDim.x;
#else
        start = 0;
        increment = 1;
//#pragma omp parallel for
#endif
        for (int i = start; i < length_here; i += increment)
        {
            // check this for adjustable f0
            double t = dataTimeIn[i];

            int ind_here = inds[max_length * bin_i + i];
            int old_length_here = old_lengths[bin_i];

            double t_old = timeOld[bin_i * old_length + ind_here];

            double x = t - t_old;
            double x2 = x * x;
            double x3 = x * x2;

            cmplx temp_channels(0.0, 0.0);

            int ind_base = bin_i * (2 * numModes + 1) * old_length;
            int int_shared = ind_base + (2 * numModes) * old_length_here + ind_here;
            // printf("CHECK1 %d %d %d %d\n", numModes, old_length, ind_here, int_shared);
            double phi_orb = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;
            // printf("CHECK2\n");
            for (int mode_i = 0; mode_i < numModes; mode_i += 1)
            {

                int l = ls[mode_i];
                int m = ms[mode_i];

                int int_shared = ind_base + (mode_i)*old_length_here + ind_here;
                double re = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

                // if ((i == 100) || (i == 101)) printf("%d %d %d %e %e %e %e %e %e\n", window_i, mode_i, i, amp, f, f_old, y[int_shared], c1[int_shared], c2[int_shared]);

                int_shared = ind_base + (numModes + mode_i) * old_length_here + ind_here;
                double imag = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

                temp_channels += LIGO_combine_information(re, imag, l, m, phi_orb);
            }

            templateChannels[bin_i * data_length + i] = temp_channels;
        }
    }
}

void TDInterp2(cmplx *templateChannels, double *dataTime, double *tsAll, double *propArraysAll, double *c1All, double *c2All, double *c3All, int old_length, int *old_lengths, int data_length, int numBinAll, int numModes, int *ls, int *ms, int *inds, int *lengths, int max_length)
{
    int nblocks3 = std::ceil((max_length + NUM_THREADS_BUILD - 1) / NUM_THREADS_BUILD);

#ifdef __CUDACC__
    dim3 gridDim(nblocks3, numBinAll);
    TD2<<<gridDim, NUM_THREADS_BUILD>>>(templateChannels, dataTime, tsAll, propArraysAll, c1All, c2All, c3All, old_length, old_lengths, data_length, numBinAll, numModes, ls, ms, inds, lengths, max_length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    TD2(templateChannels, dataTime, tsAll, propArraysAll, c1All, c2All, c3All, old_length, old_lengths, data_length, numBinAll, numModes, ls, ms, inds, lengths, max_length);
#endif
}

#define NUM_THREADS_SUM 1024
CUDA_KERNEL
void all_in_one_likelihood(cmplx *temp_sum, cmplx *hp_fd, cmplx *hc_fd, cmplx *data, double *psd, double *Fplus, double *Fcross, double *time_shift, double df, int num_bin_all, int nchannels, int data_length)
{
    cmplx sdata[NUM_THREADS_SUM];
    const cmplx I(0.0, 1.0);

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= data_length)
        return;

    double Fplus_bin_channel, Fcross_bin_channel, time_shift_bin_channel;
    cmplx temp_val;
    double psd_val, f;
    cmplx h_i, hp_i_source, hc_i_source, data_i;
    for (int bin_i = blockIdx.y; bin_i < num_bin_all; bin_i += gridDim.y)
    {
        for (int channel_i = blockIdx.z; channel_i < nchannels; channel_i += gridDim.y)
        {
            sdata[tid] = 0.0;
            CUDA_SYNC_THREADS;

            Fplus_bin_channel = Fplus[channel_i * num_bin_all + bin_i];
            Fcross_bin_channel = Fcross[channel_i * num_bin_all + bin_i];
            time_shift_bin_channel = time_shift[channel_i * num_bin_all + bin_i];
            data_i = data[channel_i * data_length + i];
            hp_i_source = hp_fd[bin_i * data_length + i];
            hc_i_source = hc_fd[bin_i * data_length + i];

            f = df * i;

            h_i = (Fplus_bin_channel * hp_i_source + Fcross_bin_channel * hc_i_source) * gcmplx::exp(-I * 2. * PI * f * time_shift_bin_channel);

            temp_val = (data_i - h_i);
            psd_val = psd[channel_i * data_length + i];
            sdata[tid] = (gcmplx::conj(temp_val) * temp_val) / psd_val;

            // if ((i == 4000) && (bin_i == 0) && (channel_i == 0))
            //   printf("%.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e\n", h_i.real(), h_i.imag(), data_i.real(), data_i.imag(), psd_val, sdata[tid].real(), sdata[tid].imag());

            CUDA_SYNC_THREADS;

            for (unsigned int s = 1; s < blockDim.x; s *= 2)
            {
                if (tid % (2 * s) == 0)
                {
                    sdata[tid] += sdata[tid + s];
                }

                CUDA_SYNC_THREADS;
            }
            CUDA_SYNC_THREADS;
            if (tid == 0)
                temp_sum[(blockIdx.x * num_bin_all + bin_i) * nchannels + channel_i] = sdata[0];
        }
    }
}

void all_in_one_likelihood_wrap(cmplx *temp_sum, cmplx *hp_fd, cmplx *hc_fd, cmplx *data, double *psd, double *Fplus, double *Fcross, double *time_shift, double df, int num_bin_all, int nchannels, int data_length)
{
    int nblocks3 = std::ceil((data_length + NUM_THREADS_SUM - 1) / NUM_THREADS_SUM);

#ifdef __CUDACC__
    dim3 gridDim(nblocks3, num_bin_all, nchannels);
    all_in_one_likelihood<<<gridDim, NUM_THREADS_SUM>>>(temp_sum, hp_fd, hc_fd, data, psd, Fplus, Fcross, time_shift, df, num_bin_all, nchannels, data_length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    all_in_one_likelihood<<<gridDim, NUM_THREADS_SUM>>>(temp_sum, hp_fd, hc_fd, data, psd, Fplus, Fcross, time_shift, df, num_bin_all, nchannels, data_length);
#endif
}
