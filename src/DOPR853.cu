#include "stdio.h"
#include "global.h"
#include "EOB.hh"
#include "math.h"

// Tolerances
const double rtol = 1e-12;
const double abstol = 1e-12;

// Coefficients for using in Dormand Prince Solver;
const double c2 = 0.526001519587677318785587544488e-01;
const double c3 = 0.789002279381515978178381316732e-01;
const double c4 = 0.118350341907227396726757197510e+00;
const double c5 = 0.281649658092772603273242802490e+00;
const double c6 = 0.333333333333333333333333333333e+00;
const double c7 = 0.25e+00;
const double c8 = 0.307692307692307692307692307692e+00;
const double c9 = 0.651282051282051282051282051282e+00;
const double c10 = 0.6e+00;
const double c11 = 0.857142857142857142857142857142e+00;
const double c14 = 0.1e+00;
const double c15 = 0.2e+00;
const double c16 = 0.777777777777777777777777777778e+00;

const double b1 = 5.42937341165687622380535766363e-2;
const double b6 = 4.45031289275240888144113950566e0;
const double b7 = 1.89151789931450038304281599044e0;
const double b8 = -5.8012039600105847814672114227e0;
const double b9 = 3.1116436695781989440891606237e-1;
const double b10 = -1.52160949662516078556178806805e-1;
const double b11 = 2.01365400804030348374776537501e-1;
const double b12 = 4.47106157277725905176885569043e-2;

const double bhh1 = 0.244094488188976377952755905512e+00;
const double bhh2 = 0.733846688281611857341361741547e+00;
const double bhh3 = 0.220588235294117647058823529412e-01;

const double er1 = 0.1312004499419488073250102996e-01;
const double er6 = -0.1225156446376204440720569753e+01;
const double er7 = -0.4957589496572501915214079952e+00;
const double er8 = 0.1664377182454986536961530415e+01;
const double er9 = -0.3503288487499736816886487290e+00;
const double er10 = 0.3341791187130174790297318841e+00;
const double er11 = 0.8192320648511571246570742613e-01;
const double er12 = -0.2235530786388629525884427845e-01;

const double a21 = 5.26001519587677318785587544488e-2;

const double a31 = 1.97250569845378994544595329183e-2;
const double a32 = 5.91751709536136983633785987549e-2;

const double a41 = 2.95875854768068491816892993775e-2;
const double a43 = 8.87627564304205475450678981324e-2;

const double a51 = 2.41365134159266685502369798665e-1;
const double a53 = -8.84549479328286085344864962717e-1;
const double a54 = 9.24834003261792003115737966543e-1;

const double a61 = 3.7037037037037037037037037037e-2;
const double a64 = 1.70828608729473871279604482173e-1;
const double a65 = 1.25467687566822425016691814123e-1;

const double a71 = 3.7109375e-2;
const double a74 = 1.70252211019544039314978060272e-1;
const double a75 = 6.02165389804559606850219397283e-2;
const double a76 = -1.7578125e-2;

const double a81 = 3.70920001185047927108779319836e-2;
const double a84 = 1.70383925712239993810214054705e-1;
const double a85 = 1.07262030446373284651809199168e-1;
const double a86 = -1.53194377486244017527936158236e-2;
const double a87 = 8.27378916381402288758473766002e-3;

const double a91 = 6.24110958716075717114429577812e-1;
const double a94 = -3.36089262944694129406857109825e0;
const double a95 = -8.68219346841726006818189891453e-1;
const double a96 = 2.75920996994467083049415600797e1;
const double a97 = 2.01540675504778934086186788979e1;
const double a98 = -4.34898841810699588477366255144e1;

const double a101 = 4.77662536438264365890433908527e-1;
const double a104 = -2.48811461997166764192642586468e0;
const double a105 = -5.90290826836842996371446475743e-1;
const double a106 = 2.12300514481811942347288949897e1;
const double a107 = 1.52792336328824235832596922938e1;
const double a108 = -3.32882109689848629194453265587e1;
const double a109 = -2.03312017085086261358222928593e-2;

const double a111 = -9.3714243008598732571704021658e-1;
const double a114 = 5.18637242884406370830023853209e0;
const double a115 = 1.09143734899672957818500254654e0;
const double a116 = -8.14978701074692612513997267357e0;
const double a117 = -1.85200656599969598641566180701e1;
const double a118 = 2.27394870993505042818970056734e1;
const double a119 = 2.49360555267965238987089396762e0;
const double a1110 = -3.0467644718982195003823669022e0;

const double a121 = 2.27331014751653820792359768449e0;
const double a124 = -1.05344954667372501984066689879e1;
const double a125 = -2.00087205822486249909675718444e0;
const double a126 = -1.79589318631187989172765950534e1;
const double a127 = 2.79488845294199600508499808837e1;
const double a128 = -2.85899827713502369474065508674e0;
const double a129 = -8.87285693353062954433549289258e0;
const double a1210 = 1.23605671757943030647266201528e1;
const double a1211 = 6.43392746015763530355970484046e-1;

const double a141 = 5.61675022830479523392909219681e-2;
const double a147 = 2.53500210216624811088794765333e-1;
const double a148 = -2.46239037470802489917441475441e-1;
const double a149 = -1.24191423263816360469010140626e-1;
const double a1410 = 1.5329179827876569731206322685e-1;
const double a1411 = 8.20105229563468988491666602057e-3;
const double a1412 = 7.56789766054569976138603589584e-3;
const double a1413 = -8.298e-3;

const double a151 = 3.18346481635021405060768473261e-2;
const double a156 = 2.83009096723667755288322961402e-2;
const double a157 = 5.35419883074385676223797384372e-2;
const double a158 = -5.49237485713909884646569340306e-2;
const double a1511 = -1.08347328697249322858509316994e-4;
const double a1512 = 3.82571090835658412954920192323e-4;
const double a1513 = -3.40465008687404560802977114492e-4;
const double a1514 = 1.41312443674632500278074618366e-1;

const double a161 = -4.28896301583791923408573538692e-1;
const double a166 = -4.69762141536116384314449447206e0;
const double a167 = 7.68342119606259904184240953878e0;
const double a168 = 4.06898981839711007970213554331e0;
const double a169 = 3.56727187455281109270669543021e-1;
const double a1613 = -1.39902416515901462129418009734e-3;
const double a1614 = 2.9475147891527723389556272149e0;
const double a1615 = -9.15095847217987001081870187138e0;

const double d41 = -0.84289382761090128651353491142e+01;
const double d46 = 0.56671495351937776962531783590e+00;
const double d47 = -0.30689499459498916912797304727e+01;
const double d48 = 0.23846676565120698287728149680e+01;
const double d49 = 0.21170345824450282767155149946e+01;
const double d410 = -0.87139158377797299206789907490e+00;
const double d411 = 0.22404374302607882758541771650e+01;
const double d412 = 0.63157877876946881815570249290e+00;
const double d413 = -0.88990336451333310820698117400e-01;
const double d414 = 0.18148505520854727256656404962e+02;
const double d415 = -0.91946323924783554000451984436e+01;
const double d416 = -0.44360363875948939664310572000e+01;

const double d51 = 0.10427508642579134603413151009e+02;
const double d56 = 0.24228349177525818288430175319e+03;
const double d57 = 0.16520045171727028198505394887e+03;
const double d58 = -0.37454675472269020279518312152e+03;
const double d59 = -0.22113666853125306036270938578e+02;
const double d510 = 0.77334326684722638389603898808e+01;
const double d511 = -0.30674084731089398182061213626e+02;
const double d512 = -0.93321305264302278729567221706e+01;
const double d513 = 0.15697238121770843886131091075e+02;
const double d514 = -0.31139403219565177677282850411e+02;
const double d515 = -0.93529243588444783865713862664e+01;
const double d516 = 0.35816841486394083752465898540e+02;

const double d61 = 0.19985053242002433820987653617e+02;
const double d66 = -0.38703730874935176555105901742e+03;
const double d67 = -0.18917813819516756882830838328e+03;
const double d68 = 0.52780815920542364900561016686e+03;
const double d69 = -0.11573902539959630126141871134e+02;
const double d610 = 0.68812326946963000169666922661e+01;
const double d611 = -0.10006050966910838403183860980e+01;
const double d612 = 0.77771377980534432092869265740e+00;
const double d613 = -0.27782057523535084065932004339e+01;
const double d614 = -0.60196695231264120758267380846e+02;
const double d615 = 0.84320405506677161018159903784e+02;
const double d616 = 0.11992291136182789328035130030e+02;

const double d71 = -0.25693933462703749003312586129e+02;
const double d76 = -0.15418974869023643374053993627e+03;
const double d77 = -0.23152937917604549567536039109e+03;
const double d78 = 0.35763911791061412378285349910e+03;
const double d79 = 0.93405324183624310003907691704e+02;
const double d710 = -0.37458323136451633156875139351e+02;
const double d711 = 0.10409964950896230045147246184e+03;
const double d712 = 0.29840293426660503123344363579e+02;
const double d713 = -0.43533456590011143754432175058e+02;
const double d714 = 0.96324553959188282948394950600e+02;
const double d715 = -0.39177261675615439165231486172e+02;
const double d716 = -0.14972683625798562581422125276e+03;

// Some additional constants for the controller
const double beta = 0.0;
const double alpha = 1.0 / 8.0 - beta * 0.2;
const double safe = 0.9;
const double minscale = 1.0 / 3.0;
const double maxscale = 6.0;

#define NUM_THREADS_DOPR 256

CUDA_KERNEL
void update_arg_and_x(
    double *arg,
    double *solOld,
    double *x,
    double *xCurrent,
    double *ak_term,
    double *h,
    double c_term,
    int numEq,
    int nargs,
    bool set_ak_term_to_zero)
{
#ifdef __CUDACC__
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int increment = blockDim.x * gridDim.x;
#else
    int start = 0;
    int increment = 1;
#pragma omp parallel for
#endif
    for (int eq_i = start; eq_i < numEq; eq_i += increment)
    {
        double h_here = h[eq_i];
        double ak_temp = 0.0;

        xCurrent[eq_i] = x[eq_i] + c_term * h_here;
        // adjust args
        for (int i = 0; i < nargs; i += 1)
        {
            if (set_ak_term_to_zero)
            {
                ak_temp = 0.0;
            }
            else
            {
                ak_temp = ak_term[i * numEq + eq_i];
            }

            arg[i * numEq + eq_i] = solOld[i * numEq + eq_i] + h_here * ak_temp;

            //if ((numEq < 4) && (eq_i < 2) && (i == 2))
            //{
            //    printf("inside update: %d %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e \n", eq_i, i, ak_temp, h_here, solOld[i * numEq + eq_i], xCurrent[eq_i], x[eq_i], c_term, arg[i * numEq + eq_i]);
            //}
        }
    }
}

void update_arg_and_x_wrap(
    double *arg,
    double *solOld,
    double *x,
    double *xCurrent,
    double *ak_term,
    double *h,
    double c_term,
    int numEq,
    int nargs,
    bool set_ak_term_to_zero)
{
#ifdef __CUDACC__
    int nblocks = int(std::ceil((numEq + NUM_THREADS_DOPR - 1) / NUM_THREADS_DOPR));
    update_arg_and_x<<<nblocks, NUM_THREADS_DOPR>>>(arg, solOld, x, xCurrent, ak_term, h, c_term, numEq, nargs, set_ak_term_to_zero);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    update_arg_and_x(arg, solOld, x, xCurrent, ak_term, h, c_term, numEq, nargs, set_ak_term_to_zero);
#endif
}

CUDA_KERNEL
void set_2d_term_to_zero(
    double *term,
    int nargs,
    int numEq)
{
#ifdef __CUDACC__
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int increment = blockDim.x * gridDim.x;
#else
    int start = 0;
    int increment = 1;
#pragma omp parallel for
#endif
    for (int eq_i = start; eq_i < numEq; eq_i += increment)
    {
        for (int i = 0; i < nargs; i += 1)
        {
            term[i * numEq + eq_i] = 0.0;
        }
    }
}

void set_2d_term_to_zero_wrap(
    double *term,
    int nargs,
    int numEq)
{
#ifdef __CUDACC__
    int nblocks = int(std::ceil((numEq + NUM_THREADS_DOPR - 1) / NUM_THREADS_DOPR));
    set_2d_term_to_zero<<<nblocks, NUM_THREADS_DOPR>>>(term, nargs, numEq);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    set_2d_term_to_zero(term, nargs, numEq);
#endif
}

CUDA_KERNEL
void add_2d_term(
    double *term_out,
    double *arr_in,
    double factor,
    int nargs,
    int numEq)
{
#ifdef __CUDACC__
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int increment = blockDim.x * gridDim.x;
#else
    int start = 0;
    int increment = 1;
#pragma omp parallel for
#endif
    for (int eq_i = start; eq_i < numEq; eq_i += increment)
    {
        for (int i = 0; i < nargs; i += 1)
        {
            if ((numEq < 4) && (eq_i < 2) && (i == 2))
            {
                printf("inside add 1: %d %d %.12e %.12e %.12e \n", eq_i, i, factor, term_out[i * numEq + eq_i], arr_in[i * numEq + eq_i]);
            }
            term_out[i * numEq + eq_i] += factor * arr_in[i * numEq + eq_i];
            if ((numEq < 4) && (eq_i < 2) && (i == 2))
            {
                printf("inside add 2: %d %d %.12e %.12e %.12e \n", eq_i, i, factor, term_out[i * numEq + eq_i], arr_in[i * numEq + eq_i]);
            }
        }
    }
}

void add_2d_term_wrap(
    double *term_out,
    double *arr_in,
    double factor,
    int nargs,
    int numEq)
{
#ifdef __CUDACC__
    int nblocks = int(std::ceil((numEq + NUM_THREADS_DOPR - 1) / NUM_THREADS_DOPR));
    add_2d_term<<<nblocks, NUM_THREADS_DOPR>>>(term_out, arr_in, factor, nargs, numEq);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    add_2d_term(term_out, arr_in, factor, nargs, numEq);
#endif
}

void dormandPrinceSteps_wrap(
    double *x,
    double *xCurrent,
    double *solOld,
    double *h,
    double *arg,
    double *additionalArgs,
    double *k1,
    double *k2,
    double *k3,
    double *k4,
    double *k5,
    double *k6,
    double *k7,
    double *k8,
    double *k9,
    double *k10,
    double *ak_term_buffer,
    int nargs,
    int numEq,
    int num_add_args,
    SEOBNRv5 *ode_class,
    int *hCoeffs_index)
{
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);

    // Step 1
    double c1 = 0.0;
    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c1, numEq, nargs, true);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k1, additionalArgs, numEq, hCoeffs_index);

    // Step 2
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a21 * k1
    add_2d_term_wrap(ak_term_buffer, k1, a21, nargs, numEq);

    // TODO: clean up hCoeffs_index
    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c2, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k2, additionalArgs, numEq, hCoeffs_index);
    
    // Step 3
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a31 * k1 + a32 * k2
    add_2d_term_wrap(ak_term_buffer, k1, a31, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k2, a32, nargs, numEq);

    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c3, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k3, additionalArgs, numEq, hCoeffs_index);

    // Step 4
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a41 * k1 + a43 * k3
    add_2d_term_wrap(ak_term_buffer, k1, a41, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k3, a43, nargs, numEq);

    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c4, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k4, additionalArgs, numEq, hCoeffs_index);

    // Step 5
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a51 * k1 + a53 * k3 + a54 * k4
    add_2d_term_wrap(ak_term_buffer, k1, a51, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k3, a53, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k4, a54, nargs, numEq);

    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c5, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k5, additionalArgs, numEq, hCoeffs_index);

    // Step 6
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a61 * k1 + a64 * k4 + a65 * k5
    add_2d_term_wrap(ak_term_buffer, k1, a61, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k4, a64, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k5, a65, nargs, numEq);

    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c6, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k6, additionalArgs, numEq, hCoeffs_index);

    // Step 7
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a71 * k1 + a74 * k4 + a75 * k5 + a76 * k6
    add_2d_term_wrap(ak_term_buffer, k1, a71, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k4, a74, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k5, a75, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k6, a76, nargs, numEq);

    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c7, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k7, additionalArgs, numEq, hCoeffs_index);

    // Step 8
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a81 * k1 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7
    add_2d_term_wrap(ak_term_buffer, k1, a81, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k4, a84, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k5, a85, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k6, a86, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k7, a87, nargs, numEq);

    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c8, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k8, additionalArgs, numEq, hCoeffs_index);

    // Step 9
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a91 * k1 + a94 * k4 + a95 * k5 + a96 * k6 + a97 * k7 + a98 * k8
    add_2d_term_wrap(ak_term_buffer, k1, a91, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k4, a94, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k5, a95, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k6, a96, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k7, a97, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k8, a98, nargs, numEq);

    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c9, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k9, additionalArgs, numEq, hCoeffs_index);

    // Step 10
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a101 * k1 + a104 * k4 + a105 * k5 + a106 * k6 + a107 * k7 + a108 * k8 + a109 * k9
    add_2d_term_wrap(ak_term_buffer, k1, a101, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k4, a104, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k5, a105, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k6, a106, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k7, a107, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k8, a108, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k9, a109, nargs, numEq);

    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c10, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k10, additionalArgs, numEq, hCoeffs_index);

    // Step 11
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a111 * k1 + a114 * k4 + a115 * k5 + a116 * k6 + a117 * k7 + a118 * k8 + a119 * k9 + a1110 * k10
    add_2d_term_wrap(ak_term_buffer, k1, a111, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k4, a114, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k5, a115, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k6, a116, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k7, a117, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k8, a118, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k9, a119, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k10, a1110, nargs, numEq);

    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c11, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k2, additionalArgs, numEq, hCoeffs_index);

    // Step 12
    set_2d_term_to_zero_wrap(ak_term_buffer, nargs, numEq);
    // a121 * k1 + a124 * k4 + a125 * k5 + a126 * k6 + a127 * k7
    // + a128 * k8 + a129 * k9 + a1210 * k10 + a1211 * k2
    add_2d_term_wrap(ak_term_buffer, k1, a121, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k4, a124, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k5, a125, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k6, a126, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k7, a127, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k8, a128, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k9, a129, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k10, a1210, nargs, numEq);
    add_2d_term_wrap(ak_term_buffer, k2, a1211, nargs, numEq);

    double c12 = 1.0;
    update_arg_and_x_wrap(arg, solOld, x, xCurrent, ak_term_buffer, h, c12, numEq, nargs, false);
    ode_class->ODE_Ham_align_AD_wrap(xCurrent, arg, k3, additionalArgs, numEq, hCoeffs_index);
}

CUDA_CALLABLE_MEMBER
double max_here(double x1, double x2)
{
    if (x1 > x2)
        return x1;
    else
        return x2;
}

CUDA_CALLABLE_MEMBER
double min_here(double x1, double x2)
{
    if (x1 < x2)
        return x1;
    else
        return x2;
}

#define MAX_ARGS 10
CUDA_KERNEL
void error(
    double *err,
    double *solOld,
    double *solNew,
    double *h,
    double *k1,
    double *k2,
    double *k3,
    double *k6,
    double *k7,
    double *k8,
    double *k9,
    double *k10,
    int numEq,
    int nargs)
{

    // Number of equations in system
    double n = (double)nargs;

#ifdef __CUDACC__
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int increment = blockDim.x * gridDim.x;
#else
    int start = 0;
    int increment = 1;
#pragma omp parallel for
#endif
    for (int eq_i = start; eq_i < numEq; eq_i += increment)
    {
        // Variables used in system
        double err_temp = 0.0;
        double err2 = 0.0;
        double sk, denom, temp;
        double k1_here, k2_here, k3_here, k6_here, k7_here, k8_here, k9_here, k10_here;
        double solOld_here, solNew_here;
        // Variable used to track index in the loop
        int index;
        double max_term;

        double h_here = h[eq_i];

#pragma unroll
        for (int j = 0; j < nargs; j++)
        {
            index = j * numEq + eq_i;

            k1_here = k1[index];
            k6_here = k6[index];
            k7_here = k7[index];
            k8_here = k8[index];
            k9_here = k9[index];
            k10_here = k10[index];
            k2_here = k2[index];
            k3_here = k3[index];

            temp = b1 * k1_here + b6 * k6_here + b7 * k7_here + b8 * k8_here + b9 * k9_here + b10 * k10_here + b11 * k2_here + b12 * k3_here;
            solOld_here = solOld[index];
            solNew_here = solOld_here + h_here * temp;
            solNew[index] = solNew_here;

            max_term = max_here(solOld_here, solNew_here);

            sk = 1.0 / (abstol + rtol * max_term);

            err2 += pow((temp - bhh1 * k1_here - bhh2 * k9_here - bhh3 * k3_here) * sk, nargs);

            err_temp += pow((er1 * k1_here + er6 * k6_here + er7 * k7_here + er8 * k8_here + er9 * k9_here + er10 * k10_here + er11 * k2_here + er12 * k3_here) * sk, nargs);
            // if (eq_i == 0) printf("%d %lf %.18e  \n", j, n, (er1 * k1_here + er6 * k6_here + er7 * k7_here + er8 * k8_here + er9 * k9_here + er10 * k10_here + er11 * k2_here + er12 * k3_here));
        }

        // TODO: BUG here on err2[i]!!!!
        // Now calculate the denominator and return the error
        denom = err_temp + 0.01 * err2;

        denom = (denom < 0.0) ? 1.0 : denom;

        err[eq_i] = fabs(h_here) * err_temp * sqrt(1.0 / (n * denom));
    }
}

void error_wrap(
    double *err,
    double *solOld,
    double *solNew,
    double *h,
    double *k1,
    double *k2,
    double *k3,
    double *k6,
    double *k7,
    double *k8,
    double *k9,
    double *k10,
    int numEq,
    int nargs)
{
#ifdef __CUDACC__
    int nblocks = int(std::ceil((numEq + NUM_THREADS_DOPR - 1) / NUM_THREADS_DOPR));
    error<<<nblocks, NUM_THREADS_DOPR>>>(
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
        numEq,
        nargs);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    error(
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
        numEq,
        nargs);
#endif
}

CUDA_KERNEL
void controllerSuccess(
    bool *flagSuccess,
    double *err,
    double *errOld,
    bool *previousReject,
    double *h,
    double *x,
    int numEq,
    int nargs)
{
#ifdef __CUDACC__
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int increment = blockDim.x * gridDim.x;
#else
    int start = 0;
    int increment = 1;
#pragma omp parallel for
#endif
    for (int eq_i = start; eq_i < numEq; eq_i += increment)
    {
        // flagSuccess and previousReject were bool*

        // The error was acceptable
        double err_here = err[eq_i];
        double errOld_here = errOld[eq_i];
        bool acceptable = (err_here <= 1.0);
        double scale = 0.0;
        double temp1 = 0.0;
        bool previousReject_here = previousReject[eq_i];

        if (acceptable)
        {
            if (err_here == 0.0)
            {
                scale = maxscale;
            }
            else
            {
                scale = safe * pow(err_here, -alpha) * pow(errOld_here, beta);
                if (scale > maxscale)
                {
                    scale = maxscale;
                }
                else if (scale < minscale)
                {
                    scale = minscale;
                }
            }
            if (previousReject_here)
            {
                if (scale > 1.0)
                    scale = 1.0;
            }
            h[eq_i] = h[eq_i] * scale;

            if (err_here < 1e-4)
            {
                err_here = 1e-4;
            }
            else if (err_here > 1e300)
            {
                err_here = 1e300;
            }
            errOld[eq_i] = err_here;
            previousReject[eq_i] = false;
            flagSuccess[eq_i] = true;
        }
        else
        {
            // The error was too big, we need to make the step size smaller
            previousReject[eq_i] = true;
            flagSuccess[eq_i] = false;
            temp1 = safe * pow(err_here, -alpha);
            if (temp1 < minscale)
            {
                temp1 = minscale;
            }
            else if (temp1 > 1e300)
            {
                temp1 = 1e300;
            }

            // Reduce the size of the step
            scale = temp1;
            h[eq_i] = h[eq_i] * scale;
        }
    }
}

void controllerSuccess_wrap(
    bool *flagSuccess,
    double *err,
    double *errOld,
    bool *previousReject,
    double *h,
    double *x,
    int numEq,
    int nargs)
{
#ifdef __CUDACC__
    int nblocks = int(std::ceil((numEq + NUM_THREADS_DOPR - 1) / NUM_THREADS_DOPR));
    controllerSuccess<<<nblocks, NUM_THREADS_DOPR>>>(
        flagSuccess,
        err,
        errOld,
        previousReject,
        h,
        x,
        numEq,
        nargs);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    controllerSuccess(
        flagSuccess,
        err,
        errOld,
        previousReject,
        h,
        x,
        numEq,
        nargs);
#endif
}

/*
        // Create temportary index for use in loops

        // Step 1


        self.ode(xCurrent, arg, k1, additionalArgs)


        // Step 2
        arg[:] = solOld + h * (a21 * k1)
        xCurrent = x + c2 * h

        self.ode(xCurrent, arg, k2, additionalArgs)

        // Step 3
        arg[:] = solOld + h * (a31 * k1 + a32 * k2)
        xCurrent = x + c3 * h

        self.ode(xCurrent, arg, k3, additionalArgs)

        // Step 4
        arg[:] = solOld + h * (a41 * k1 + a43 * k3)
        xCurrent = x + c4 * h

        self.ode(xCurrent, arg, k4, additionalArgs)

        // Step 5
        arg[:] = solOld + h * (a51 * k1 + a53 * k3 + a54 * k4)
        xCurrent = x + c5 * h

        self.ode(xCurrent, arg, k5, additionalArgs)

        // Step 6
        //pragma unroll
        arg[:] = solOld + h * (a61 * k1 + a64 * k4 + a65 * k5)
        xCurrent = x + c6 * h

        self.ode(xCurrent, arg, k6, additionalArgs)

        // Step 7
        arg[:] = solOld + h * (a71 * k1 + a74 * k4 + a75 * k5 + a76 * k6)
        xCurrent = x + c7 * h

        self.ode(xCurrent, arg, k7, additionalArgs)

        // Step 8
        arg[:] = solOld + h * (a81 * k1 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7)
        xCurrent = x + c8 * h

        self.ode(xCurrent, arg, k8, additionalArgs)//

        // Step 9
        arg[:] = solOld + h * (a91 * k1 + a94 * k4 + a95 * k5 + a96 * k6 + a97 * k7 + a98 * k8)//
        xCurrent = x + c9 * h

        self.ode(xCurrent, arg, k9, additionalArgs)

        // Step 10
        arg[:] = solOld + h * (a101 * k1 + a104 * k4 + a105 * k5 + a106 * k6 + a107 * k7 + a108 * k8 + a109 * k9)
        xCurrent = x + c10 * h

        self.ode(xCurrent, arg, k10, additionalArgs)

        // Step 11
        arg[:] = solOld + h * (a111 * k1 + a114 * k4 + a115 * k5 + a116 * k6 + a117 * k7 + a118 * k8 + a119 * k9 + a1110 * k10)
        xCurrent = x + c11 * h

        self.ode(xCurrent, arg, k2, additionalArgs)

        // Step 12 - Note the use of x + h for this step
        arg[:] = solOld + h * (a121 * k1 + a124 * k4 + a125 * k5 + a126 * k6 + a127 * k7 + a128 * k8 + a129 * k9 + a1210 * k10 + a1211 * k2)
        xCurrent = x + h

        self.ode(xCurrent, arg, k3, additionalArgs)*/
