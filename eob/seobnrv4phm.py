from .pytdinterp_cpu import interpolate_TD_wrap as interpolate_TD_wrap_cpu
from .pytdinterp_cpu import TDInterp_wrap2 as TDInterp_wrap_cpu
from .pyEOB_cpu import ODE_Ham_align_AD as ODE_Ham_align_AD_cpu
from .pyEOB_cpu import ODE as ODE_cpu
from .pyEOB_cpu import root_find_scalar_all as root_find_scalar_all_cpu
from .pyEOB_cpu import root_find_all as root_find_all_cpu
from .pyEOB_cpu import compute_hlms as compute_hlms_cpu
import numpy as np
from bbhx.utils.constants import *

MTSUN_SI = 4.925491025543576e-06

from bilby.gw.utils import greenwich_mean_sidereal_time
from bilby.core.utils import speed_of_light

import sys

#from odex.pydopr853 import DOPR853
from .new_Dormand_horiz import DOPR853
from .ode_for_test import ODE_1

try:
    import cupy as xp
    from .pyEOB import compute_hlms as compute_hlms_gpu
    from .pyEOB import root_find_all as root_find_all_gpu
    from .pyEOB import root_find_scalar_all as root_find_scalar_all_gpu
    # from .pyEOB import ODE as ODE_gpu
    from .pyEOB import ODE_Ham_align_AD as ODE_Ham_align_AD_gpu
    from .pytdinterp import TDInterp_wrap2 as TDInterp_wrap_gpu
    from .pytdinterp import interpolate_TD_wrap as interpolate_TD_wrap_gpu
    from .pytdinterp import all_in_one_likelihood as all_in_one_likelihood_gpu
    gpu_available = True
    from cupy.cuda.runtime import setDevice
    setDevice(6)

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")
    import numpy as xp
    gpu_available = False

sys.path.append(
    "/Users/michaelkatz/Research/EOBGPU/eob_development/toys/hamiltonian_prototype/"
)

sys.path.append(
    "/home/mlk667/EOBGPU/eob_development/toys/hamiltonian_prototype/")


#from HTMalign_AC import HTMalign_AC
#from RR_force_aligned import RR_force_2PN
#from initial_conditions_aligned import computeIC


class StoppingCriterion:
    def __init__(self, use_gpu=False, read_out_to_cpu=False):
        self.use_gpu = use_gpu
        self.read_out_to_cpu = read_out_to_cpu

    def __call__(self, step_num, denseOutput):
        if self.use_gpu and self.read_out_to_cpu:
            stop = denseOutput[(step_num.get(), np.zeros_like(
                step_num.get()), np.arange(len(step_num)))] < 6.0
        else:
            stop = denseOutput[(step_num, np.zeros_like(
                step_num), np.arange(len(step_num)))] < 3.0
        return stop


class CubicSplineInterpolantTD:
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines on a GPU. It has a CPU option
    as well. The cubic splines are produced with "not-a-knot" boundary
    conditions.

    This class can be run on GPUs and CPUs.

    args:
        t (1D double xp.ndarray): t values as input for the spline.
        y_all (2D double xp.ndarray): y values for the spline.
            Shape: (ninterps, length).
        use_gpu (bool, optional): If True, prepare arrays for a GPU. Default is
            False.

    """

    def __init__(self, x, y_all, lengths, nsubs, num_bin_all, use_gpu=False):

        if use_gpu:
            self.xp = xp
            self.interpolate_arrays = interpolate_TD_wrap_gpu

        else:
            self.xp = np
            self.interpolate_arrays = interpolate_TD_wrap_cpu

        ninterps = nsubs * num_bin_all
        self.degree = 3

        self.lengths = lengths.astype(self.xp.int32)
        self.max_length = lengths.max().item()
        self.num_bin_all = num_bin_all
        self.nsubs = nsubs

        self.reshape_shape = (self.max_length, nsubs, num_bin_all)

        B = self.xp.zeros((ninterps * self.max_length,))
        self.c1 = upper_diag = self.xp.zeros_like(B)
        self.c2 = diag = self.xp.zeros_like(B)
        self.c3 = lower_diag = self.xp.zeros_like(B)
        self.y = y_all

        self.interpolate_arrays(
            x, y_all, B, upper_diag, diag, lower_diag, lengths, num_bin_all, nsubs,
        )
        # TODO: need to fix last point
        self.x = x

    @property
    def t_shaped(self):
        return self.x.reshape(self.max_length, self.num_bin_all).T

    @property
    def y_shaped(self):
        return self.y.reshape(self.max_length, self.nsubs, self.num_bin_all).T

    @property
    def c1_shaped(self):
        return self.c1.reshape(self.max_length, self.nsubs, self.num_bin_all).T

    @property
    def c2_shaped(self):
        return self.c2.reshape(self.max_length, self.nsubs, self.num_bin_all).T

    @property
    def c3_shaped(self):
        return self.c3.reshape(self.max_length, self.nsubs, self.num_bin_all).T

    @property
    def container(self):
        return [self.x, self.y, self.c1, self.c2, self.c3]


class TDInterp:
    def __init__(self, max_init_len=-1, use_gpu=False):

        if use_gpu:
            self.template_gen = TDInterp_wrap_gpu
            self.xp = xp

        else:
            self.template_gen = TDInterp_wrap_cpu
            self.xp = np

        if max_init_len > 0:
            self.use_buffers = True
            raise NotImplementedError

        else:
            self.use_buffers = False

    def _initialize_template_container(self):
        self.template_carrier = self.xp.zeros(
            (self.num_bin_all * self.data_length),
            dtype=self.xp.complex128,
        )

    @property
    def template_channels(self):
        return [
            self.template_carrier[i].reshape(self.data_length)
            for i in range(self.num_bin_all)
        ]

    def __call__(
        self,
        dataTime,
        interp_container,
        length,
        data_length,
        num_modes,
        ls,
        ms,
        numBinAll,
        dt=1 / 1024,
    ):

        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = numBinAll
        self.data_length = data_length

        # self._initialize_template_container()
        # (ts, y, c1, c2, c3) = interp_container
        splines_ts = interp_container.t_shaped

        ends = self.xp.max(splines_ts, axis=1)
        start_and_end = self.xp.asarray(
            [self.xp.full(self.num_bin_all, 0.0), ends, ]).T

        inds_start_and_end = self.xp.asarray(
            [
                self.xp.searchsorted(dataTime, temp, side="left")
                for temp in start_and_end
            ]
        )

        self.lengths = inds_start_and_end[:, 1].astype(self.xp.int32)
        max_length = self.lengths.max().item()

        inds = self.xp.empty(
            (self.num_bin_all * max_length), dtype=self.xp.int32)

        old_lengths = interp_container.lengths
        old_length = self.xp.max(old_lengths).item()

        for i, ((st, et), ts, current_old_length) in enumerate(
            zip(inds_start_and_end, splines_ts, old_lengths)
        ):
            inds[i * max_length + st: i * max_length + et] = (
                self.xp.searchsorted(ts, dataTime[st:et], side="right").astype(
                    self.xp.int32
                )
                - 1
            )

        self.template_carrier = self.xp.zeros(
            int(data_length * self.num_bin_all),
            dtype=self.xp.complex128,
        )

        ls = ls.astype(np.int32)
        ms = ms.astype(np.int32)

        self.template_gen(
            self.template_carrier,
            dataTime,
            interp_container.t_shaped.flatten(),
            interp_container.y_shaped.flatten(),
            interp_container.c1_shaped.flatten(),
            interp_container.c2_shaped.flatten(),
            interp_container.c3_shaped.flatten(),
            old_length,
            old_lengths,
            self.data_length,
            self.num_bin_all,
            self.num_modes,
            ls,
            ms,
            inds,
            self.lengths,
            max_length,
        )

        return self.template_carrier


class BBHWaveformTD:
    def __init__(
        self, bilby_interferometers, amp_phase_kwargs={}, interp_kwargs={}, use_gpu=False,
    ):
        amp_phase_kwargs["use_gpu"] = use_gpu
        interp_kwargs["use_gpu"] = use_gpu
        self.amp_phase_gen = SEOBNRv4PHM(**amp_phase_kwargs)

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        # TODO: should probably be 2
        self.num_interp_params = 2
        self.interp_response = TDInterp(**interp_kwargs)

        if bilby_interferometers is None:
            raise ValueError("Must provide detector_tensors kwarg.")

        self.detector_tensors = self.xp.asarray(
            [bi.detector_tensor for bi in bilby_interferometers])

        self.detector_start_times = self.xp.asarray(
            [bi.strain_data.start_time for bi in bilby_interferometers], dtype=self.xp.float64
        )

        self.detector_vertex = self.xp.asarray(
            [bi.geometry.vertex for bi in bilby_interferometers], dtype=self.xp.float64
        )

        self.psd = self.xp.asarray(
            [bi.amplitude_spectral_density_array ** 2 for bi in bilby_interferometers], dtype=self.xp.float64
        ).flatten()

        self.psd[self.xp.isinf(self.psd)] = 1e300

        self.data = self.xp.asarray(
            [bi.frequency_domain_strain for bi in bilby_interferometers], dtype=self.xp.complex128
        )
        self.fd_data_length = self.data.shape[1]
        self.nChannels = self.data.shape[0]

        self.data = self.data.flatten()

        self.bilby_interferometers = bilby_interferometers

        self.all_in_one_likelihood = all_in_one_likelihood_gpu

    def get_detector_response(self, ra, dec, geocent_time, psi):
        """ Get the detector response for a particular waveform

        Parameters
        ==========
        waveform_polarizations: dict
            polarizations of the waveform
        parameters: dict
            parameters describing position and time of arrival of the signal

        Returns
        =======
        array_like: A 3x3 array representation of the detector response (signal observed in the interferometer)
        """
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = self.antenna_response(
                parameters['ra'],
                parameters['dec'],
                parameters['geocent_time'],
                parameters['psi'], mode)

            signal[mode] = waveform_polarizations[mode] * det_response
        signal_ifo = sum(signal.values())

        signal_ifo *= self.strain_data.frequency_mask

        time_shift = self.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])

        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - self.strain_data.start_time
        dt = dt_geocent + time_shift

        signal_ifo[self.strain_data.frequency_mask] = signal_ifo[self.strain_data.frequency_mask] * np.exp(
            -1j * 2 * np.pi * dt * self.strain_data.frequency_array[self.strain_data.frequency_mask])

        signal_ifo[self.strain_data.frequency_mask] *= self.calibration_model.get_calibration_factor(
            self.strain_data.frequency_array[self.strain_data.frequency_mask],
            prefix='recalib_{}_'.format(self.name), **parameters)

        return signal_ifo

    def get_detector_information(self, ra, dec, time, psi):
        # TODO: fix lal part

        gmst = self.xp.fmod(self.xp.asarray(
            [greenwich_mean_sidereal_time(t_i) for t_i in time]), 2 * np.pi)

        # convert to phi, theta from ra, dec
        phi = self.xp.asarray(ra) - gmst
        theta = np.pi / 2 - self.xp.asarray(dec)
        psi = self.xp.asarray(psi)

        u = self.xp.array([self.xp.cos(phi) * self.xp.cos(theta), self.xp.cos(theta)
                           * self.xp.sin(phi), -self.xp.sin(theta)]).T
        v = self.xp.array(
            [-self.xp.sin(phi), self.xp.cos(phi), self.xp.zeros_like(phi)]).T
        m = -u * self.xp.sin(psi)[:, None] - v * self.xp.cos(psi)[:, None]
        n = -u * self.xp.cos(psi)[:, None] + v * self.xp.sin(psi)[:, None]

        polarization_tensor_plus = self.xp.einsum(
            '...i,...j->...ij', m, m) - self.xp.einsum('...i,...j->...ij', n, n)
        polarization_tensor_cross = self.xp.einsum(
            '...i,...j->...ij', m, n) + self.xp.einsum('...i,...j->...ij', n, m)

        antenna_response_plus = self.xp.einsum(
            'kij,lij->kl', self.detector_tensors, polarization_tensor_plus)
        antenna_response_cross = self.xp.einsum(
            'kij,lij->kl', self.detector_tensors, polarization_tensor_cross)

        # get time shift
        omega = self.xp.array([self.xp.sin(theta) * self.xp.cos(phi),
                              self.xp.sin(theta) * self.xp.sin(phi), self.xp.cos(theta)]).T

        # detector_coords is delta_d because detector2 in the determination of delta_d is 0,0,0 for from geocenter
        # delta_d = detector2 - dector1  # second array is zeros in bilby
        time_shift = self.xp.einsum(
            "ij, kj->ki", omega, self.detector_vertex) / speed_of_light

        return (antenna_response_plus, antenna_response_cross, time_shift)

    def __call__(
        self,
        m1,
        m2,
        # chi1x,
        # chi1y,
        chi1z,
        # chi2x,
        # chi2y,
        chi2z,
        distance,
        phiRef,
        inc,
        ra,
        dec,
        psi,
        geocent_time,
        sampling_frequency=1024,
        Tobs=60.0,
        modes=None,
        bufferSize=None,
        fill=False,
        fs=20.0,
        return_type="like",
    ):

        # TODO: if t_obs_end = t_mrg

        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        # chi1x = np.atleast_1d(chi1x)
        # chi1y = np.atleast_1d(chi1y)
        chi1z = np.atleast_1d(chi1z)
        # chi2x = np.atleast_1d(chi2x)
        # chi2y = np.atleast_1d(chi2y)
        chi2z = np.atleast_1d(chi2z)
        distance = np.atleast_1d(distance)
        phiRef = np.atleast_1d(phiRef)
        inc = np.atleast_1d(inc)
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        psi = np.atleast_1d(psi)
        geocent_time = np.atleast_1d(geocent_time)

        self.num_bin_all = len(m1)

        self.data_length = data_length = int(Tobs * sampling_frequency)

        self.dataTime = (
            self.xp.arange(data_length, dtype=self.xp.float64)
            * 1.0
            / sampling_frequency
        )

        if modes is None:
            self.num_modes = len(self.amp_phase_gen.allowable_modes)
        else:
            self.num_modes = len(modes)

        """import time as tttttt
        num = 1
        st = tttttt.perf_counter()
        for _ in range(num):"""
        self.amp_phase_gen(
            m1,
            m2,
            # chi1x,
            # chi1y,
            chi1z,
            # chi2x,
            # chi2y,
            chi2z,
            distance,
            phiRef,
            modes=modes,
            fs=fs,
        )
        """et = tttttt.perf_counter()

        print("amp phase", self.num_bin_all, (et - st) /
              num, (et - st) / num / self.num_bin_all)

        st = tttttt.perf_counter()
        for _ in range(num):"""
        splines = CubicSplineInterpolantTD(
            self.amp_phase_gen.t.T.flatten().copy(),
            self.amp_phase_gen.hlms_real.transpose(
                2, 1, 0).flatten().copy(),
            self.amp_phase_gen.lengths,
            (2 * self.num_modes + 1),
            self.num_bin_all,
            use_gpu=self.use_gpu,
        )
        """et = tttttt.perf_counter()

        print("splines", self.num_bin_all, (et - st) /
              num, (et - st) / num / self.num_bin_all)

        st = tttttt.perf_counter()
        for _ in range(num):"""
        # TODO: try single block reduction for likelihood (will probably be worse for smaller batch, but maybe better for larger batch)?
        template_channels = self.interp_response(
            self.dataTime,
            splines,
            self.amp_phase_gen.lengths,
            self.data_length,
            self.num_modes,
            self.amp_phase_gen.ells,
            self.amp_phase_gen.mms,
            self.num_bin_all,
            dt=1 / sampling_frequency,
        )

        """et = tttttt.perf_counter()
        print("interp", self.num_bin_all, (et - st) /
              num, (et - st) / num / self.num_bin_all)
        """
        template_channels = template_channels.reshape(
            self.num_bin_all, self.data_length)
        if return_type == "geocenter_td":
            return template_channels

        """st = tttttt.perf_counter()
        for _ in range(num):"""

        breakpoint()
        template_channels_fd_plus = self.xp.fft.rfft(
            template_channels.real, axis=-1) * 1 / sampling_frequency
        template_channels_fd_cross = self.xp.fft.rfft(
            template_channels.imag, axis=-1) * 1 / sampling_frequency

        if return_type == "geocenter_fd":
            return (template_channels_fd_plus, template_channels_fd_cross)

        Fplus, Fcross, time_shift = self.get_detector_information(
            ra, dec, geocent_time, psi)

        if return_type == "detector_td":
            raise NotImplementedError

        if return_type == "detector_fd":
            f = self.xp.fft.rfftfreq(
                self.data_length, 1.0 / sampling_frequency)
            signal_out = (Fplus.T[:, :, None] * template_channels_fd_plus[:, None, :] + Fcross.T[:, :, None] *
                          template_channels_fd_cross[:, None, :]) * self.xp.exp(-1j * 2. * np.pi * f[None, None, :] * time_shift.T[:, :, None])
            return signal_out

        template_channels_fd_plus = template_channels_fd_plus.flatten()
        template_channels_fd_cross = template_channels_fd_cross.flatten()

        num_threads_sum = 1024

        num_temps_per_bin = int(
            np.ceil((self.fd_data_length + num_threads_sum - 1) / num_threads_sum))

        temp_sums = self.xp.zeros(
            num_temps_per_bin * self.num_bin_all * self.nChannels, dtype=self.xp.complex128)
        dt = 1/sampling_frequency
        T = self.data_length * dt
        df = 1/T
        Fplus = Fplus.flatten()
        Fcross = Fcross.flatten()
        time_shift = time_shift.flatten()

        self.all_in_one_likelihood(
            temp_sums, template_channels_fd_plus, template_channels_fd_cross, self.data, self.psd, Fplus, Fcross, time_shift, df, self.num_bin_all, self.nChannels, self.fd_data_length
        )

        like = -1./2. * df * 4. * \
            temp_sums.reshape(-1, self.num_bin_all,
                              self.nChannels).sum(axis=(0, 2))

        """et = tttttt.perf_counter()
        print("final part", self.num_bin_all, (et - st) /
              num, (et - st) / num / self.num_bin_all)

        breakpoint()"""
        return like


class ODEWrapper:
    def __init__(self, use_gpu=False):
        if use_gpu:
            self.xp = xp
            self.ode = ODE_Ham_align_AD_gpu
        else:
            self.xp = np
            self.ode = ODE_Ham_align_AD_cpu
        self.count = 0

    def __call__(self, x, args, k, additionalArgs):
        self.count += 1
        reshape = k.shape
        numSys = reshape[-1]
        x_in = x.flatten()
        args_in = args.flatten()
        k_in = k.flatten()
        additionalArgs_in = additionalArgs.flatten()

        self.ode(x_in, args_in, k_in, additionalArgs_in, numSys)
        breakpoint()
        k[:] = k_in.reshape(reshape)

from scipy.special import gamma as scipy_gamma
from cupyx.scipy.special import gamma as cupy_gamma

def factorial(n, xp=np):
    
    gamma = scipy_gamma if xp == np else cupy_gamma
    return gamma(n + 1)

def factorial2(n, xp=np):
    if isinstance(n, int):
        squeeze = True
        n = np.atleast_1d(n)
    else:
        squeeze = False
    gamma = scipy_gamma if xp == np else cupy_gamma

    is_odd = (n % 2).astype(bool)

    out = xp.zeros_like(n, dtype=xp.float64)
    if xp.any(is_odd):
        out[is_odd] = gamma(n[is_odd]/2+1)*2**((n[is_odd]+1)/2)/xp.sqrt(xp.pi)
    if xp.any(~is_odd):
        out[~is_odd] = 2**(n[~is_odd]/2) * factorial(n[~is_odd]/2) 

    if squeeze:
        out = out[0]
    return out

def EOBComputeNewtonMultipolePrefixes(m1_in, m2_in, l_in, m_in, xp=np):
    m1_in = xp.atleast_1d(m1_in)
    m2_in = xp.atleast_1d(m2_in)
    l_in = xp.atleast_1d(l_in)
    m_in = xp.atleast_1d(m_in)

    m1 = m1_in[:, None]
    m2 = m2_in[:, None]
    l = l_in[None, :]
    m = m_in[None, :]

    totalMass = m1 + m2

    epsilon = (l + m) % 2
    
    x1 = (m1 / totalMass)
    x2 = (m2 / totalMass)

    eta = m1 * m2 / (totalMass * totalMass)
    sign_test = xp.abs(m % 2) == 0
    sign = 1 * (sign_test) + -1 * (~sign_test)

    #
    # Eq. 7 of Damour, Iyer and Nagar 2008.
    # For odd m, c is proportional to dM = m1-m2. In the equal-mass case, c = dM = 0.
    # In the equal-mass unequal-spin case, however, when spins are different, the odd m term is generally not zero.
    # In this case, c can be written as c0 * dM, while spins terms in PN expansion may take the form chiA/dM.
    # Although the dM's cancel analytically, we can not implement c and chiA/dM with the possibility of dM -> 0.
    # Therefore, for this case, we give numerical values of c0 for relevant modes, and c0 is calculated as
    # c / dM in the limit of dM -> 0. Consistently, for this case, we implement chiA instead of chiA/dM
    # in LALSimIMRSpinEOBFactorizedWaveform.c.

    test1 = (m1 != m2) | (sign == 1)

    c = (
        (test1) * (xp.power(x2, l + epsilon - 1) + sign * xp.power(x1, l + epsilon - 1))
        + (~test1 & (l == 2)) * -1.0
        + (~test1 & (l == 3)) * -1.0
        + (~test1 & (l == 4)) * -0.5
        + (~test1 & (l == 5)) * -0.5
        + (~test1 & (l > 5)) * 0.0
    ) 

    n = xp.zeros_like(l, dtype=xp.complex128)
    # Eqs 5 and 6. Dependent on the value of epsilon (parity), we get different n
    inds_0 = epsilon == 0
    if xp.any(inds_0):
        n[inds_0] = 1j * m[inds_0]
        n[inds_0] = n[inds_0] ** l[inds_0]

        mult1 = 8.0 * xp.pi / factorial2(2 * l[inds_0] + 1, xp=xp)
        mult2 = ((l[inds_0] + 1) * (l[inds_0] + 2)) / (l[inds_0] * (l[inds_0] - 1))
        mult2 = xp.sqrt(mult2)

        n[inds_0] *= mult1
        n[inds_0] *= mult2

    inds_1 = epsilon == 1
    if xp.any(inds_1):
        n[inds_1] = 1j * m[inds_1]
        n[inds_1] = n[inds_1] ** l[inds_1]
        n[inds_1] = -n[inds_1]

        mult1 = 16.0 * xp.pi / factorial2(2 * l[inds_1] + 1, xp=xp)
        
        mult2 = (2. * l[inds_1] + 1) * (l[inds_1] + 2) * (l[inds_1] * l[inds_1] - m[inds_1] * m[inds_1])
        mult2 /= (2. * l[inds_1] - 1) * (l[inds_1] + 1) * l[inds_1] * (l[inds_1] - 1)
        mult2 = xp.sqrt(mult2)

        n[inds_1] *= 1j * mult1
        n[inds_1] *= mult2

    prefix = n * eta * c
    return prefix


class SEOBNRv4PHM:
    def __init__(self, max_init_len=-1, use_gpu=False, ell_max=8, **kwargs):

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
            self.compute_hlms = compute_hlms_gpu
            self.root_find = root_find_all_gpu
            self.root_find_scalar = root_find_scalar_all_gpu

        else:
            self.xp = np
            self.compute_hlms = compute_hlms_cpu
            self.root_find = root_find_all_cpu
            self.root_find_scalar = root_find_scalar_all_cpu

        if max_init_len > 0:
            self.use_buffers = True
            raise NotImplementedError

        else:
            self.use_buffers = False

        self.ell_max = ell_max
        lm = []
        for l in range(2, ell_max + 1):
            for m in range(1, l + 1):
                lm.append([l, m])
        l, m = np.asarray(lm).T
        self.l_vals = self.xp.asarray(l, dtype=self.xp.int32)
        self.m_vals = self.xp.asarray(m, dtype=self.xp.int32)
        self.num_lm = np.sum(np.arange(2, self.ell_max + 1))

        # TODO: do we really need the (l, 0) modes
        self.allowable_modes = [
            # (2, 0),
            (2, 1),
            (2, 2),
            # (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            # (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            # (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            # (6, 0),
            (6, 2),
            (6, 4),
            (6, 6),
        ]
        self.ells_default = self.xp.array(
            [temp for (temp, _) in self.allowable_modes], dtype=self.xp.int32
        )

        self.mms_default = self.xp.array(
            [temp for (_, temp) in self.allowable_modes], dtype=self.xp.int32
        )

        self.nparams = 2

        #self.HTM_AC = HTMalign_AC()
        self.integrator = DOPR853(ODEWrapper(use_gpu=True), stopping_criterion=StoppingCriterion(
            True, read_out_to_cpu=False), tmax=1e7*1e6, max_step=300, use_gpu=True, read_out_to_cpu=False)  # self.use_gpu)  # use_gpu=use_gpu)

        self.num_args = 4
        self.num_add_args = 10
        self.num_add_args_total = self.num_add_args + 2 * self.num_lm 
        
        
    def _sanity_check_modes(self, ells, mms):
        for (ell, mm) in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes
                    )
                )

    def setup_add_args_array(self, m_1_scaled, m_2_scaled, chi_1, chi_2, omega0):

        prefixes = self.xp.zeros( (self.num_lm * self.num_bin_all,), dtype=self.xp.complex128)

        # transpose needed
        prefixes = EOBComputeNewtonMultipolePrefixes(
            m_1_scaled, m_2_scaled, self.l_vals, self.m_vals, xp=self.xp).T

        prefixes = prefixes.reshape(self.num_lm, self.num_bin_all)

        prefixes_new = self.xp.zeros((self.num_lm * 2, self.num_bin_all))
        prefixes_new[0::2] = prefixes.real
        prefixes_new[1::2] = prefixes.imag

        self.prefixes_new = prefixes_new
        
        # add args here contains the newtonian prefixes
        # 2 for re and im
        self.additionalArgs = self.xp.zeros(
            (self.num_add_args_total, self.num_bin_all))
        self.additionalArgs[0] = m_1_scaled
        self.additionalArgs[1] = m_2_scaled
        self.additionalArgs[2] = chi_1
        self.additionalArgs[3] = chi_2
        self.additionalArgs[4] = omega0
        self.additionalArgs[5] = 0.#3  # K
        self.additionalArgs[6] = 0.#4  # d5
        self.additionalArgs[7] = 0.#8  # dSO omega0
        self.additionalArgs[8] = 0.#6 # dSSomega0
        self.additionalArgs[9] = 0.#5  # h22_calib

        self.additionalArgs[10:] = prefixes_new

    def get_initial_conditions(
        self, m_1, m_2, chi_1, chi_2, fs=20.0, max_iter=1000, err=1e-12, **kwargs
    ):

        # TODO: constants from LAL
        mt = self.xp.asarray(m_1 + m_2)  # Total mass in solar masses
        omega0 = fs * (mt * MTSUN_SI * np.pi)
        m_1_scaled = self.xp.asarray(m_1) / mt
        m_2_scaled = self.xp.asarray(m_2) / mt
        chi_1 = self.xp.asarray(chi_1)
        chi_2 = self.xp.asarray(chi_2)
        dt = 1.0 / 16384 / (mt * MTSUN_SI)

        self.setup_add_args_array(m_1_scaled, m_2_scaled, chi_1, chi_2, omega0)
       
        additionalArgsIn = self.additionalArgs.copy().flatten()

        argsIn = self.xp.zeros((self.num_args, self.num_bin_all)).flatten()

        r_guess = omega0 ** (-2.0 / 3)
        # z = [r_guess, np.sqrt(r_guess)]
        
        # print(f"Initial guess is {z}")
        # The conservative bit: solve for r and pphi
        n = 2
        x0In = self.xp.zeros((n, self.num_bin_all))
        x0In[0] = r_guess
        x0In[1] = self.xp.sqrt(r_guess)

        xOut = self.xp.zeros_like(x0In).flatten()
        x0In = x0In.flatten()

        
        """
        from eob.pyEOB import RR_force
    
        force_out = self.xp.zeros((2 * self.num_bin_all))
        grad_out = self.xp.zeros((4 * self.num_bin_all))

        argsIn = self.xp.zeros((self.num_args, self.num_bin_all))
        argsIn[0, :] =  10.0
        argsIn[2, :] = 1.0
        argsIn[3, :] = 5.2
        argsIn = argsIn.flatten().copy()
        RR_force(force_out, grad_out, argsIn, additionalArgsIn, self.num_bin_all)
        breakpoint()
        
        
        from eob.pyEOB import IC_cons
        grad_out = self.xp.zeros((4 * self.num_bin_all))

        IC_cons(xOut, x0In, argsIn, additionalArgsIn, grad_out, self.num_bin_all)

        
        breakpoint()
        """
        self.root_find(
            xOut,
            x0In,
            argsIn,
            additionalArgsIn,
            max_iter,
            err,
            self.num_bin_all,
            n,
            self.num_args,
            self.num_add_args_total,
        )

        r0, pphi0 = xOut.reshape(n, self.num_bin_all)

        pr0 = self.xp.zeros(self.num_bin_all)
        start_bounds = self.xp.tile(
            self.xp.array([-3e-2, 0.0]), (self.num_bin_all, 1)
        ).T.flatten()

        argsIn = self.xp.zeros((self.num_args, self.num_bin_all))
        argsIn[0] = r0
        argsIn[3] = pphi0

        """
        argsIn = argsIn.flatten()
        grad_out = self.xp.zeros((4 * self.num_bin_all))
        grad_temp_force = self.xp.zeros((4 * self.num_bin_all))
        hess_out = self.xp.zeros((4 * 4 * self.num_bin_all))
        force_out = self.xp.zeros((2 * self.num_bin_all))
        pr = self.xp.full(self.num_bin_all, -0.01)
        out = self.xp.zeros(self.num_bin_all)
        from eob.pyEOB import IC_diss
        IC_diss(out, pr, argsIn, additionalArgsIn, grad_out, grad_temp_force, hess_out, force_out, self.num_bin_all)
        breakpoint()
        """
        additionalArgsIn = self.additionalArgs.copy().flatten()
        self.root_find_scalar(
            pr0,
            start_bounds,
            argsIn,
            additionalArgsIn,
            max_iter,
            err,
            self.num_bin_all,
            self.num_args,
            self.num_add_args_total,
        )
        return r0, pphi0, pr0

    def run_trajectory(self, r0, pphi0, pr0, m_1, m_2, chi_1, chi_2, fs=20.0, **kwargs):

        # TODO: constants from LAL
        mt = m_1 + m_2  # Total mass in solar masses
        omega0 = fs * (mt * MTSUN_SI * np.pi)
        m_1_scaled = m_1 / mt
        m_2_scaled = m_2 / mt
        dt = 1.0 / 16384 / (mt * MTSUN_SI)
        K = np.zeros_like(m_1_scaled)
        d5 = np.zeros_like(m_1_scaled)
        dSO = np.zeros_like(m_1_scaled)
        dSS = np.zeros_like(m_1_scaled)

        condBound = self.xp.array([r0, np.full_like(r0, 0.0), pr0, pphi0])

        argsData = additionalArgsIn = self.additionalArgs.copy()
        # TODO: make adjustable
        # TODO: debug dopr?
        breakpoint()
        t, traj, num_steps = self.integrator.integrate(
            condBound.copy(), argsData
        )

        num_steps = num_steps.astype(np.int32)

        num_steps_max = num_steps.max().item()

        return (
            (t[:num_steps_max, :] *
             self.xp.asarray(mt[self.xp.newaxis, :]) * MTSUN_SI).T,
            traj[:num_steps_max, :, :].transpose(2, 1, 0),
            num_steps,
        )

    def get_hlms(self, traj, m_1_full, m_2_full, chi_1, chi_2, num_steps, ells, mms):

        # TODO: check dimensionality (unit to 1?)
        m_1 = self.xp.asarray(m_1_full / (m_1_full + m_2_full))
        m_2 = self.xp.asarray(m_2_full / (m_1_full + m_2_full))
        chi_1 = self.xp.asarray(chi_1)
        chi_2 = self.xp.asarray(chi_2)
        r = traj[:, 0].flatten()
        phi = traj[:, 1].flatten()
        pr = traj[:, 2].flatten()
        L = traj[:, 3].flatten()

        num_steps_max = num_steps.max().item()

        hlms = self.xp.zeros(
            self.num_bin_all * self.num_modes * num_steps_max, dtype=self.xp.complex128
        )
        self.compute_hlms(
            hlms,
            r,
            phi,
            pr,
            L,
            m_1,
            m_2,
            chi_1,
            chi_2,
            num_steps,
            num_steps_max,
            ells,
            mms,
            self.num_modes,
            self.num_bin_all,
        )

        return hlms.reshape(self.num_bin_all, self.num_modes, num_steps_max)

    @property
    def hlms(self):
        num_modes = (self.hlms_real.shape[1] - 1) / 2
        assert (self.hlms_real.shape[1] % 2) == 1

        hlms_real = self.hlms_real[:, 0:num_modes]
        hlms_imag = self.hlms_real[:, num_modes:2 * num_modes]
        return hlms_real + 1j * hlms_imag

    def __call__(
        self,
        m1,
        m2,
        # chi1x,
        # chi1y,
        chi1z,
        # chi2x,
        # chi2y,
        chi2z,
        distance,
        phiRef,
        modes=None,
        fs=20.0,  # Hz
    ):
        if modes is not None:
            ells = self.xp.asarray(
                [ell for ell, mm in modes], dtype=self.xp.int32)
            mms = self.xp.asarray(
                [mm for ell, mm in modes], dtype=self.xp.int32)

            self._sanity_check_modes(ells, mms)

        else:
            ells = self.ells_default
            mms = self.mms_default

        self.num_modes = len(ells)

        self.num_bin_all = len(m1)

        r0, pphi0, pr0 = self.get_initial_conditions(
            m1, m2, chi1z, chi2z, fs=fs)

        t, traj, num_steps = self.run_trajectory(
            r0, pphi0, pr0, m1, m2, chi1z, chi2z, fs=fs
        )
        self.traj = traj

        distance = self.xp.asarray(distance)
        hlms = self.get_hlms(traj, m1, m2, chi1z, chi2z,
                             num_steps, ells, mms) / distance[:, None, None]

        phi = traj[:, 1]

        self.lengths = num_steps.astype(self.xp.int32)
        num_steps_max = num_steps.max()
        self.t = t[:, :num_steps_max]
        self.hlms_real = self.xp.concatenate(
            [
                hlms[:, :num_steps_max].real,
                hlms[:, :num_steps_max].imag,
                phi[:, self.xp.newaxis, :num_steps_max],
            ],
            axis=1,
        )

        self.ells = ells
        self.mms = mms


if __name__ == "__main__":
    from cupy.cuda.runtime import setDevice
    setDevice(2)
    eob = SEOBNRv4PHM(use_gpu=True)  # gpu_available)

    mt = 40.0  # Total mass in solar masses
    q = 2.5
    num = int(1e0)
    m1 = np.full(num, mt * q / (1.+q))
    m2 = np.full(num, mt / (1.+q))
    # chi1x,
    # chi1y,
    chi1z = np.full(num, 0.0001)
    # chi2x,
    # chi2y,
    chi2z = np.full(num, 0.0001)
    from bbhx.utils.constants import PC_SI
    distance = np.full(num, 100.0) * 1e6 * PC_SI  # Mpc -> m
    phiRef = np.full(num, 0.0)
    inc = np.full(num, np.pi / 3.0)
    ra = np.full(num, np.pi / 4.0)
    dec = np.full(num, np.pi / 5.0)
    psi = np.full(num, np.pi / 6.0)
    geocent_time = np.full(num, np.pi / 7.0)

    # eob(m1, m2, chi1z, chi2z, distance, phiRef)
    bbh = BBHWaveformTD(use_gpu=True)
    import time
    n = 1
    st = time.perf_counter()
    for jj in range(n):
        """
        eob(
                m1,
                m2,
                # chi1x,
                # chi1y,
                chi1z,
                # chi2x,
                # chi2y,
                chi2z,
                distance,
                phiRef,
                #modes=modes,
                #fs=fs,
            )
        """
        out = bbh(
            m1,
            m2,
            # chi1x,
            # chi1y,
            chi1z,
            # chi2x,
            # chi2y,
            chi2z,
            distance,
            phiRef,
            inc,
            ra,
            dec,
            psi,
            geocent_time,
            sampling_frequency=1024.,
            Tobs=5.0,
            modes=None,
            bufferSize=None,
            fill=False,
        )
        print(jj)
    et = time.perf_counter()
    print((et - st)/n/num, "done")

    breakpoint()
