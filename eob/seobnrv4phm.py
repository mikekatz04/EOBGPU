from .pytdinterp_cpu import interpolate_TD_wrap as interpolate_TD_wrap_cpu
from .pytdinterp_cpu import TDInterp_wrap2 as TDInterp_wrap_cpu
from .pyEOB_cpu import ODE_Ham_align_AD as ODE_Ham_align_AD_cpu
from .pyEOB_cpu import ODE as ODE_cpu
from .pyEOB_cpu import root_find_scalar_all as root_find_scalar_all_cpu
from .pyEOB_cpu import root_find_all as root_find_all_cpu
#from .pyEOB_cpu import compute_hlms as compute_hlms_cpu
import numpy as np
import warnings
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
    #from .pyEOB import compute_hlms as compute_hlms_gpu
    from .pyEOB import root_find_all as root_find_all_gpu
    #from .pyEOB import root_find_scalar_all as root_find_scalar_all_gpu
    # from .pyEOB import ODE as ODE_gpu
    from .pyEOB import evaluate_Ham_align_AD as evaluate_Ham_align_AD_gpu
    from .pyEOB import grad_Ham_align_AD as grad_Ham_align_AD_gpu
    from .pyEOB import SEOBNRv5Class as SEOBNRv5Class_gpu
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
    def __init__(self, eob_c_class, max_step, use_gpu=False, read_out_to_cpu=False):
        self.use_gpu = use_gpu
        self.read_out_to_cpu = read_out_to_cpu
        self.eob_c_class = eob_c_class
        self.max_step = max_step

        self.old_omegas = None

        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

    def setup(self, numSys):
        self.numSys = numSys
        self.old_omegas = self.xp.zeros((self.max_step, numSys))

    def reset(self):
        self.old_omegas = None

    def __call__(self, step_num, denseOutput, additionalArgs, index_update):

        if self.read_out_to_cpu:
            raise NotImplementedError
            
        nargs, numSys_here = denseOutput.shape
        
        args = denseOutput.flatten()
        additionalArgsIn = additionalArgs.flatten()
        derivs = self.xp.zeros((nargs, numSys_here)).flatten()
        x = self.xp.zeros(1)
        self.eob_c_class.ODE_Ham_align_AD(x, args, derivs, additionalArgsIn, numSys_here)

        derivs = derivs.reshape(nargs, numSys_here)

        r = denseOutput[0]
        stop = self.xp.zeros_like(r, dtype=int)

        drdt = derivs[0]
        omega = derivs[1]
        dprdt = derivs[2]

        if self.old_omegas is None:
            raise ValueError("Must use `setup` attribute before using this method.")

        # if np.isnan(np.min(y)) or np.isnan(np.min(derivs)):
        #     return 1

        inds_remaining = np.ones_like(r, dtype=bool)

        stop[inds_remaining] += 2 * (r[inds_remaining] < 1.0)
        inds_remaining = ~(stop.astype(bool))

        stop[inds_remaining] += 3 * ((r[inds_remaining] < 6.0) & (drdt[inds_remaining] > 0.0))
        inds_remaining = ~(stop.astype(bool))

        stop[inds_remaining] += 4 * ((r[inds_remaining] < 6.0) & (dprdt[inds_remaining] > 0.0))
        inds_remaining = ~(stop.astype(bool))

        inds_here = inds_remaining * (step_num > 5)
        if self.xp.any(inds_here):
            num_still_here = self.xp.sum(inds_here).item()

            get_inds = (step_num[inds_here, None] + self.xp.tile(self.xp.arange(-4, 0), (num_still_here, 1))).flatten()

            still_have = self.xp.arange(len(inds_here))[inds_here]
            
            first_check = r[inds_here] < 6.0
            second_check = omega[inds_here] < self.old_omegas[(step_num - 1, still_have)]

            close_old_omegas = self.old_omegas[(get_inds, self.xp.repeat(still_have, 4))].reshape(num_still_here, 4)
            third_check = self.xp.all(self.xp.diff(close_old_omegas, axis=1) < 0.0, axis=1)
            
            stop[inds_here] += 5 * (first_check) * (second_check) * (third_check)

        self.old_omegas[(step_num, index_update)] = omega
   
        """if self.use_gpu and :
            stop = denseOutput[(step_num.get(), np.zeros_like(
                step_num.get()), np.arange(len(step_num)))] < 3.0
        else:
            stop = denseOutput[(step_num, np.zeros_like(
                step_num), np.arange(len(step_num)))] < 3.0"""
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

        ninterps = self.ninterps = nsubs * num_bin_all
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

    def _get_inds(self, tnew):
        # find were in the old t array the new t values split

        inds = self.xp.zeros((self.num_bin_all, tnew.shape[1]), dtype=int)

        # Optional TODO: remove loop ? if speed needed
        for i, (t, tnew_i) in enumerate(zip(self.t_shaped, tnew)):
            inds[i] = self.xp.searchsorted(t, tnew_i, side="right") - 1

            # fix end value
            inds[i][tnew_i == t[-1]] = len(t) - 2

        # get values outside the edges
        inds_bad_left = tnew < self.t_shaped[:, 0][:, None]
        inds_bad_right = tnew > self.t_shaped[:, -1][:, None]

        if self.xp.any(inds < 0) or self.xp.any(inds >= self.t_shaped.shape[1]):
            warnings.warn(
                "New t array outside bounds of input t array. These points are filled with edge values."
            )
        return inds, inds_bad_left, inds_bad_right

    def __call__(self, tnew, deriv_order=0):
        """Evaluation function for the spline

        Put in an array of new t values at which all interpolants will be
        evaluated. If t values are outside of the original t values, edge values
        are used to fill the new array at these points.

        args:
            tnew (1D or 2D double xp.ndarray): Array of new t values. All of these new
                t values must be within the bounds of the input t values,
                including the beginning t and **excluding** the ending t. If tnew is 1D
                and :code:`self.t` is 2D, tnew will be cast to 2D.
            deriv_order (int, optional): Order of the derivative to evaluate. Default
                is 0 meaning the basic spline is evaluated. deriv_order of 1, 2, and 3
                correspond to their respective derivatives of the spline. Unlike :code:`scipy`,
                this is purely an evaluation of the derivative values, not a new class to evaluate
                for the derivative.

        raises:
            ValueError: a new t value is not in the bounds of the input t array.

        returns:
            xp.ndarray: 1D or 2D array of evaluated spline values (or derivatives).

        """

        tnew = self.xp.atleast_1d(tnew)

        if tnew.ndim == 2:
            if tnew.shape[0] != self.t_shaped.shape[0]:
                raise ValueError(
                    "If providing a 2D tnew array, must have some number of interpolants as was entered during initialization."
                )

        # copy input to all splines
        elif tnew.ndim == 1:
            tnew = self.xp.tile(tnew, (self.t.shape[0], 1))

        tnew = self.xp.atleast_2d(tnew)

        # get indices into spline
        inds, inds_bad_left, inds_bad_right = self._get_inds(tnew)

        # x value for spline

        # indexes for which spline
        inds0 = self.xp.tile(self.xp.arange(self.num_bin_all), (tnew.shape[1], 1)).T

        t_here = self.t_shaped[(inds0.flatten(), inds.flatten())].reshape(
            self.num_bin_all, tnew.shape[1]
        )

        x = tnew - t_here
        x2 = x * x
        x3 = x2 * x

        inds0 = self.xp.repeat(inds0.flatten(), self.nsubs)
        inds = self.xp.repeat(inds.flatten(), self.nsubs)

        inds_subs = self.xp.tile(self.xp.arange(self.nsubs), (self.num_bin_all * tnew.shape[1], 1)).flatten()
        # get spline coefficients
        y = self.y_shaped[(inds0.flatten(), inds_subs, inds.flatten())].reshape(
            self.num_bin_all, self.nsubs, tnew.shape[1]
        )

        c1 = self.c1_shaped[(inds0.flatten(), inds_subs, inds.flatten())].reshape(
            self.num_bin_all, self.nsubs, tnew.shape[1]
        )
        c2 = self.c2_shaped[(inds0.flatten(), inds_subs, inds.flatten())].reshape(
            self.num_bin_all, self.nsubs, tnew.shape[1]
        )
        c3 = self.c3_shaped[(inds0.flatten(), inds_subs, inds.flatten())].reshape(
            self.num_bin_all, self.nsubs, tnew.shape[1]
        )

        # evaluate spline
        if deriv_order == 0:
            out = y + c1 * x[:, None, :] + c2 * x2[:, None, :] + c3 * x3[:, None, :]
            # fix bad values
            if self.xp.any(inds_bad_left):
                raise ValueError(
                    "x points outside of the domain of the spline are not supported when taking derivatives."
                )
                temp = self.xp.tile(self.y_shaped[:, 0], (tnew.shape[1], 1)).T
                out[inds_bad_left] = temp[inds_bad_left]

            if self.xp.any(inds_bad_right):
                raise ValueError(
                    "x points outside of the domain of the spline are not supported when taking derivatives."
                )
                temp = self.xp.tile(self.y_shaped[:, 0], (tnew.shape[1], 1)).T
                out[inds_bad_left] = temp[inds_bad_left]
                temp = self.xp.tile(self.y_shaped[:, -1], (tnew.shape[1], 1)).T
                out[inds_bad_right] = temp[inds_bad_right]

        else:
            # derivatives
            if self.xp.any(inds_bad_right) or self.xp.any(inds_bad_left):
                raise ValueError(
                    "x points outside of the domain of the spline are not supported when taking derivatives."
                )
            if deriv_order == 1:
                out = c1 + 2 * c2 * x[:, None, :] + 3 * c3 * x2[:, None, :]
            elif deriv_order == 2:
                out = 2 * c2 + 6 * c3 * x[:, None, :]
            elif deriv_order == 3:
                out = 6 * c3
            else:
                raise ValueError("deriv_order must be within 0 <= deriv_order <= 3.")

        return out.squeeze()


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
            self.num_modes = len(self.amp_phase_gen.allowable_modes_hlms)
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

"""
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
"""
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

def EOBGetNRSpinPeakDeltaTv4(ell, m, m1, m2, chi1, chi2):

    eta = m1 * m2 / (m1 + m2) / (m1 + m2)
    chi = 0.5 * (chi1 + chi2) + 0.5 * (chi1 - chi2) * (m1 - m2) / (m1 + m2) / (
        1.0 - 2.0 * eta
    )
    eta2 = eta * eta
    eta3 = eta2 * eta
    chiTo2 = chi * chi
    chiTo3 = chiTo2 * chi

    # Calibrationv21_Sep8a
    coeff00 = 2.50499
    coeff01 = 13.0064
    coeff02 = 11.5435
    coeff03 = 0
    coeff10 = 45.8838
    coeff11 = -40.3183
    coeff12 = 0
    coeff13 = -19.0538
    coeff20 = 13.0879
    coeff21 = 0
    coeff22 = 0
    coeff23 = 0.192775
    coeff30 = -716.044
    coeff31 = 0
    coeff32 = 0
    coeff33 = 0
    res = (
        coeff00
        + coeff01 * chi
        + coeff02 * chiTo2
        + coeff03 * chiTo3
        + coeff10 * eta
        + coeff11 * eta * chi
        + coeff12 * eta * chiTo2
        + coeff13 * eta * chiTo3
        + coeff20 * eta2
        + coeff21 * eta2 * chi
        + coeff22 * eta2 * chiTo2
        + coeff23 * eta2 * chiTo3
        + coeff30 * eta3
        + coeff31 * eta3 * chi
        + coeff32 * eta3 * chiTo2
        + coeff33 * eta3 * chiTo3
    )

    # RC: for the 55 mode the attachment is done at tpeak22 -10M, note that since here deltat22 is defined as -deltat22 with respect
    # to SEOBNRv4 paper, here I need to add 10 M
    if (ell == 5) and (m == 5):
        res = res + 10.0

    return res



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
            self.root_find = root_find_all_gpu
            #self.root_find_scalar = root_find_scalar_all_gpu
            self.eob_c_class = SEOBNRv5Class_gpu()

        else:
            self.xp = np
            self.root_find = root_find_all_cpu
            #self.root_find_scalar = root_find_scalar_all_cpu

        if max_init_len > 0:
            self.use_buffers = True
            raise NotImplementedError

        else:
            self.use_buffers = False

        self.allowable_modes_hlms = [(2,2),(2,1),(3,3),(4,4),(5,5)]
        
        self.inds_keep_newtonian_prefixes_hlms  = self.xp.zeros(len(self.allowable_modes_hlms), dtype=int)

        self.ell_max = ell_max
        lm = []
        ind_temp = 0
        for l in range(2, ell_max + 1):
            for m in range(1, l + 1):
                lm.append([l, m])
                if (l,m) in self.allowable_modes_hlms:
                    ind_hlm_temp = self.allowable_modes_hlms.index((l, m))
                    self.inds_keep_newtonian_prefixes_hlms[ind_hlm_temp] = ind_temp

                ind_temp += 1
        l, m = np.asarray(lm).T
        self.l_vals = self.xp.asarray(l, dtype=self.xp.int32)
        self.m_vals = self.xp.asarray(m, dtype=self.xp.int32)
        self.num_lm = np.sum(np.arange(2, self.ell_max + 1))

        # TODO: do we really need the (l, 0) modes
        
        self.ells_default = self.xp.array(
            [temp for (temp, _) in self.allowable_modes_hlms], dtype=self.xp.int32
        )
        self.mms_default = self.xp.array(
            [temp for (_, temp) in self.allowable_modes_hlms], dtype=self.xp.int32
        )
    
        self.nparams = 2
        max_step = 400
        #self.HTM_AC = HTMalign_AC()
        self.integrator = DOPR853(self.eob_c_class, stopping_criterion=StoppingCriterion(self.eob_c_class, max_step, use_gpu=True, read_out_to_cpu=False), tmax=1e7*1e6, max_step=max_step, use_gpu=True, read_out_to_cpu=False)  # self.use_gpu)  # use_gpu=use_gpu)

        self.num_args = 4
        self.num_add_args = 10
        self.num_add_args_total = self.num_add_args + 2 * self.num_lm     

        self.K_val = 0.0
        self.d5_val = 0.0
        self.dSO_val = 0.0
        self.dSS_val = 0.0
        self.h22_calib = 0.0
        
    def _sanity_check_modes(self, ells, mms):
        for (ell, mm) in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes_hlms:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes_hlms
                    )
                )

    def setup_class(self, m_1_scaled, m_2_scaled, chi_1, chi_2, omega0):

        eta = m_1_scaled * m_2_scaled / (m_1_scaled + m_2_scaled) ** 2
        chi_S = (chi_1 + chi_2) / 2
        chi_A = (chi_1 - chi_2) / 2
        tplspin = (1 - 2 * eta) * chi_S + (m_1_scaled - m_2_scaled) / (
            m_1_scaled + m_2_scaled
        ) * chi_A

        # transpose needed
        prefixes = EOBComputeNewtonMultipolePrefixes(
            m_1_scaled, m_2_scaled, self.l_vals, self.m_vals, xp=self.xp).T

        prefixes = prefixes.reshape(self.num_lm, self.num_bin_all)

        self.prefixes_hlms = prefixes[self.inds_keep_newtonian_prefixes_hlms, :]

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
        self.additionalArgs[5] = self.K_val  # K
        self.additionalArgs[6] = self.d5_val # d5
        self.additionalArgs[7] = self.dSO_val  # dSO omega0
        self.additionalArgs[8] = self.dSS_val # dSSomega0
        self.additionalArgs[9] = self.h22_calib # h22_calib

        self.additionalArgs[10:] = prefixes_new
        
        use_hm = 1

        self.eob_c_class.update_information(m_1_scaled, m_2_scaled, eta, tplspin, chi_S, chi_A, use_hm, self.num_bin_all, self.prefixes_hlms)

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
        self.eob_c_class.root_find_scalar_all(
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

    def run_trajectory(self, r0, pphi0, pr0, m_1, m_2, chi_1, chi_2, fs=20.0, step_back=10.0, fine_step=0.05, **kwargs):

        # TODO: constants from LAL
        mt = self.xp.asarray(m_1 + m_2)  # Total mass in solar masses
        m_1 = self.xp.asarray(m_1)
        m_2 = self.xp.asarray(m_2)
        m_1_scaled = m_1 / mt
        m_2_scaled = m_2 / mt
        dt = 1.0 / 16384 / (mt * MTSUN_SI)
        K = np.full_like(m_1_scaled, self.K_val)
        d5 = np.full_like(m_1_scaled, self.d5_val)
        dSO = np.full_like(m_1_scaled, self.dSO_val)
        dSS = np.full_like(m_1_scaled, self.dSS_val)

        condBound = self.xp.array([r0, np.full_like(r0, 0.0), pr0, pphi0])

        argsData = additionalArgsIn = self.additionalArgs.copy()
        # TODO: make adjustable
        # TODO: debug dopr?
        t, traj, num_steps = self.integrator.integrate(
            condBound.copy(), argsData
        )

        """
        # steps are axis 0
        last_ind = self.xp.where(self.xp.diff(t[1:].astype(bool).astype(int), axis=0) == -1)

        t_stop = t[last_ind]

        # The starting point of fine integration
        t_desired = t_stop - step_back

        idx_restart = self.xp.argmin(self.xp.abs(t - t_desired[None, :]), axis=0)

        prep_inds = self.xp.tile(self.xp.arange(t.shape[0]), (self.num_bin_all, 1)).T

        # walk back
        t[(prep_inds >= idx_restart[None, :])] = 0.0
        traj[self.xp.tile((prep_inds[:, None, :] >= idx_restart[None, None, :]), (1, 4, 1))] = 0.0

        step_num = idx_restart - 1

        t, traj, num_steps = self.integrator.integrate(
            condBound.copy(), argsData,
            step_num=step_num,
            denseOutput=traj,
            denseOutputLoc=t,
            hInit=fine_step,
            fix_step=True,
        )
        # TODO: check integrator difference if needed
        """
        num_steps = num_steps.astype(np.int32)

        num_steps_max = num_steps.max().item()

        return (
            (t[:num_steps_max, :] *
             self.xp.asarray(mt[self.xp.newaxis, :]) * MTSUN_SI).T,
            traj[:num_steps_max, :, :].transpose(2, 1, 0),
            num_steps,
        )

    def get_hlms(self, traj, m_1_full, m_2_full, chi_1, chi_2, num_steps, ells, mms):

        m_1 = self.xp.asarray(m_1_full / (m_1_full + m_2_full))
        m_2 = self.xp.asarray(m_2_full / (m_1_full + m_2_full))
        chi_1 = self.xp.asarray(chi_1)
        chi_2 = self.xp.asarray(chi_2)
        r = traj[:, 0]
        phi = traj[:, 1]
        pr = traj[:, 2]
        pphi = L = traj[:, 3]

        numSys_eff = int(np.prod(r.shape))
        numSys = r.shape[0]
        traj_length = r.shape[1]

        M = m_1 + m_2
        mu = m_1 * m_2 / M
        nu = mu / M


        additionalArgsIn = self.xp.tile(self.additionalArgs, (1, traj_length))

        argsIn = self.xp.asarray([r.flatten(), phi.flatten(), pr.flatten(), pphi.flatten()]).flatten()
        gradient = self.xp.zeros((numSys_eff * self.num_args,))
        grad_Ham_align_AD_gpu(argsIn, gradient, additionalArgsIn, numSys_eff)
        
        argsIn_circ = self.xp.asarray([r.flatten(), phi.flatten(), self.xp.zeros_like(pphi).flatten(), pphi.flatten()]).flatten()
        gradient_circ = self.xp.zeros((numSys_eff * self.num_args,))
        grad_Ham_align_AD_gpu(argsIn_circ, gradient_circ, additionalArgsIn, numSys_eff)
        gradient_circ = gradient_circ.reshape((self.num_args, numSys, traj_length))
        gradient = gradient.reshape((self.num_args, numSys, traj_length))

        omega_circ = gradient_circ[3]
        omega = gradient[3]
        
        vPhi = 1/(omega_circ**2*r**3)
        H_val = self.xp.zeros(numSys_eff)
        
        evaluate_Ham_align_AD_gpu(
            H_val, 
            r.flatten(), 
            phi.flatten(), 
            pr.flatten(), 
            pphi.flatten(), 
            self.xp.repeat(m_1, traj_length), 
            self.xp.repeat(m_2, traj_length), 
            self.xp.repeat(chi_1, traj_length), 
            self.xp.repeat(chi_2, traj_length), 
            self.K_val, self.d5_val, self.dSO_val, self.dSS_val, numSys_eff)

        H_val = nu[:, None] * H_val.reshape(numSys, traj_length)

        v = omega**(1./3)
        # Needed for hCoeffs
        chiS = 0.5 * (chi_1 + chi_2)
        chiA = 0.5 * (chi_1 - chi_2)
        tplspin = (1.0 - 2.0 * nu) * chiS + (m_1 - m_2) / (m_1 + m_2) * chiA
    
        h22_calib = self.h22_calib
        modes = self.xp.zeros((numSys, self.num_modes, traj_length), dtype=self.xp.complex128)
        tmp = self.xp.zeros((numSys * traj_length), dtype=self.xp.complex128)
        for mode_i, (ell, mm) in enumerate(zip(ells, mms)):
            # TODO: need to put in varying amount of trajectory length?
            newtonian_prefixes = self.prefixes_hlms[mode_i, :].copy()
            self.eob_c_class.EOBGetSpinFactorizedWaveform(
                tmp,
                r.flatten().copy(),
                phi.flatten().copy(),
                pr.flatten().copy(),
                pphi.flatten().copy(),
                v.flatten(),
                H_val.flatten(),
                nu.flatten(),
                vPhi.flatten(),
                int(ell),
                int(mm),
                newtonian_prefixes,
                h22_calib,
                numSys,
                traj_length 
            )
            modes[:, mode_i, :] = tmp.reshape(numSys, traj_length)

        """
        
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

        additionalArgsIn = self.additionalArgs.copy().flatten()
        
        self.eob_c_class.compute_hlms(
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
            additionalArgsIn,
            self.num_add_args,  # NOT num_add_args_total
        )
        breakpoint()
        """
        return modes

    @property
    def hlms(self):
        num_modes = (self.hlms_real.shape[1] - 1) / 2
        assert (self.hlms_real.shape[1] % 2) == 1

        hlms_real = self.hlms_real[:, 0:num_modes]
        hlms_imag = self.hlms_real[:, num_modes:2 * num_modes]
        return hlms_real + 1j * hlms_imag

    def compute_full_waveform(self, 
                dynamics,
                hlm_interp,
                t_interp,
                m_1,
                m_2,
                chi_1,
                chi_2,
                M,
                ells,
                mms,
                sampling_frequency=1024.0,
                nrDeltaT=None,
            ):

        dt = 1 / sampling_frequency

        delta_T = dt / (M * MTSUN_SI)
        
        t_in = (self.t / (M[:, None] * MTSUN_SI))

        phi_orb = dynamics[:, 1]
        phi_spline = CubicSplineInterpolantTD(
            t_in.T.flatten().copy(),
            phi_orb.T.flatten().copy(),
            self.lengths,
            1,
            self.num_bin_all,
            use_gpu=self.use_gpu,
        )

        r_pr_spline = CubicSplineInterpolantTD(
            t_in.T.flatten().copy(),
            dynamics[:, np.array([0, 2])].transpose((2, 1, 0)).flatten().copy(),
            self.lengths,
            2,
            self.num_bin_all,
            use_gpu=self.use_gpu,
        )
        num_pts_new = ((t_in[:, -1] - t_in[:, 0]) / delta_T).astype(int) + 1
        max_num_pts = num_pts_new.max().item()
 
        t_new = self.xp.tile(self.xp.arange(max_num_pts), (dynamics.shape[0], 1)) * delta_T[:, None] + t_in[:, 0][:, None]

        inds_bad = t_new > t_in.max(axis=1)[:, None]
        t_new[inds_bad] = self.xp.tile(t_in.max(axis=1)[:, None], (1, max_num_pts))[inds_bad]
        omega_orb_mine = phi_spline(t_new, deriv_order=1)
        phi_orb_interp = phi_spline(t_new, deriv_order=0)

        idx_omega_peak = self.xp.argmax(omega_orb_mine, axis=1)
        t_omega_peak = t_new[(self.xp.arange(self.num_bin_all), idx_omega_peak)]

        tmp = hlm_interp * self.xp.exp(1j * mms[None, :, None] * phi_orb[:, None, :])
        hlms_real = self.xp.concatenate(
            [
                hlm_interp.real,
                hlm_interp.imag,
            ],
            axis=1,
        )

        splines = CubicSplineInterpolantTD(
            t_in.T.flatten().copy(),
            hlms_real.transpose(
                2, 1, 0).flatten().copy(),
            self.lengths,
            2 * self.num_modes,
            self.num_bin_all,
            use_gpu=self.use_gpu,
        )

        result = splines(t_new)

        modes = result[:, 0::2, :] + 1j * result[:, 1::2, :]
        modes *= self.xp.exp(-1j * mms[None, :, None] * phi_orb_interp[:, None, :])

        if nrDeltaT is None:
            nrDeltaT = EOBGetNRSpinPeakDeltaTv4(2, 2, m_1, m_2, chi_1, chi_2)

        t_attach = t_omega_peak - nrDeltaT
        
        tmp3 = r_pr_spline(t_new)
        r_new = tmp3[:, 0]
        pr_new = tmp3[:, 1]

        amp = self.xp.abs(modes)
        phase = self.xp.unwrap(self.xp.angle(modes))

        if (
            m % 2
            and np.abs(m_1 - m_2) < 1e-4
            and np.abs(chi_1) < 1e-4
            and np.abs(chi_2) < 1e-4
        ) or (m % 2 and np.abs(m_1 - m_2) < 1e-4 and np.abs(chi_1 - chi_2) < 1e-4):
            continue

        breakpoint()
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
        nrDeltaT=None,
        sampling_frequency=16384.0,
    ):

        if nrDeltaT is None:
            self.nrDeltaT = 0.0

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

        # TODO: constants from LAL
        # TODO: remove this happening multiple times?
        M = self.xp.asarray(m1 + m2)  # Total mass in solar masses
    
        m_1_scaled = self.xp.asarray(m1) / M
        m_2_scaled = self.xp.asarray(m2) / M
        chi1z = self.xp.asarray(chi1z)
        chi2z = self.xp.asarray(chi2z)
        omega0 = fs * (M * MTSUN_SI * np.pi)

        self.setup_class(m_1_scaled, m_2_scaled, chi1z, chi2z, omega0)

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
            ],
            axis=1,
        )
        self.compute_full_waveform(
            traj,
            hlms,
            self.t,
            m_1_scaled,
            m_2_scaled,
            chi1z,
            chi2z,
            M,
            ells,
            mms,
            sampling_frequency=sampling_frequency,
            nrDeltaT=None
        )
        self.ells = ells
        self.mms = mms

        self.eob_c_class.deallocate_information()
        


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
            sampling_frequency=16384.,
            Tobs=5.0,
            modes=None,
            bufferSize=None,
            fill=False,
        )
        print(jj)
    et = time.perf_counter()
    print((et - st)/n/num, "done")

    breakpoint()
