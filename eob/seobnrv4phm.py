"""from .pytdinterp_cpu import interpolate_TD_wrap as interpolate_TD_wrap_cpu
from .pytdinterp_cpu import TDInterp_wrap2 as TDInterp_wrap_cpu
from .pyEOB_cpu import ODE_Ham_align_AD as ODE_Ham_align_AD_cpu
from .pyEOB_cpu import ODE as ODE_cpu
from .pyEOB_cpu import root_find_scalar_all as root_find_scalar_all_cpu
from .pyEOB_cpu import root_find_all as root_find_all_cpu
#from .pyEOB_cpu import compute_hlms as compute_hlms_cpu"""
import numpy as np
import warnings
from bbhx.utils.constants import *

import sys

MTSUN_SI = 4.925491025543576e-06

import matplotlib.pyplot as plt
from bilby.gw.utils import greenwich_mean_sidereal_time
from bilby.core.utils import speed_of_light

import sys

#from odex.pydopr853 import DOPR853
from .new_Dormand_horiz import DOPR853
from .ode_for_test import ODE_1

try:
    import cupy as xp
    import cupy as cp
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
from .eobstuff import *

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

        self.eob_c_class.ODE_Ham_align_AD(x, args, derivs, additionalArgsIn, numSys_here, index_update.astype(self.xp.int32))

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

        inds_here = inds_remaining * (step_num[index_update] > 5)
        if self.xp.any(inds_here):
            num_still_here = self.xp.sum(inds_here).item()
            get_inds = (step_num[index_update][inds_here, None] + self.xp.tile(self.xp.arange(-4, 0), (num_still_here, 1))).flatten()

            still_have = self.xp.arange(len(inds_here))[inds_here]
            
            first_check = r[inds_here] < 6.0
            
            second_check = omega[inds_here] < self.old_omegas[(step_num[index_update[still_have]] - 1, index_update[still_have])]

            close_old_omegas = self.old_omegas[(get_inds, self.xp.repeat(still_have, 4))].reshape(num_still_here, 4)
            third_check = self.xp.all(self.xp.diff(close_old_omegas, axis=1) < 0.0, axis=1)
            
            stop[inds_here] += 5 * (first_check) * (second_check) * (third_check)

        else:
            still_have = self.xp.ones_like(index_update, dtype=bool)

        check_old = self.old_omegas.copy()
        self.old_omegas[(step_num[index_update[still_have]], index_update[still_have])] = omega[still_have]
        
        """if self.use_gpu and :
            stop = denseOutput[(step_num.get(), np.zeros_like(
                step_num.get()), np.arange(len(step_num)))] < 3.0
        else:
            stop = denseOutput[(step_num, np.zeros_like(
                step_num), np.arange(len(step_num)))] < 3.0"""
        return stop

def searchsorted2d_vec(a,b, xp=None, **kwargs):
    if xp is None:
        xp = np
    m,n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num*xp.arange(a.shape[0])[:,None]
    p = xp.searchsorted( (a+r).ravel(), (b+r).ravel(), **kwargs).reshape(m,-1)
    return p - n*(xp.arange(m)[:,None])

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
            x, y_all, B, upper_diag, diag, lower_diag, self.lengths, num_bin_all, nsubs,
        )
        # TODO: need to fix last point
        self.x = x

        self.t_shaped = self.x.reshape(self.max_length, self.num_bin_all).T
        max_t_vals = self.t_shaped.max(axis=1)
        inds_for_fix = self.xp.tile(self.xp.arange(self.max_length), (self.num_bin_all,1))
        inds_fix = np.where(inds_for_fix >= self.lengths[:, None])
        if len(inds_fix[0]) > 0:
            # need to fix this for the search sorted algorithm
            # fixes different length time arrays
            self.t_shaped[inds_fix] = max_t_vals[inds_fix[0]] * 2.0


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

        """import time
        st = time.perf_counter()"""
        # Optional TODO: remove loop ? if speed needed
        inds = searchsorted2d_vec(self.t_shaped, tnew, side="right", xp=self.xp) - 1
        inds[inds == self.lengths[:, None]] -= 1
        """et = time.perf_counter()
        print((et - st))
        breakpoint()
        for i, (t, tnew_i, length_i) in enumerate(zip(self.t_shaped, tnew, self.lengths)):
            inds[i] = self.xp.searchsorted(t[:length_i], tnew_i, side="right") - 1

            # fix end value
            inds[i][tnew_i == t[length_i - 1]] = length_i - 2
        """
        # get values outside the edges
        test_ts_end_inds = (self.xp.arange(self.num_bin_all), self.lengths - 1) 

        
        inds_bad_left = tnew < self.t_shaped[:, 0][:, None]
        inds_bad_right = tnew > self.t_shaped[test_ts_end_inds][:, None]
        if self.xp.any(inds_bad_left) or self.xp.any(inds_bad_right):
            breakpoint()
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
            raise NotImplementedError
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
        inds_orig = inds.copy()

        inds = np.tile(inds, (self.nsubs, 1)).reshape(self.nsubs, self.num_bin_all, -1).transpose(1, 0, 2).flatten()
        #flattens itself
        inds_subs = self.xp.repeat(self.xp.tile(self.xp.arange(self.nsubs), (self.num_bin_all, 1))[:, :, None], tnew.shape[1])
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
        combine_modes=True,
        bilby_start_ind=0
    ):

        # TODO: if t_obs_end = t_mrg
        self.sampling_frequency = sampling_frequency
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

        if modes is None:
            self.num_modes = len(self.amp_phase_gen.allowable_modes_hlms)
        else:
            self.num_modes = len(modes)

        """import time as tttttt
        num = 1
        st = tttttt.perf_counter()
        for _ in range(num):"""
        hlm_out = self.amp_phase_gen(
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

        self.data_length = data_length = int(Tobs * sampling_frequency)

        #self.dataTime = (
        #    self.xp.arange(data_length, dtype=self.xp.float64)
        #    * 1.0
        #    / sampling_frequency
        #)
        template_channels = self.xp.zeros((self.num_bin_all, self.data_length), dtype=self.xp.complex128)

        # adjust lengths
        length_out = hlm_out.shape[-1] if hlm_out.shape[-1] < template_channels.shape[-1] else template_channels.shape[-1]
        try:
            template_channels[:, :length_out] = hlm_out[:, :, :length_out].sum(axis=1)  # if combine_modes else hlm_out
        except ValueError:
            breakpoint()
        """et = tttttt.perf_counter()

        print("amp phase", self.num_bin_all, (et - st) /
              num, (et - st) / num / self.num_bin_all)

        st = tttttt.perf_counter()
        for _ in range(num):"""

        """et = tttttt.perf_counter()
        print("interp", self.num_bin_all, (et - st) /
              num, (et - st) / num / self.num_bin_all)
        """
        
        if return_type == "geocenter_td":
            return template_channels

        """st = tttttt.perf_counter()
        for _ in range(num):"""

        template_channels_fd_plus = self.xp.fft.rfft(template_channels.real, axis=-1)  #  * 1 / sampling_frequency
        template_channels_fd_cross = self.xp.fft.rfft(
            template_channels.imag, axis=-1)  # * 1 / sampling_frequency

        # adjust to bilby loaded interferometers
        template_channels_fd_plus[:, :bilby_start_ind] = 0.0
        template_channels_fd_cross[:, :bilby_start_ind] = 0.0

        self.templates_out = [template_channels_fd_plus, template_channels_fd_cross]

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

        if return_type != "like":
            raise ValueError("return_type must be geocenter_td, geocenter_fd, like, or detector_fd.")

    def get_ll(self, params, sampling_frequency=1024, return_complex=False, return_cupy=False, **kwargs):

        # setup kwargs specific for likelihood
        kwargs["return_type"] = "geocenter_fd"

        template_channels_fd_plus, template_channels_fd_cross = self(*params, sampling_frequency=sampling_frequency, **kwargs)

        ra, dec, psi, geocent_time = params[-4:]

        Fplus, Fcross, time_shift = self.get_detector_information(
            ra, dec, geocent_time, psi)

        template_channels_fd_plus = template_channels_fd_plus.flatten()
        template_channels_fd_cross = template_channels_fd_cross.flatten()

        num_threads_sum = 1024

        num_temps_per_bin = int(
            np.ceil((self.fd_data_length + num_threads_sum - 1) / num_threads_sum))

        temp_sums = self.xp.zeros(
            num_temps_per_bin * self.num_bin_all * self.nChannels, dtype=self.xp.complex128)
        dt = 1/self.sampling_frequency
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
        """

        if not return_complex:
            like = like.real
        
        if self.use_gpu and return_cupy is False:
            try:
                like = like.get()
            except AttributeError:
                pass

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

def CombineTPLEQMFits(eta, A1, fEQ, fTPL):
    eta2 = eta * eta
    # Impose that TPL and equal-mass limit are exactly recovered
    A0 = (
        -0.00099601593625498 * A1
        - 0.00001600025600409607 * fEQ
        + 1.000016000256004 * fTPL
    )
    A2 = -3.984063745019967 * A1 + 16.00025600409607 * fEQ - 16.0002560041612 * fTPL
    # Final formula
    return A0 + A1 * eta + A2 * eta2


def EOBGetNRSpinPeakAmplitudeV4(ell, m, m1, m2, chiS, chiA):

    eta = (m1 * m2) / ((m1 + m2) * (m1 + m2))
    dM = xp.sqrt(1.0 - 4.0 * eta)

    if m1 < m2:
        tempm1 = m1
        m1 = m2
        m2 = tempm1
        chiA = -chiA

    eta2 = eta * eta
    chi = chiS + chiA * (m1 - m2) / (m1 + m2) / (1.0 - 2.0 * eta)
    chi21 = chiS * dM / (1.0 - 1.3 * eta) + chiA
    chi33 = chiS * dM + chiA
    chi44 = chiS * (1 - 5 * eta) + chiA * dM
    chi2 = chi * chi
    chi3 = chi * chi2

    if ell == 2:
        if m == 2:
            # TPL fit
            fTPL = (
                1.4528573105413543
                + 0.16613449160880395 * chi
                + 0.027355646661735258 * chi2
                - 0.020072844926136438 * chi3
            )
            # Equal-mass fit
            fEQ = (
                1.577457498227
                - 0.0076949474494639085 * chi
                + 0.02188705616693344 * chi2
                + 0.023268366492696667 * chi3
            )
            # Global fit coefficients
            e0 = -0.03442402416125921
            e1 = -1.218066264419839
            e2 = -0.5683726304811634
            e3 = 0.4011143761465342
            A1 = e0 + e1 * chi + e2 * chi2 + e3 * chi3

            # adjusted
            
            # res = eta * CombineTPLEQMFits(eta, A1, fEQ, fTPL)

            # filled in here for vectorize capability
            eta2 = eta * eta
            # Impose that TPL and equal-mass limit are exactly recovered
            A0 = (
                -0.00099601593625498 * A1
                - 0.00001600025600409607 * fEQ
                + 1.000016000256004 * fTPL
            )
            A2 = -3.984063745019967 * A1 + 16.00025600409607 * fEQ - 16.0002560041612 * fTPL
            # Final formula
            # TODO: supposed to be eta here?
            res = eta * (A0 + A1 * eta + A2 * eta2)
        
        elif m == 1:
            res = -(
                (0.29256703361640224 - 0.19710255145276584 * eta) * eta * chi21
                + dM
                * eta
                * (
                    -0.42817941710649793
                    + 0.11378918021042442 * eta
                    - 0.7736772957051212 * eta2
                    + chi21 * chi21 * (0.047004057952214004 - eta * 0.09326128322462478)
                )
                + dM
                * eta
                * chi21
                * (-0.010195081244587765 + 0.016876911550777807 * chi21 * chi21)
            )

    elif ell == 3:
        if m == 3:
            res = (
                0.10109183988848384 * eta
                - 0.4704095462146807 * eta2
                + 1.0735457779890183 * eta2 * eta
            ) * chi33 + dM * (
                0.5636580081367962 * eta
                - 0.054609013952480856 * eta2
                + 2.3093699480319234 * eta2 * eta
                + chi33
                * chi33
                * (0.029812986680919126 * eta - 0.09688097244145283 * eta2)
            )

    elif ell == 4:
        if m == 4:
            res = (
                eta
                * (
                    0.2646580063832686
                    + 0.067584186955327 * chi44
                    + 0.02925102905737779 * chi44 * chi44
                )
                + eta2
                * (
                    -0.5658246076387973
                    - 0.8667455348964268 * chi44
                    + 0.005234192027729502 * chi44 * chi44
                )
                + eta
                * eta2
                * (
                    -2.5008294352355405
                    + 6.880772754797872 * chi44
                    - 1.0234651570264885 * chi44 * chi44
                )
                + eta2 * eta2 * (7.6974501716202735 - 16.551524307203252 * chi44)
            )

    elif ell == 5:
        if m == 5:
            res = (
                0.128621 * dM * eta
                - 0.474201 * dM * eta * eta
                + 1.0833 * dM * eta * eta * eta
                + 0.0322784 * eta * chi33
                - 0.134511 * chi33 * eta * eta
                + 0.0990202 * chi33 * eta * eta * eta
            )

    return res


EOBGetNRSpinPeakAmplitudeV4 = xp.vectorize(EOBGetNRSpinPeakAmplitudeV4)


def EOBGetNRSpinPeakADotV4(ell, m, m1, m2, chiS, chiA):

    eta = (m1 * m2) / ((m1 + m2) * (m1 + m2))
    dM = xp.sqrt(1.0 - 4.0 * eta)
    dM2 = dM * dM

    if m1 < m2:
        # RC: The fits for the HMs are done under the assumption m1>m2, so if m2>m1 we just swap the two bodies
        tempm1 = m1
        m1 = m2
        m2 = tempm1
        chiA = -chiA

    eta2 = eta * eta
    chi21 = chiS * dM / (1.0 - 2.0 * eta) + chiA
    chi33 = chiS * dM + chiA
    chi44 = chiS * (1 - 7 * eta) + chiA * dM

    if ell == 2:
        if m == 2:
            res = 0.0
        elif m == 1:
            res = (
                dM * eta * (0.007147528020812309 - eta * 0.035644027582499495)
                + dM * eta * chi21 * (-0.0087785131749995 + eta * 0.03054672006241107)
                + eta
                * 0.00801714459112299
                * xp.abs(
                    -dM
                    * (
                        0.7875612917853588
                        + eta * 1.161274164728927
                        + eta2 * 11.306060006923605
                    )
                    + chi21
                )
            )
    elif ell == 3:
        if m == 3:
            res = dM * eta * (
                -0.00309943555972098 + eta * 0.010076527264663805
            ) * chi33 * chi33 + eta * 0.0016309606446766923 * xp.sqrt(
                dM2 * (8.811660714437027 + 104.47752236009688 * eta)
                + dM * chi33 * (-5.352043503655119 + eta * 49.68621807460999)
                + chi33 * chi33
            )

    elif ell == 4:
        if m == 4:
            res = (
                eta
                * (
                    0.004347588211099233
                    - 0.0014612210699052148 * chi44
                    - 0.002428047910361957 * chi44 * chi44
                )
                + eta2
                * (
                    0.023320670701084355
                    - 0.02240684127113227 * chi44
                    + 0.011427087840231389 * chi44 * chi44
                )
                + eta * eta2 * (-0.46054477257132803 + 0.433526632115367 * chi44)
                + eta2 * eta2 * (1.2796262150829425 - 1.2400051122897835 * chi44)
            )
    elif ell == 5:
        if m == 5:
            res = eta * (
                dM * (-0.008389798844109389 + 0.04678354680410954 * eta)
                + dM * chi33 * (-0.0013605616383929452 + 0.004302712487297126 * eta)
                + dM
                * chi33
                * chi33
                * (-0.0011412109287400596 + 0.0018590391891716925 * eta)
                + 0.0002944221308683548
                * xp.abs(dM * (37.11125499129578 - 157.79906814398277 * eta) + chi33)
            )
    #else:
    #    raise NotImplementedError(f"The requested ({ell,m}) mode is not available!")
    return res


EOBGetNRSpinPeakADotV4 = xp.vectorize(EOBGetNRSpinPeakADotV4)

def EOBGetNRSpinPeakADDotV4(ell, m, m1, m2, chiS, chiA):

    eta = (m1 * m2) / ((m1 + m2) * (m1 + m2))
    dM = xp.sqrt(1.0 - 4.0 * eta)
    if m1 < m2:
        # RC: The fits for the HMs are done under the assumption m1>m2, so if m2>m1 we just swap the two bodies
        tempm1 = m1
        m1 = m2
        m2 = tempm1
        chiA = -chiA

    eta2 = eta * eta
    chi21 = chiS * dM / (1.0 - 2.0 * eta) + chiA
    chi = chiS + chiA * (m1 - m2) / (m1 + m2) / (1.0 - 2.0 * eta)
    chi33 = chiS * dM + chiA
    chiMinus1 = -1.0 + chi

    if ell == 2:
        if m == 2:
            # TPL fit
            fTPL = (
                0.002395610769995033 * chiMinus1
                - 0.00019273850675004356 * chiMinus1 * chiMinus1
                - 0.00029666193167435337 * chiMinus1 * chiMinus1 * chiMinus1
            )
            # Equal-mass fit
            fEQ = -0.004126509071377509 + 0.002223999138735809 * chi
            # Global fit coefficients
            e0 = -0.005776537350356959
            e1 = 0.001030857482885267
            A1 = e0 + e1 * chi
            # res = eta * CombineTPLEQMFits(eta, A1, fEQ, fTPL)

            # filled in here for vectorize capability
            eta2 = eta * eta
            # Impose that TPL and equal-mass limit are exactly recovered
            A0 = (
                -0.00099601593625498 * A1
                - 0.00001600025600409607 * fEQ
                + 1.000016000256004 * fTPL
            )
            A2 = -3.984063745019967 * A1 + 16.00025600409607 * fEQ - 16.0002560041612 * fTPL
            # Final formula
            res = eta * (A0 + A1 * eta + A2 * eta2)

        elif m == 1:
            res = eta * dM * 0.00037132201959950333 - xp.abs(
                dM * eta * (-0.0003650874948532221 - eta * 0.003054168419880019)
                + dM
                * eta
                * chi21
                * chi21
                * (
                    -0.0006306232037821514
                    - eta * 0.000868047918883389
                    + eta2 * 0.022306229435339213
                )
                + eta * chi21 * chi21 * chi21 * 0.0003402427901204342
                + dM * eta * chi21 * 0.00028398490492743
            )
    elif ell == 3:
        if m == 3:
            res = dM * eta * (
                0.0009605689249339088 - 0.00019080678283595965 * eta
            ) * chi33 - 0.00015623760412359145 * eta * xp.abs(
                dM
                * (
                    4.676662024170895
                    + 79.20189790272218 * eta
                    - 1097.405480250759 * eta2
                    + 6512.959044311574 * eta * eta2
                    - 13263.36920919937 * eta2 * eta2
                )
                + chi33
            )
    elif ell == 4:
        if m == 4:
            res = (
                eta * (-0.000301722928925693 + 0.0003215952388023551 * chi)
                + eta2 * (0.006283048344165004 + 0.0011598784110553046 * chi)
                + eta2 * eta * (-0.08143521096050622 - 0.013819464720298994 * chi)
                + eta2 * eta2 * (0.22684871200570564 + 0.03275749240408555 * chi)
            )
    elif ell == 5:
        if m == 5:
            res = eta * (
                dM * (0.00012727220842255978 + 0.0003211670856771251 * eta)
                + dM * chi33 * (-0.00006621677859895541 + 0.000328855327605536 * eta)
                + chi33
                * chi33
                * (-0.00005824622885648688 + 0.00013944293760663706 * eta)
            )
    #else:
    #    raise NotImplementedError(f"The requested ({ell,m}) mode is not available!")
    return res

EOBGetNRSpinPeakADDotV4 = xp.vectorize(EOBGetNRSpinPeakADDotV4)


def EOBGetNRSpinPeakOmegaV4(ell, m, eta, a):

    chi = a
    eta2 = eta * eta
    if ell == 2:
        if m == 2:
            # From TPL fit
            c0 = 0.5626787200433265
            c1 = -0.08706198756945482
            c2 = 25.81979479453255
            c3 = 25.85037751197443
            # From equal-mass fit
            d2 = 7.629921628648589
            d3 = 10.26207326082448
            # Combine TPL and equal-mass
            A4 = d2 + 4 * (d2 - c2) * (eta - 0.25)
            A3 = d3 + 4 * (d3 - c3) * (eta - 0.25)
            c4 = 0.00174345193125868
            # Final formula
            res = c0 + (c1 + c4 * chi) * xp.log(A3 - A4 * chi)
        elif m == 1:
            res = (
                0.1743194440996283
                + eta * 0.1938944514123048
                + 0.1670063050527942 * eta2
                + 0.053508705425291826 * chi
                - eta * chi * 0.18460213023455802
                + eta2 * chi * 0.2187305149636044
                + chi * chi * 0.030228846150378793
                - eta * chi * chi * 0.11222178038468673
            )
    elif ell == 3:
        if m == 3:
            res = (
                0.3973947703114506
                + 0.16419332207671075 * chi
                + 0.1635531186118689 * chi * chi
                + 0.06140164491786984 * chi * chi * chi
                + eta
                * (
                    0.6995063984915486
                    - 0.3626744855912085 * chi
                    - 0.9775469868881651 * chi * chi
                )
                + eta2
                * (
                    -0.3455328417046369
                    + 0.31952307610699876 * chi
                    + 1.9334166149686984 * chi * chi
                )
            )
    elif ell == 4:
        if m == 4:
            res = (
                0.5389359134370971
                + 0.16635177426821202 * chi
                + 0.2075386047689103 * chi * chi
                + 0.15268115749910835 * chi * chi * chi
                + eta
                * (
                    0.7617423831337586
                    + 0.009587856087825369 * chi
                    - 1.302303785053009 * chi * chi
                    - 0.5562751887042064 * chi * chi * chi
                )
                + eta2
                * (
                    0.9675153069365782
                    - 0.22059322127958586 * chi
                    + 2.678097398558074 * chi * chi
                )
                - eta2 * eta * 4.895381222514275
            )
    elif ell == 5:
        if m == 5:
            res = (
                0.6437545281817488
                + 0.22315530037902315 * chi
                + 0.2956893357624277 * chi * chi
                + 0.17327819169083758 * chi * chi * chi
                + eta
                * (
                    -0.47017798518175785
                    - 0.3929010618358481 * chi
                    - 2.2653368626130654 * chi * chi
                    - 0.5512998466154311 * chi * chi * chi
                )
                + eta2
                * (
                    2.311483807604238
                    + 0.8829339243493562 * chi
                    + 5.817595866020152 * chi * chi
                )
            )

    #else:
    #    raise NotImplementedError(f"The requested ({ell,m}) mode is not available!")
    return res

EOBGetNRSpinPeakOmegaV4 = xp.vectorize(EOBGetNRSpinPeakOmegaV4)


def EOBGetNRSpinPeakOmegaDotV4(ell, m, eta, a):

    chi = a
    eta2 = eta * eta

    if ell == 2:
        if m == 2:
            # TPL fit
            fTPL = -0.011209791668428353 + (
                0.0040867958978563915 + 0.0006333925136134493 * chi
            ) * xp.log(68.47466578100956 - 58.301487557007206 * chi)
            # Equal-mass fit
            fEQ = 0.01128156666995859 + 0.0002869276768158971 * chi
            # Global fit coefficients
            e0 = 0.01574321112717377
            e1 = 0.02244178140869133
            A1 = e0 + e1 * chi
            # res = CombineTPLEQMFits(eta, A1, fEQ, fTPL)

            # filled in here for vectorize capability
            eta2 = eta * eta
            # Impose that TPL and equal-mass limit are exactly recovered
            A0 = (
                -0.00099601593625498 * A1
                - 0.00001600025600409607 * fEQ
                + 1.000016000256004 * fTPL
            )
            A2 = -3.984063745019967 * A1 + 16.00025600409607 * fEQ - 16.0002560041612 * fTPL
            # Final formula
            res = (A0 + A1 * eta + A2 * eta2)

        elif m == 1:
            res = (
                0.0070987396362959514
                + eta * 0.024816844694685373
                - eta2 * 0.050428973182277494
                + eta * eta2 * 0.03442040062259341
                - chi * 0.0017751850002442097
                + eta * chi * 0.004244058872768811
                - eta2 * chi * 0.031996494883796855
                - chi * chi * 0.0035627260615894584
                + eta * chi * chi * 0.01471807973618255
                - chi * chi * chi * 0.0019020967877681962
            )
    elif ell == 3:
        if m == 3:
            res = (
                0.010337157192240338
                - 0.0053067782526697764 * chi * chi
                - 0.005087932726777773 * chi * chi * chi
                + eta
                * (
                    0.027735564986787684
                    + 0.018864151181629343 * chi
                    + 0.021754491131531044 * chi * chi
                    + 0.01785477515931398 * chi * chi * chi
                )
                + eta2 * (0.018084233854540898 - 0.08204268775495138 * chi)
            )
    elif ell == 4:
        if m == 4:
            res = (
                0.013997911323773867
                - 0.0051178205260273574 * chi
                - 0.0073874256262988 * chi * chi
                + eta
                * (
                    0.0528489379269367
                    + 0.01632304766334543 * chi
                    + 0.02539072293029433 * chi * chi
                )
                + eta2 * (-0.06529992724396189 + 0.05782894076431308 * chi)
            )
    elif ell == 5:
        if m == 5:
            res = (
                0.01763430670755021
                - 0.00024925743340389135 * chi
                - 0.009240404217656968 * chi * chi
                - 0.007907831334704586 * chi * chi * chi
                + eta
                * (
                    -0.1366002854361568
                    + 0.0561378177186783 * chi
                    + 0.16406275673019852 * chi * chi
                    + 0.07736232247880881 * chi * chi * chi
                )
                + eta2
                * (
                    0.9875890632901151
                    - 0.31392112794887855 * chi
                    - 0.5926145463423832 * chi * chi
                )
                - 1.6943356548192614 * eta2 * eta
            )
    #else:
    #    raise NotImplementedError(f"The requested ({ell,m}) mode is not available!")

    return res

EOBGetNRSpinPeakOmegaDotV4 = xp.vectorize(EOBGetNRSpinPeakOmegaDotV4)

def EOBCalculateNQCCoefficientsV4_freeattach(
    amplitude, phase, r, pr, omega_orb, ell, m, time_peak, time, m1, m2, chi1, chi2,nrDeltaT, num_pts_new, xp=None
):
    """
    This is just like the SEOBNRv4HM function but allows nrDeltaT to be passed in, instead of
    internally calculated.
    """
    if xp is None:
        xp = np
        use_gpu = False
    if xp == cp:
        use_gpu = True

    num_bin_all, num_modes, length = amplitude.shape 

    coeffs = {}

    eta = (m1 * m2) / ((m1 + m2) * (m1 + m2))
    chiA = (chi1 - chi2) / 2
    chiS = (chi1 + chi2) / 2

    # num bin x num modes
    nrTimePeak = time_peak[:, None] - nrDeltaT
    #print(f"nrTimePeak={nrTimePeak}")
    # Evaluate the Qs at the right time and build Q matrix (LHS)
    idx = xp.argmin(xp.abs(time[:, None]-nrTimePeak[:, :, None]), axis=-1)
    N = 7

    num_t_fit = (num_pts_new[:, None] - (idx - N)).astype(xp.int32)
    num_t_fit_max = num_t_fit.max().item()
    t_fit_inds = xp.tile((idx - N)[:, :, None], num_t_fit_max) + xp.tile(xp.arange(num_t_fit_max), (amplitude.shape[:-1] + (1,)))

    t_fit_inds[t_fit_inds > length] = -1
    t_fit = xp.take_along_axis(time[:, None, :], t_fit_inds, axis=-1)
    

    r_fit = xp.take_along_axis(r[:, None, :], t_fit_inds, axis=-1)
    omega_orb_fit = xp.take_along_axis(omega_orb[:, None, :], t_fit_inds, axis=-1)
    pr_fit = xp.take_along_axis(pr[:, None, :], t_fit_inds, axis=-1)
    amplitude_fit = xp.take_along_axis(amplitude, t_fit_inds, axis=-1)
    phase_fit = xp.take_along_axis(phase, t_fit_inds, axis=-1)

    # Eq (3) in T1100433
    rOmega = r_fit * omega_orb_fit
    q1 = pr_fit ** 2 / rOmega ** 2
    q2 = q1 / r_fit
    q3 = q2 / xp.sqrt(r_fit)

    # Eq (4) in T1100433
    p1 = pr_fit / rOmega
    p2 = p1 * pr_fit ** 2

    # Compute the time offset
    #nrDeltaT = EOBGetNRSpinPeakDeltaTv4(ell, m, m1, m2, chi1, chi2)

    # nrDeltaT defined in EOBGetNRSpinPeakDeltaT is a minus sign different from Eq. (33) of Taracchini et al.
    # Therefore, the plus sign in Eq. (21) of Taracchini et al and Eq. (18) of DCC document T1100433v2 is
    # changed to a minus sign here.

    # See below Eq(9)
    
    q1LM = q1 * amplitude_fit
    q2LM = q2 * amplitude_fit
    q3LM = q3 * amplitude_fit

    t_fit[t_fit_inds == -1] = 0.0
    q1LM[t_fit_inds == -1] = 0.0
    q2LM[t_fit_inds == -1] = 0.0
    q3LM[t_fit_inds == -1] = 0.0

    intrp_q1LM = CubicSplineInterpolantTD(
            t_fit.transpose(2, 1, 0).flatten().copy(),
            q1LM.transpose(2, 1, 0).flatten().copy(),
            num_t_fit.T.flatten(),
            1,
            num_bin_all * num_modes,
            use_gpu=use_gpu,
        )

    intrp_q2LM = CubicSplineInterpolantTD(
            t_fit.transpose(2, 1, 0).flatten().copy(),
            q2LM.transpose(2, 1, 0).flatten().copy(),
            num_t_fit.T.flatten(),
            1,
            num_bin_all * num_modes,
            use_gpu=use_gpu,
        )

    intrp_q3LM = CubicSplineInterpolantTD(
            t_fit.transpose(2, 1, 0).flatten().copy(),
            q3LM.transpose(2, 1, 0).flatten().copy(),
            num_t_fit.T.flatten(),
            1,
            num_bin_all * num_modes,
            use_gpu=use_gpu,
        )
    
    

    nrTimePeak_in = nrTimePeak.T.reshape(-1, 1)
    Q = xp.zeros((num_bin_all, num_modes, 3, 3))
    
    for i in range(3):
        Q[:, :, i, 0] = intrp_q1LM(nrTimePeak_in, deriv_order=i).reshape(num_modes, num_bin_all).T
        Q[:, :, i, 1] = intrp_q2LM(nrTimePeak_in, deriv_order=i).reshape(num_modes, num_bin_all).T
        Q[:, :, i, 2] = intrp_q3LM(nrTimePeak_in, deriv_order=i).reshape(num_modes, num_bin_all).T

    # Build the RHS

    # Compute the NR fits
    ell = xp.tile(ell, (num_bin_all, 1)).flatten()
    m = xp.tile(m, (num_bin_all, 1)).flatten()
    m1 = xp.repeat(m1, num_modes)
    m2 = xp.repeat(m2, num_modes)
    chiS = xp.repeat(chiS, num_modes)
    chiA = xp.repeat(chiA, num_modes)
    nra = xp.abs(EOBGetNRSpinPeakAmplitudeV4(ell, m, m1, m2, chiS, chiA)).reshape(num_bin_all, num_modes)
    
    nraDot = EOBGetNRSpinPeakADotV4(ell, m, m1, m2, chiS, chiA).reshape(num_bin_all, num_modes)
    """
    RC: In SEOBNRv4 nraDot is zero because the NQC are defining the peak of the 22 mode
    which by definition has a first derivative	equal to 0.
    For SEOBNRv4HM we are not computing the NQC at the peak fo the modes (see Eq.(4.3))
    of https://arxiv.org/pdf/1803.10701.pdf, so first the derivative
    is entering as a fitted coefficient.
    """
    nraDDot = EOBGetNRSpinPeakADDotV4(ell, m, m1, m2, chiS, chiA).reshape(num_bin_all, num_modes)

    # Compute amplitude and derivatives at the right time

    intrp_amp = CubicSplineInterpolantTD(
            t_fit.transpose(2, 1, 0).flatten().copy(),
            amplitude_fit.transpose(2, 1, 0).flatten().copy(),
            num_t_fit.T.flatten(),
            1,
            num_bin_all * num_modes,
            use_gpu=use_gpu,
        )

    amp = intrp_amp(nrTimePeak_in, deriv_order=0).reshape(num_modes, num_bin_all).T
    damp = intrp_amp(nrTimePeak_in, deriv_order=1).reshape(num_modes, num_bin_all).T
    ddamp = intrp_amp(nrTimePeak_in, deriv_order=2).reshape(num_modes, num_bin_all).T

    # Assemble RHS
    amps = xp.array([nra - amp, nraDot - damp, nraDDot - ddamp]).transpose((1, 2, 0))

    # Solve the equation Q*coeffs = amps
    res = xp.linalg.solve(Q, amps)

    coeffs["a1"] = res[:, :, 0].copy()
    coeffs["a2"] = res[:, :, 1].copy()
    coeffs["a3"] = res[:, :, 2].copy()

    # Now we (should) have calculated the a values. Now we can do the b values
    # Populate the P matrix in Eq. 18 of the LIGO DCC document T1100433v2
    intrp_p1 = CubicSplineInterpolantTD(
            t_fit.transpose(2, 1, 0).flatten().copy(),
            p1.transpose(2, 1, 0).flatten().copy(),
            num_t_fit.T.flatten(),
            1,
            num_bin_all * num_modes,
            use_gpu=use_gpu,
        )
    intrp_p2 = CubicSplineInterpolantTD(
            t_fit.transpose(2, 1, 0).flatten().copy(),
            p2.transpose(2, 1, 0).flatten().copy(),
            num_t_fit.T.flatten(),
            1,
            num_bin_all * num_modes,
            use_gpu=use_gpu,
        )
    P = xp.zeros((num_bin_all, num_modes, 2, 2))
    for i in range(1, 3):
        P[:, :, i - 1, 0] = -intrp_p1(nrTimePeak_in, deriv_order=i).reshape(num_modes, num_bin_all).T
        P[:, :, i - 1, 1] = -intrp_p2(nrTimePeak_in, deriv_order=i).reshape(num_modes, num_bin_all).T

    # TODO: put all of these together into one and evalute at once (in C + derivative in C?)
    intrp_phase = CubicSplineInterpolantTD(
        t_fit.transpose(2, 1, 0).flatten().copy(),
        phase_fit.transpose(2, 1, 0).flatten().copy(),
        num_t_fit.T.flatten(),
        1,
        num_bin_all * num_modes,
        use_gpu=use_gpu,
    )

    omega = intrp_phase(nrTimePeak_in, deriv_order=1).reshape(num_modes, num_bin_all).T
    omegaDot = intrp_phase(nrTimePeak_in, deriv_order=2).reshape(num_modes, num_bin_all).T

    # Since the phase can be decreasing, we need to take care not to have a -ve frequency
    omega_tmp = omega.copy()
    omegaDot_tmp = omegaDot.copy()

    omega = xp.abs(omega_tmp)
    omegaDot = (omega_tmp * omegaDot_tmp > 0.0) * xp.abs(omegaDot_tmp) + (omega_tmp * omegaDot_tmp <= 0.0) * -xp.abs(omegaDot_tmp)
    """if omega * omegaDot > 0.0:
        omega = np.abs(omega)
        omegaDot = np.abs(omegaDot)
    else:
        omega = np.abs(omega)
        omegaDot = -np.abs(omegaDot)"""

    # recalculate for the flattened case with all modes
    eta = (m1 * m2) / ((m1 + m2) * (m1 + m2))
    nromega = EOBGetNRSpinPeakOmegaV4(
        ell, m, eta, chiS + chiA * (m1 - m2) / (m1 + m2) / (1.0 - 2.0 * eta)
    ).reshape(num_bin_all, num_modes)
    nromegaDot = EOBGetNRSpinPeakOmegaDotV4(
        ell, m, eta, chiS + chiA * (m1 - m2) / (m1 + m2) / (1.0 - 2.0 * eta)
    ).reshape(num_bin_all, num_modes)

    omegas = xp.array([nromega - omega, nromegaDot - omegaDot]).transpose((1, 2, 0))
    # Solve the equation Q*coeffs = amps

    res = xp.linalg.solve(P, omegas)
    coeffs["b1"] = res[:, :, 0].copy()
    coeffs["b2"] = res[:, :, 1].copy()

    return coeffs


def EOBNonQCCorrection (r,phi,pr,pphi, omega,coeffs, xp=None):
    "Evaluate the SEOBNRv4 NQC correction, given the coefficients"

    if xp is None:
        xp = np

    sqrtR = xp.sqrt (r)
    rOmega = r * omega;
    rOmegaSq = rOmega * rOmega
    p = pr
    mag = 1. + (p * p / rOmegaSq) * (coeffs['a1']+ coeffs['a2'] / r + (coeffs['a3'] + coeffs['a3S']) / (r *sqrtR)+ coeffs['a4'] / (r * r) + coeffs['a5'] / (r * r * sqrtR))
    phase = coeffs['b1'] * p / rOmega + p * p * p / rOmega * (coeffs['b2']+coeffs['b3'] / sqrtR +coeffs['b4'] / r)

    nqc = mag * xp.cos (phase) + 1j * mag * xp.sin (phase)
    return nqc


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
    

def compute_MR_mode_free(
    t, m1, m2, chi1, chi2, attach_params, ell, m, t_match=0, phi_match=0, debug=False, xp=None
):

    if xp == None:
        xp = np
    # Step 1 - use the NR fits for amplitude and frequency at attachment time
    chiA = (chi1 - chi2) / 2
    chiS = (chi1 + chi2) / 2
    eta = m1 * m2 / (m1 + m2) ** 2
    chi = 0.5 * (chi1 + chi2) + 0.5 * (chi1 - chi2) * (m1 - m2) / (m1 + m2) / (
        1.0 - 2.0 * eta
    )
    nra = attach_params["amp"]
    nraDot = attach_params["damp"]
    nromega = attach_params["omega"]

    # Step 2 - compute the  QNMs
    final_mass = attach_params["final_mass"]
    final_spin = attach_params["final_spin"]

    omega_complex = xp.asarray([compute_QNM(int(ell_i), int(m_i), 0, final_spin.get(), final_mass.get()).conjugate() for (ell_i, m_i) in zip(ell, m)]).T
    sigmaR = -np.imag(omega_complex)
    sigmaI = -np.real(omega_complex)

    
    # Step 3 - use the fits for the free coefficients in the RD anzatse
    c1f = EOBCalculateRDAmplitudeCoefficient1(ell, m, eta, chi, xp=xp)
    c2f = EOBCalculateRDAmplitudeCoefficient2(ell, m, eta, chi, xp=xp)

    # Ensure that the ampitude of the (2,2) mode has a maximum at t_match
    #if ell == 2 and m == 2:
    #    if sigmaR > 2.0 * c1f * xp.tanh(c2f):
    #        c1f = sigmaR / (2.0 * xp.tanh(c2f))

    c1f[(ell == 2) & (m == 2) & (sigmaR > 2.0 * c1f * xp.tanh(c2f))] = sigmaR / (2.0 * xp.tanh(c2f))
    d1f = EOBCalculateRDPhaseCoefficient1(ell, m, eta, chi, xp=xp)
    d2f = EOBCalculateRDPhaseCoefficient2(ell, m, eta, chi, xp=xp)

    # Step 4 - compute the constrainted coefficients
    c1c = EOBCalculateRDAmplitudeConstraintedCoefficient1(
        c1f, c2f, sigmaR, nra, nraDot, eta[:, None], xp=xp
    )
    c2c = EOBCalculateRDAmplitudeConstraintedCoefficient2(
        c1f, c2f, sigmaR, nra, nraDot, eta[:, None], xp=xp
    )

    d1c = EOBCalculateRDPhaseConstraintedCoefficient1(d1f, d2f, sigmaI, nromega)

    """if debug:
        print(f"sigmaR:{sigmaR}, sigmaI: {sigmaI}")
        print("Free coeffs")
        print(c1f, c2f, d1f, d2f)
        print("Constrained coeffs")
        print(c1c, c2c, d1c)"""
    # Step 5 - assemble the amplitude and phase
    # First the ansatze part
    Alm = c1c[:, :, None] * xp.tanh(c1f[:, :, None] * (t[:, None, :] - t_match[:, None, None]) + c2f[:, :, None]) + c2c[:, :, None]
    philm = phi_match[:, :, None] - d1c[:, :, None] * xp.log(
        (1 + d2f[:, :, None] * xp.exp(-d1f[:, :, None] * (t[:, None, :] - t_match[:, None, None]))) / (1 + d2f[:, :, None])
    )

    test_omega = -xp.real(omega_complex) + 1j * xp.imag(omega_complex)
    # hlm = eta*Alm*xp.exp(1j*philm)*xp.exp(1j*omega_complex*(t-t_match))
    hlm = eta[:, None, None] * Alm * xp.exp(1j * philm) * xp.exp(1j * test_omega[:, :, None] * (t[:, None, :] - t_match[:, None, None]))
    return hlm, philm


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
        if self.xp.any(self.xp.isnan(r0)):
            breakpoint()
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

        # steps are axis 0
        last_ind = self.xp.where(self.xp.diff(t[1:].astype(bool).astype(int), axis=0) == -1)

        t_stop = t[last_ind][np.argsort(last_ind[-1])]
        
        # The starting point of fine integration
        t_desired = t_stop - step_back
        try:
            idx_restart = self.xp.argmin(self.xp.abs(t - t_desired[None, :]), axis=0)

        except ValueError:
            breakpoint()

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

        # add initial step
        num_steps = num_steps.astype(np.int32) + 1

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


        additionalArgsIn = self.xp.repeat(self.additionalArgs[:, :, None], traj_length, axis=-1).flatten()

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
                traj_length,
                self.xp.arange(numSys, dtype=self.xp.int32)
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
        
        """
        # mismatch with my trajectory is ~1e-5
        # mismatch with group trajectory is ~3.5e-6
        tmp  = self.xp.asarray(np.load("dynamics.npy")).copy()
        hlmtmp = np.load("hlms.npz", allow_pickle=True)
        hlmtmp = {key: self.xp.asarray(val) for key, val in hlmtmp["arr_0"][()].items()} 

        hlmtmp_sparse = np.load("hlm_sparse.npz", allow_pickle=True)
        hlmtmp_sparse = {key: self.xp.asarray(val) for key, val in hlmtmp_sparse["arr_0"][()].items()} 
        t_interp = self.xp.tile(tmp[:, 0].copy(), (self.num_bin_all, 1))

        dynamics = self.xp.tile(tmp[:, 1:].copy().T, (self.num_bin_all, 1, 1))
        tmptmp = self.xp.asarray([hlmtmp[(ell.item(), mm.item())] for ell, mm in zip(ells, mms)])
        modestmp = self.xp.tile(tmptmp, (self.num_bin_all, 1, 1))

        tmptmp_sparse = self.xp.asarray([hlmtmp_sparse[(ell.item(), mm.item())] for ell, mm in zip(ells, mms)])
        hlm_interp = self.xp.tile(tmptmp_sparse, (self.num_bin_all, 1, 1))
        self.lengths[:] = 286
        """

        dt = 1 / sampling_frequency

        delta_T = dt / (M * MTSUN_SI)
        
        t_in = (self.t / (M[:, None] * MTSUN_SI))  
        # t_in = t_interp.copy()

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
        num_pts_new = ((t_in[(self.xp.arange(t_in.shape[0]), phi_spline.lengths - 1)] - t_in[:, 0]) / delta_T).astype(int) + 1
        max_num_pts = num_pts_new.max().item()

        t_new = self.xp.tile(self.xp.arange(max_num_pts), (dynamics.shape[0], 1)) * delta_T[:, None] + t_in[:, 0][:, None]

        inds_bad = t_new > t_in.max(axis=1)[:, None]
        t_new[inds_bad] = self.xp.tile(t_in.max(axis=1)[:, None], (1, max_num_pts))[inds_bad]

        omega_orb_mine_sparse = phi_spline(t_in, deriv_order=1)
        # TODO: check this determination of the peak?
        omega_orb_mine = phi_spline(t_new, deriv_order=1)
        phi_orb_interp = phi_spline(t_new, deriv_order=0)

        idx_omega_peak = self.xp.argmax(omega_orb_mine_sparse, axis=1)
        t_omega_peak = t_in[(self.xp.arange(self.num_bin_all), idx_omega_peak)]

        tmp = hlm_interp * self.xp.exp(1j * mms[None, :, None] * phi_orb[:, None, :])

        hlms_real = self.xp.zeros((self.num_bin_all, 2 * self.num_modes, hlm_interp.shape[-1]))
        hlms_real[:, :self.num_modes, :] = tmp.real
        hlms_real[:, self.num_modes:, :] = tmp.imag

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

        #modes = modestmp.copy()
        
        if nrDeltaT is None:
            nrDeltaT = EOBGetNRSpinPeakDeltaTv4(2, 2, m_1, m_2, chi_1, chi_2)

        t_attach = t_omega_peak - nrDeltaT
        
        tmp3 = r_pr_spline(t_new)
        r_new = tmp3[:, 0]
        pr_new = tmp3[:, 1]

        amp = self.xp.abs(modes)
        phase = self.xp.unwrap(self.xp.angle(modes))

        # TODO: add fix bad modes from 
        inds_fix = ((self.xp.abs(m_1 - m_2) < 1e-4) & (self.xp.abs(chi_1) < 1e-4) & (np.abs(chi_2)< 1e-4)) | ((self.xp.abs(m_1 - m_2) < 1e-4) & (self.xp.abs(chi_1 - chi_2) < 1e-4))

        if self.xp.any(inds_fix):
            breakpoint()
            raise NotImplementedError
        
        # TODO setup inds fix
        #inds_run = ~((inds_fix[None, :] & (mms[:, None] % 2).astype(bool)).T)

        # - extra in orig script
        nrDeltaT_in = nrDeltaT[:, None] + self.xp.full((nrDeltaT.shape[0], ells.shape[0]), 10.0) * ((ells[None, :] == 5) & (mms[None, :] == 5))

        if True:  # self.xp.any(~inds_fix):
            # Compute the NQC coeffs
            t_peak = t_omega_peak
            NQC_coeffs = EOBCalculateNQCCoefficientsV4_freeattach(
                amp,
                phase,
                r_new,
                pr_new,
                omega_orb_mine,
                ells,
                mms,
                t_peak,
                t_new,
                m_1,
                m_2,
                chi_1,
                chi_2,
                nrDeltaT_in,
                num_pts_new,
                xp=self.xp
            )

            NQC_coeffs["a3S"] = np.zeros_like(NQC_coeffs["a1"])
            NQC_coeffs["a4"] = np.zeros_like(NQC_coeffs["a1"])
            NQC_coeffs["a5"] = np.zeros_like(NQC_coeffs["a1"])
            NQC_coeffs["b3"] = np.zeros_like(NQC_coeffs["a1"])
            NQC_coeffs["b4"] = np.zeros_like(NQC_coeffs["a1"])

            NQC_coeffs = {key: value[:, :, None] for key, value in NQC_coeffs.items()}
        
        # 13 ms (part below is 1ms)
        # Evaluate the correction
        NQC_correction = EOBNonQCCorrection(r_new[:, None, :], None, pr_new[:, None, :], None, omega_orb_mine[:, None, :], NQC_coeffs, xp=self.xp)
        
        # Modify the modes
        modes *= NQC_correction

        # update these
        amp = self.xp.abs(modes)
        phase = self.xp.unwrap(self.xp.angle(modes))

        # ~12 ms (this part below is 1ms)
        hIMR = {}

        amp22 = self.xp.abs(modes[:, 0])
        t_max = t_new[(self.xp.arange(self.num_bin_all), self.xp.argmax(amp22, axis=-1))]

        idx = self.xp.argmin(self.xp.abs(t_new - t_attach[:, None]), axis=-1)

        t_new = t_new - t_max[:, None]

        dt = t_new[:, 1] - t_new[:, 0]
        N = (20 / dt).astype(int) + 1
        t_match = t_new[(self.xp.arange(self.num_bin_all), idx)]
        # FIXME: the ringdown duration should be computed from QNM damping time

        final_mass = compute_final_mass_SEOBNRv4(m_1.get(), m_2.get(), chi_1.get(), chi_2.get())
        
        final_spin = bbh_final_spin_non_precessing_HBR2016(
            m_1.get(), m_2.get(), chi_1.get(), chi_2.get(), version="M3J4"
        )

        omega_complex = compute_QNM(2, 2, 0, final_spin, final_mass).conjugate()
        damping_time = 1 / np.imag(omega_complex)
        ringdown_time = (30 * damping_time).astype(int)

        t_match_all = self.xp.tile(t_match, (self.num_modes, 1)).T

        t_match_all -= 10.0 * ((ells[None, :] == 5) & (mms[None, :] == 5))

        num_t_fit = (num_pts_new - (idx - N)).astype(xp.int32)
        num_t_fit_max = num_t_fit.max().item()

        t_fit_inds = xp.tile((idx - N)[:, None], num_t_fit_max)[:, None, :] + xp.tile(xp.arange(num_t_fit_max), (amp.shape[:-1] + (1,)))
            
        t_fit_inds[t_fit_inds > amp.shape[-1]] = -1
        t_fit = xp.take_along_axis(t_new[:, None, :], t_fit_inds, axis=-1)
        amplitude_fit = xp.take_along_axis(amp, t_fit_inds, axis=-1)
        phase_fit = xp.take_along_axis(phase, t_fit_inds, axis=-1)

        num_t_fit = self.xp.tile(num_t_fit, (self.num_modes,))

        intrp_amp = CubicSplineInterpolantTD(
            t_fit.T.flatten().copy(),
            amplitude_fit.transpose(
                2, 1, 0).flatten().copy(),
            num_t_fit, # lengths
            1,
            self.num_bin_all* self.num_modes,
            use_gpu=self.use_gpu,
        )
        intrp_phase = CubicSplineInterpolantTD(
            t_fit.T.flatten().copy(),
            phase_fit.transpose(
                2, 1, 0).flatten().copy(),
            num_t_fit, # lengths
            1,
            self.num_bin_all* self.num_modes,
            use_gpu=self.use_gpu,
        )

        amp_max = intrp_amp(t_match_all.T.flatten()[:, None]).reshape(self.num_modes, self.num_bin_all).T
        damp_max = intrp_amp(t_match_all.T.flatten()[:, None], deriv_order=1).reshape(self.num_modes, self.num_bin_all).T
        phi_match = intrp_phase(t_match_all.T.flatten()[:, None]).reshape(self.num_modes, self.num_bin_all).T
        omega_max = intrp_phase(t_match_all.T.flatten()[:, None], deriv_order=1).reshape(self.num_modes, self.num_bin_all).T
        attach_params = dict(
            amp=amp_max,
            damp=damp_max,
            omega=omega_max,
            final_mass=self.xp.asarray(final_mass),
            final_spin=self.xp.asarray(final_spin),
        )

        try:
            tmp_dt = dt.get()
        except AttributeError:
            tmp_dt = dt
        num_add = int((ringdown_time / tmp_dt).max()) + 1
        t_ringdown = t_match[:, None] + self.xp.arange(num_add)[None, :] * dt[:, None]

        # TODO: change it so the number added is the actual num add for each not max
        # 13 ms (this following part is 3)
        hring, philm = compute_MR_mode_free(
                t_ringdown,
                m_1,
                m_2,
                chi_1,
                chi_2,
                attach_params,
                ells,
                mms,
                t_match=t_match,
                phi_match=phi_match,
                debug=False,
                xp=self.xp
            )
        
        # TODO: correct weird thing in modes seen earlier in code
        max_idx = idx.max().item()
        num_to_add = (hring.shape[-1] - (modes.shape[-1] - idx)).max().item()
        modes = self.xp.concatenate([modes, self.xp.zeros(modes.shape[:-1] + (num_to_add,))], axis=-1)
        
        ring_add_inds = (idx + 1)[:, None, None] + self.xp.tile(self.xp.arange(hring.shape[-1] - 1), hring.shape[:-1] + (1,))

        max_ringdown_inds = ring_add_inds.max(axis=-1)
        ring_add_inds = ring_add_inds.flatten()
        inds_bins = self.xp.tile(self.xp.arange(self.num_bin_all), (self.num_modes, hring.shape[-1] - 1, 1)).transpose(2, 0, 1).flatten()

        inds_modes = self.xp.tile(self.xp.arange(self.num_modes), (self.num_bin_all, hring.shape[-1] - 1, 1)).transpose(0, 2, 1).flatten()
        
        modes[(inds_bins, inds_modes, ring_add_inds)] = hring[:, :, 1:].flatten()
    
        fix = self.xp.tile(self.xp.arange(modes.shape[-1]), (self.num_bin_all, self.num_modes, 1)) > max_ringdown_inds[:, :, None]

        modes[fix] = 0.0
        # fix anything above maximum ringdown piece to zero


        # 16 ms
        #(2, 3), 'constant', constant_values=(4, 6)

        """
        # adjust this to cut off ends of shorter signals
        
        for ell_m, mode in hlms.items():
p
            #if ell_m == (5, 5):
            
            
                # print(t_match)
            ell, m = ell_m
            
            amp_fit = self.xp.take_along_axis(self.xp.abs(mode[idx - N :])
            phase = self.xp.unwrap(self.xp.angle(mode))
            intrp_amp = CubicSpline(t[idx - N :], amp)
            intrp_phase = CubicSpline(t[idx - N :], phase[idx - N :])
            amp_max = intrp_amp(t_match)
            damp_max = intrp_amp.derivative()(t_match)
            phi_match = intrp_phase(t_match)
            omega_max = intrp_phase.derivative()(t_match)
            attach_params = dict(
                amp=amp_max,
                damp=damp_max,
                omega=omega_max,
                final_mass=final_mass,
                final_spin=final_spin,
            )
            hring, philm = compute_MR_mode_free(
                t_ringdown,
                m1,
                m2,
                chi1,
                chi2,
                attach_params,
                ell,
                m,
                t_match=t_match,
                phi_match=phi_match,
                debug=False,
            )
            # Construct the full IMR waveform
            hIMR[(ell, m)] = self.xp.concatenate((hlms[(ell, m)][: idx + 1], hring[1:]))
            t_match = t[idx]
            t_IMR = self.xp.concatenate((t[: idx + 1], t_ringdown[1:]))"""

        """et = tttttt.perf_counter()

        print("splines", self.num_bin_all, (et - st) /
              num, (et - st) / num / self.num_bin_all)

        st = tttttt.perf_counter()
        for _ in range(num):
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
        """
        return modes

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
                            num_steps, ells, mms)

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
    
        # 4 ms
        modes = self.compute_full_waveform(
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

        modes /= distance[:, None, None]

        self.eob_c_class.deallocate_information()
        # 30 ms
        return modes
        


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
