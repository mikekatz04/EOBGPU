import numpy as np
from bbhx.utils.constants import *

import sys

#from odex.pydopr853 import DOPR853
from new_Dormand_horiz import DOPR853
from ode_for_test import ODE_1

try:
    import cupy as xp
    from pyEOB import compute_hlms as compute_hlms_gpu
    from pyEOB import root_find_all as root_find_all_gpu
    from pyEOB import root_find_scalar_all as root_find_scalar_all_gpu
    from pyEOB import ODE as ODE_gpu
    gpu_available = True
    from cupy.cuda.runtime import setDevice
    setDevice(2)

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")
    import numpy as xp
    gpu_available = False

sys.path.append(
    "/Users/michaelkatz/Research/EOBGPU/eob_development/toys/hamiltonian_prototype/"
)

sys.path.append("/home/mlk667/EOBGPU/eob_development/toys/hamiltonian_prototype/")

from pyEOB_cpu import compute_hlms as compute_hlms_cpu
from pyEOB_cpu import root_find_all as root_find_all_cpu
from pyEOB_cpu import root_find_scalar_all as root_find_scalar_all_cpu
from pyEOB_cpu import ODE as ODE_cpu

#from HTMalign_AC import HTMalign_AC
#from RR_force_aligned import RR_force_2PN
#from initial_conditions_aligned import computeIC

def stopping_criterion(step_num, denseOutput):
    stop = denseOutput[(step_num, np.zeros_like(step_num), np.arange(len(step_num)))] < 6.0
    return stop

class ODEWrapper:
    def __init__(self, use_gpu=False):
        if use_gpu:
            self.xp = xp
            self.ode = ODE_gpu
        else:
            self.xp = np
            self.ode = ODE_cpu

    def __call__(self, x, args, k, additionalArgs):
        reshape = k.shape
        numSys = reshape[-1]
        x_in = x.flatten()
        args_in = args.flatten()
        k_in = k.flatten()
        additionalArgs_in = additionalArgs.flatten()

        self.ode(x_in, args_in, k_in, additionalArgs_in, numSys)

        k[:] = k_in.reshape(reshape)


class SEOBNRv4PHM:
    def __init__(self, max_init_len=-1, use_gpu=False, **kwargs):

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
        self.integrator = DOPR853(ODEWrapper(use_gpu=use_gpu), stopping_criterion=stopping_criterion, tmax=1e7, max_step=500, use_gpu=self.use_gpu)  # use_gpu=use_gpu)

    def _sanity_check_modes(self, ells, mms):
        for (ell, mm) in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes
                    )
                )

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

        r_guess = omega0 ** (-2.0 / 3)
        # z = [r_guess, np.sqrt(r_guess)]
        n = 2
        # print(f"Initial guess is {z}")
        # The conservative bit: solve for r and pphi
        num_args = 4
        num_add_args = 5
        argsIn = self.xp.zeros((num_args, self.num_bin_all)).flatten()
        additionalArgsIn = self.xp.zeros((num_add_args, self.num_bin_all))
        additionalArgsIn[0] = m_1_scaled
        additionalArgsIn[1] = m_2_scaled
        additionalArgsIn[2] = chi_1
        additionalArgsIn[3] = chi_2
        additionalArgsIn[4] = omega0
        additionalArgsIn = additionalArgsIn.flatten()

        x0In = self.xp.zeros((n, self.num_bin_all))
        x0In[0] = r_guess
        x0In[1] = self.xp.sqrt(r_guess)

        xOut = self.xp.zeros_like(x0In).flatten()
        x0In = x0In.flatten()

        self.root_find(
            xOut,
            x0In,
            argsIn,
            additionalArgsIn,
            max_iter,
            err,
            self.num_bin_all,
            n,
            num_args,
            num_add_args,
        )

        r0, pphi0 = xOut.reshape(n, self.num_bin_all)

        pr0 = self.xp.zeros(self.num_bin_all)
        start_bounds = self.xp.tile(
            self.xp.array([-1e-2, 0.0]), (self.num_bin_all, 1)
        ).T.flatten()

        argsIn = self.xp.zeros((num_args, self.num_bin_all))
        argsIn[0] = r0
        argsIn[3] = pphi0

        argsIn = argsIn.flatten()

        self.root_find_scalar(
            pr0,
            start_bounds,
            argsIn,
            additionalArgsIn,
            max_iter,
            err,
            self.num_bin_all,
            num_args,
            num_add_args,
        )

        return r0, pphi0, pr0

    def run_trajectory(self, r0, pphi0, pr0, m_1, m_2, chi_1, chi_2, fs=10.0, **kwargs):

        # TODO: constants from LAL
        mt = m_1 + m_2  # Total mass in solar masses
        omega0 = fs * (mt * MTSUN_SI * np.pi)
        m_1_scaled = m_1 / mt
        m_2_scaled = m_2 / mt
        dt = 1.0 / 16384 / (mt * MTSUN_SI)

        condBound = self.xp.array([r0, np.full_like(r0, 0.0), pr0, pphi0])
        argsData = self.xp.array([m_1_scaled, m_2_scaled, chi_1, chi_2])

        # TODO: make adjustable
        # TODO: debug dopr?
        t, traj, deriv, num_steps = self.integrator.integrate(
                condBound.copy(), argsData.copy()
            )

        num_steps_max = num_steps.max().item()
        
        return (
            (t[:num_steps_max, :] * self.xp.asarray(mt[self.xp.newaxis, :]) * MTSUN_SI).T,
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
        return NotImplementedError

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
        fs=10.0,  # Hz
    ):
        if modes is not None:
            ells = self.xp.asarray([ell for ell, mm in modes], dtype=self.xp.int32)
            mms = self.xp.asarray([mm for ell, mm in modes], dtype=self.xp.int32)

            self._sanity_check_modes(ells, mms)

        else:
            ells = self.ells_default
            mms = self.mms_default

        self.num_modes = len(ells)

        self.num_bin_all = len(m1)

        r0, pphi0, pr0 = self.get_initial_conditions(m1, m2, chi1z, chi2z, fs=fs)

        t, traj, num_steps = self.run_trajectory(
            r0, pphi0, pr0, m1, m2, chi1z, chi2z, fs=fs
        )

        hlms = self.get_hlms(traj, m1, m2, chi1z, chi2z, num_steps, ells, mms)

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
    eob = SEOBNRv4PHM(use_gpu=gpu_available)

    num = 100000
    m1 = np.full(num, 35.0)
    m2 = np.full(num, 30.0)
    # chi1x,
    # chi1y,
    chi1z = np.full(num, 0.6)
    # chi2x,
    # chi2y,
    chi2z = np.full(num, 0.05)
    distance = np.full(num, 100.0)  # Mpc
    phiRef = np.full(num, 0.0)
    inc = np.full(num, np.pi / 3.0)
    lam = np.full(num, np.pi / 4.0)
    beta = np.full(num, np.pi / 5.0)
    psi = np.full(num, np.pi / 6.0)
    t_ref = np.full(num, np.pi / 7.0)

    import time
    n = 5
    st = time.perf_counter()
    for _ in range(n):
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
    et = time.perf_counter()
    print((et - st)/n, "done")
    # eob(m1, m2, chi1z, chi2z, distance, phiRef)
    breakpoint()
    bbh = BBHWaveformTD(lisa=False, use_gpu=False)

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
        lam,
        beta,
        psi,
        t_ref,
        sampling_frequency=1024,
        Tobs=60.0,
        modes=None,
        bufferSize=None,
        fill=False,
    )
    breakpoint()
