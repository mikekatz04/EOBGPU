from eob.seobnrv4phm import BBHWaveformTD
import numpy as np
import cupy as xp
xp.cuda.runtime.setDevice(2)


mt = 40.0 # Total mass in solar masses
q = 2.5
num = int(1e4)
m1 = mt * q / (1.+q)
m2 = mt / (1.+q)
# chi1x,
# chi1y,
chi1z = 0.0001
# chi2x,
# chi2y,
chi2z = 0.0001
from bbhx.utils.constants import PC_SI
distance = 100.0  # Mpc -> m
phiRef = 0.0
inc = np.pi / 3.0
lam = np.pi / 4.0
beta = np.pi / 5.0
psi = np.pi / 6.0
t_ref = np.pi / 7.0

injection_params = np.array([mt, q, chi1z, chi2z, distance, phiRef, np.cos(inc), lam, np.sin(beta), psi, t_ref])

bbh = BBHWaveformTD(use_gpu=True)

from lisatools.sampling.likelihood import Likelihood

sampling_frequency = 1024.  # Hz
dt = 1./sampling_frequency
num_channels = 3
Tobs = 5.0  # seconds

waveform_kwargs = dict(
    sampling_frequency=sampling_frequency,
    Tobs=Tobs,
    modes=None,
    bufferSize=None,
    fill=False,
)

transforms = {
    4: (lambda x: x * 1e6 * PC_SI),
    6: np.arccos,
    8: np.arcsin,
    (0, 1): (lambda mt, q: (mt * q / (1 + q), mt / (1 + q)))
}

from eryn.utils import TransformContainer

transform_fn = {"bbh": TransformContainer(parameter_transforms=transforms, fill_dict=None)}

injection_params_in = transform_fn["bbh"].both_transforms(injection_params)
params_temp_in = np.repeat(injection_params_in[:, None], 2, axis=-1).copy()
data = bbh(*params_temp_in, **waveform_kwargs)

data_in = list(data[0].copy())

from eryn.prior import uniform_dist, PriorContainer

priors = {"bbh": PriorContainer({
    0: uniform_dist(30., 50.),
    1: uniform_dist(1., 5.),
    2: uniform_dist(-0.99, 0.99),
    3: uniform_dist(-0.99, 0.99),
    4: uniform_dist(10., 1000),
    5: uniform_dist(0., 2. * np.pi),
    6: uniform_dist(-1., 1.),
    7: uniform_dist(0., 2. * np.pi),
    8: uniform_dist(-1., 1.),
    9: uniform_dist(0., np.pi),
    10: uniform_dist(0., Tobs),
})}

periodic = {
    "bbh": {
        5: 2* np.pi,
        7: 2 * np.pi,
        9: np.pi
    }
}


like = Likelihood(
    bbh,
    num_channels,
    dt=dt,
    parameter_transforms=transform_fn,
    use_gpu=True,
    vectorized=True,
    separate_d_h=False,
    return_cupy=False,
    fill_data_noise=False,
    transpose_params=True,
    subset=None,
)

from lisatools.sensitivity import get_sensitivity

like.inject_signal(
    data_stream=data_in,
    noise_fn=get_sensitivity,
    noise_kwargs={"sens_fn": "lisasens"}
)
nwalkers = 30
ntemps = 10

from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.state import State
backend = HDFBackend("test_eob_sampling.h5")
sampler = EnsembleSampler(
    nwalkers,
    [11],  # assumes ndim_max
    like,
    priors,
    tempering_kwargs={"ntemps": ntemps},
    nbranches=1,
    kwargs=waveform_kwargs,
    backend=backend,
    vectorize=True,
    plot_iterations=-1,
    periodic=periodic,  # TODO: add periodic to proposals
    branch_names=["bbh"],
)


factor = 1e-2
cov = np.ones(11) * 1e-3
mean_vals = injection_params

max_iter = 2
ii = 0 
run_flag = True
while run_flag: 
    factor *=2
    start_points = np.tile(
        mean_vals,
        (ntemps, nwalkers, 1, 1),
    ) * (1 + factor * cov * np.random.randn(ntemps, nwalkers, 1, cov.shape[0]))

    start_points[:, :, :, 5] %= (2 * np.pi)
    start_points[:, :, :, 7] %= (2 * np.pi)
    start_points[:, :, :, 9] %= (np.pi)

    start_points[:, :, :, 6] = np.clip(start_points[:, :, :, 6], -0.99, 0.99)
    start_points[:, :, :, 8] = np.clip(start_points[:, :, :, 8], -0.99, 0.99)

    start_state = State({"bbh": start_points})
    lp_temp = sampler.compute_log_prob(start_state.branches_coords)[0]

    ii += 1
    print("CHECK", np.std(lp_temp))
    if np.std(lp_temp) > 5.0 or ii >= max_iter:
        run_flag = False

start_state = State({"bbh": start_points})

nsteps = 100
sampler.run_mcmc(start_state, nsteps, progress=True)



breakpoint()

bbh(
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
        sampling_frequency=1024.,
        Tobs=5.0,
        modes=None,
        bufferSize=None,
        fill=False,
    )
breakpoint()