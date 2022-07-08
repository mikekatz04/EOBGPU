from eob.seobnrv4phm import BBHWaveformTD
import numpy as np
import cupy as xp
import bilby
import matplotlib.pyplot as plt
xp.cuda.runtime.setDevice(3)
np.random.seed(10001)

mt = 60.0 # Total mass in solar masses
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
phiRef = 0.8
inc = np.pi / 3.0
lam = np.pi / 4.0
beta = np.pi / 5.0
psi = np.pi / 6.0
geocent_time = 2.0

injection_params = np.array([mt, q, chi1z, chi2z, distance, phiRef, np.cos(inc), lam, np.sin(beta), psi, geocent_time])


bilby_interferometers = [bilby.gw.detector.get_empty_interferometer(
    interf) for interf in ['L1', "H1", "V1"]]

Tobs = 6.

sampling_frequency = 4096.0 # Hz
data_length = int(Tobs * sampling_frequency)
for inter in bilby_interferometers:
    inter.set_strain_data_from_zero_noise(
        sampling_frequency, Tobs, start_time=0)

bbh = BBHWaveformTD(bilby_interferometers, use_gpu=True)

from lisatools.sampling.likelihood import Likelihood
 
dt = 1./sampling_frequency
num_channels = len(bilby_interferometers)

waveform_kwargs = dict(
    sampling_frequency=sampling_frequency,
    Tobs=Tobs,
    #modes=[(2, 2)],
    bufferSize=None,
    fill=False,
    bilby_start_ind=0,
    return_type="detector_fd"
)
# return_type="like",

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

from eryn.prior import uniform_dist, PriorContainer

del bilby_interferometers
bilby_interferometers = [bilby.gw.detector.get_empty_interferometer(
    interf) for interf in ['L1', "H1", "V1"]]

for i, inter in enumerate(bilby_interferometers):
    inter.set_strain_data_from_frequency_domain_strain(
        data[0, i].get(), sampling_frequency=sampling_frequency, duration=Tobs)
        
del bbh
bbh = BBHWaveformTD(bilby_interferometers, use_gpu=True)

priors = {"bbh": PriorContainer({
    0: uniform_dist(58., 62.),
    1: uniform_dist(1., 5.),
    2: uniform_dist(-0.1, 0.1),
    3: uniform_dist(-0.1, 0.1),
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

nwalkers = 200
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

max_iter = 10
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

    lprior_temp = sampler.compute_log_prior(start_state.branches_coords)[0]
    lp_temp = sampler.compute_log_prob(start_state.branches_coords)[0]

    ii += 1
    print("CHECK", np.std(lp_temp))
    if np.std(lp_temp) > 5.0 or ii >= max_iter:
        run_flag = False

start_state = State({"bbh": start_points})

nsteps = 100

sampler.run_mcmc(start_state, nsteps, progress=True)

breakpoint()
