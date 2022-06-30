import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import bilby
import time
from eob.seobnrv4phm import BBHWaveformTD, SEOBNRv4PHM
from bbhx.utils.constants import PC_SI
import numpy as np
from cupy.cuda.runtime import setDevice
setDevice(2)


mt = 60.0  # Total mass in solar masses
q = 2.5
num = int(4)
# was 16.82307190336287 0.001682307190336287
m1 = np.full(num, 0.55 * mt)  # mt * q / (1.+q))
m2 = np.full(num, 0.45 * mt)  # mt / (1.+q))
# chi1x,
# chi1y,
chi1z = np.full(num, 0.2)
# chi2x,
# chi2y,
chi2z = np.full(num, 0.3)
distance = np.full(num, 100.0) * 1e6 * PC_SI  # Mpc -> m
phiRef = np.full(num, 0.0)
inc = np.full(num, np.pi / 3.0)
ra = np.full(num, np.pi / 4.0)
dec = np.full(num, np.pi / 5.0)
psi = np.full(num, np.pi / 6.0)
geocent_time = np.full(num, 10.0)

chi1z[0::2] *= 0.98
chi2z[0::2] *= 1.01



eob = SEOBNRv4PHM(use_gpu=True)  # gpu_available)
# eob(m1, m2, chi1z, chi2z, distance, phiRef)

bilby_interferometers = [bilby.gw.detector.get_empty_interferometer(
    interf) for interf in ['L1', "H1"]]

Tobs = 3.

sampling_frequency = 4096.0
data_length = int(Tobs * sampling_frequency)
for inter in bilby_interferometers:
    # time_series = TimeSeries.fetch_open_data(
    #    inter.name, start, start + Tobs)
    # inter.set_strain_data_from_gwpy_timeseries(time_series=time_series)
    # inter.set_strain_data_from_frequency_domain_strain(
    #    frequency_domain_strain, sampling_frequency=None, duration=None, start_time=0, frequency_array=None)
    inter.set_strain_data_from_zero_noise(
        sampling_frequency, Tobs, start_time=0)
    # inter.set_strain_data_from_power_spectral_density(
    #    sampling_frequency, Tobs, start_time=0)

bbh = BBHWaveformTD(bilby_interferometers, use_gpu=True)
n = 2
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
    sampling_frequency=sampling_frequency,
    Tobs=Tobs,
    #modes=[(2, 2)],
    bufferSize=None,
    fill=False,
    return_type="geocenter_td"  # "detector_fd"
)

#for i, inter in enumerate(bilby_interferometers):
#    inter.set_strain_data_from_frequency_domain_strain(
#        out[0, i].get(), sampling_frequency=sampling_frequency, duration=Tobs)

del bbh
bbh = BBHWaveformTD(bilby_interferometers, use_gpu=True)

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
            # modes=modes,
            # fs=fs,
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
        sampling_frequency=sampling_frequency,
        Tobs=Tobs,
        #modes=[(2, 2)],
        bufferSize=None,
        fill=False,
        return_type="geocenter_td"  # "detector_fd"
    )

    print(jj)
et = time.perf_counter()
print((et - st)/n/num, "done")


breakpoint()
