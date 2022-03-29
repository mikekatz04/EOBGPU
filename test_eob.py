from gwpy.timeseries import TimeSeries
import bilby
import time
from eob.seobnrv4phm import BBHWaveformTD, SEOBNRv4PHM
from bbhx.utils.constants import PC_SI
import numpy as np
from cupy.cuda.runtime import setDevice
setDevice(1)


mt = 40.0  # Total mass in solar masses
q = 2.5
num = int(5e3)
# was 16.82307190336287 0.001682307190336287
m1 = np.full(num, mt * q / (1.+q))
m2 = np.full(num, mt / (1.+q))
# chi1x,
# chi1y,
chi1z = np.full(num, 0.0001)
# chi2x,
# chi2y,
chi2z = np.full(num, 0.0001)
distance = np.full(num, 100.0) * 1e6 * PC_SI  # Mpc -> m
phiRef = np.full(num, 0.0)
inc = np.full(num, np.pi / 3.0)
ra = np.full(num, np.pi / 4.0)
dec = np.full(num, np.pi / 5.0)
psi = np.full(num, np.pi / 6.0)
geocent_time = np.full(num, 10.0)

eob = SEOBNRv4PHM(use_gpu=True)  # gpu_available)
# eob(m1, m2, chi1z, chi2z, distance, phiRef)

bilby_interferometers = [bilby.gw.detector.get_empty_interferometer(
    interf) for interf in ['L1', "H1"]]

Tobs = 3.
start = 1126259446
for inter in bilby_interferometers:
    time_series = TimeSeries.fetch_open_data(
        inter.name, start, start + Tobs)
    inter.set_strain_data_from_gwpy_timeseries(time_series=time_series)


sampling_frequency = inter.sampling_frequency
data_length = int(Tobs * sampling_frequency)

bbh = BBHWaveformTD(bilby_interferometers, use_gpu=True)
n = 20
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
    modes=None,
    bufferSize=None,
    fill=False,
)
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
        modes=None,
        bufferSize=None,
        fill=False,
    )

    print(jj)
et = time.perf_counter()
print((et - st)/n/num, "done")


breakpoint()
