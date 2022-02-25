import numpy as np
from cupy.cuda.runtime import setDevice
setDevice(2)

mt = 40.0 # Total mass in solar masses
q = 2.5
num = int(3e4)
# was 16.82307190336287 0.001682307190336287
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
lam = np.full(num, np.pi / 4.0)
beta = np.full(num, np.pi / 5.0)
psi = np.full(num, np.pi / 6.0)
t_ref = np.full(num, np.pi / 7.0)

from eob.seobnrv4phm import BBHWaveformTD, SEOBNRv4PHM
eob = SEOBNRv4PHM(use_gpu=True)  # gpu_available)
# eob(m1, m2, chi1z, chi2z, distance, phiRef)
bbh = BBHWaveformTD(use_gpu=True)
import time
n = 10
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
    sampling_frequency=1024.,
    Tobs=3.0,
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
    
    print(jj)
et = time.perf_counter()
print((et - st)/n/num, "done")



breakpoint()
