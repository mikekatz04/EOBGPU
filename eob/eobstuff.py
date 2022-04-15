import numpy as np

import lalsimulation as lalsim
import lal

def compute_final_mass_SEOBNRv4(mass1,mass2,a1,a2):
    q = mass1 / mass2
    eta = mass1*mass2/(mass1+mass2)**2
    InvQ = 1./q
    OnePlusInvQ = 1. + InvQ
    InvQ2 = InvQ * InvQ

    atl = (a1 + a2 * InvQ2) / (OnePlusInvQ * OnePlusInvQ)
    tmpVar = (a1 + a2 * InvQ2) / (1. + InvQ2)
    rISCO = lalsim.SimRadiusKerrISCO( atl )
    eISCO = lalsim.SimEnergyKerrISCO( rISCO )
    finalMass = 1. - ((1. - eISCO) * eta + 16. * eta * eta * \
                       (0.00258 - 0.0773 / (1. / ((1. + InvQ2) / (OnePlusInvQ * OnePlusInvQ)) * atl - 1.6939) - 0.25 * (1. - eISCO)))
    return finalMass

compute_final_mass_SEOBNRv4 = np.vectorize(compute_final_mass_SEOBNRv4)

from pesummary.gw.conversions.nrutils import (
    bbh_final_mass_non_precessing_UIB2016,
    bbh_final_spin_non_precessing_HBR2016
)

bbh_final_mass_non_precessing_UIB2016 = np.vectorize(bbh_final_mass_non_precessing_UIB2016)
bbh_final_spin_non_precessing_HBR2016 = np.vectorize(bbh_final_spin_non_precessing_HBR2016)


def compute_QNM(ell,m,n,af,Mf):
    qnm_temp = lal.CreateCOMPLEX16Vector(1)
    lalsim.SimIMREOBGenerateQNMFreqV2fromFinal(qnm_temp, Mf,af, ell, m, 1)
    omega = (qnm_temp.data*lal.MTSUN_SI).conjugate()
    #omega2 = compute_QNM2(ell, m, n, af, Mf)
    #print(omega,omega2)
    return omega

compute_QNM = np.vectorize(compute_QNM, excluded=[0, 1, 2])



def EOBCalculateRDAmplitudeCoefficient1( ell, m,  eta,  chi, xp=None):

    if xp is None:
        xp = np
    A1coeff00={}
    A1coeff01={}
    A1coeff02={}
    A1coeff10={}
    A1coeff11={}
    A1coeff20={}
    A1coeff21={}
    A1coeff31={}
    modes = [(2,2),(2,1),(3,3),(4,4),(5,5)]
    for mode in modes:
        A1coeff00[mode] = 0.0
        A1coeff01[mode] = 0.0
        A1coeff02[mode] = 0.0
        A1coeff10[mode] = 0.0
        A1coeff11[mode] = 0.0
        A1coeff20[mode] = 0.0
        A1coeff21[mode] = 0.0
        A1coeff31[mode] = 0.0
    """
    /* Fit coefficients that enter amplitude of the RD signal for SEOBNRv4 and SEOBNRv4HM*/
    /* Eq. (54) of https://dcc.ligo.org/T1600383 for SEOBNRv4 and 22 mode SEOBNRv4HM*/
    /* Eqs. (C1, C3, C5, C7) of https://arxiv.org/pdf/1803.10701.pdf for SEOBNRv4HM*/
    """
    A1coeff00[(2,2)] = 0.0830664
    A1coeff01[(2,2)] = -0.0196758
    A1coeff02[(2,2)] = -0.0136459
    A1coeff10[(2,2)] = 0.0612892
    A1coeff11[(2,2)] = 0.00146142
    A1coeff20[(2,2)] = -0.0893454

    A1coeff00[(2,1)] = 0.07780330893915006
    A1coeff01[(2,1)] = -0.05070638166864379
    A1coeff10[(2,1)] = 0.24091006920322164
    A1coeff11[(2,1)] = 0.38582622780596576
    A1coeff20[(2,1)] = -0.7456327888190485
    A1coeff21[(2,1)] = -0.9695534075470388

    A1coeff00[(3,3)] = 0.07638733045623343
    A1coeff01[(3,3)] = -0.030993441267953236
    A1coeff10[(3,3)] = 0.2543447497371546
    A1coeff11[(3,3)] = 0.2516879591102584
    A1coeff20[(3,3)] = -1.0892686061231245
    A1coeff21[(3,3)] = -0.7980907313033606

    A1coeff00[(4,4)] = -0.06392710223439678
    A1coeff01[(4,4)] = -0.03646167590514318
    A1coeff10[(4,4)] = 0.345195277237925
    A1coeff11[(4,4)] = 1.2777441574118558
    A1coeff20[(4,4)] = -1.764352185878576
    A1coeff21[(4,4)] = -14.825262897834696
    A1coeff31[(4,4)] = 40.67135475479875

    A1coeff00[(5,5)] = -0.06704614393611373
    A1coeff01[(5,5)] = 0.021905949257025503
    A1coeff10[(5,5)] = -0.24754936787743445
    A1coeff11[(5,5)] = -0.0943771022698497
    A1coeff20[(5,5)] = 0.7588042862093705
    A1coeff21[(5,5)] = 0.4357768883690394

    out = xp.asarray([A1coeff00[(ell_i.item(),m_i.item())] + A1coeff01[(ell_i.item(),m_i.item())] * chi + A1coeff02[(ell_i.item(),m_i.item())] * chi * chi + A1coeff10[(ell_i.item(),m_i.item())] * eta +A1coeff11[(ell_i.item(),m_i.item())] * eta * chi + A1coeff20[(ell_i.item(),m_i.item())] * eta * eta + A1coeff21[(ell_i.item(),m_i.item())]*eta*eta*chi + A1coeff31[(ell_i.item(),m_i.item())]*eta*eta*eta*chi for (ell_i, m_i) in zip(ell, m)]).T
    return out


def EOBCalculateRDAmplitudeCoefficient2(ell,  m,  eta,  chi, xp=None):

    if xp is None:
        xp = np

    A2coeff00={}
    A2coeff01={}
    A2coeff10={}
    A2coeff11={}
    A2coeff20={}
    A2coeff21={}
    A2coeff31={}
    modes = [(2,2),(2,1),(3,3),(4,4),(5,5)]
    for mode in modes:
        A2coeff00[mode] = 0.0
        A2coeff01[mode] = 0.0
        A2coeff10[mode] = 0.0
        A2coeff11[mode] = 0.0
        A2coeff20[mode] = 0.0
        A2coeff21[mode] = 0.0
        A2coeff31[mode] = 0.0

    """
    /* Fit coefficients that enter amplitude of the RD signal of SEOBNRv4 and SEOBNRv4HM*/
    /* Eq. (55) of https://dcc.ligo.org/T1600383 for SEOBNRv4 and 22 mode SEOBNRv4HM*/
    /* Eqs. (C1, C3, C5, C7) of https://arxiv.org/pdf/1803.10701.pdf for SEOBNRv4HM*/
    """
    A2coeff00[(2,2)] = -0.623953
    A2coeff01[(2,2)] = -0.371365
    A2coeff10[(2,2)] = 1.39777
    A2coeff11[(2,2)] = 2.40203
    A2coeff20[(2,2)] = -1.82173
    A2coeff21[(2,2)] = -5.25339

    A2coeff00[(2,1)] = -1.2451852641667298
    A2coeff01[(2,1)] = -1.195786238319961
    A2coeff10[(2,1)] = 6.134202504312409
    A2coeff11[(2,1)] = 15.66696619631313
    A2coeff20[(2,1)] = -14.67251127485556
    A2coeff21[(2,1)] = -44.41982201432511

    A2coeff00[(3,3)] = -0.8325292359346013
    A2coeff01[(3,3)] = -0.598880303198448
    A2coeff10[(3,3)] = 2.767989795032018
    A2coeff11[(3,3)] = 5.904371617277156
    A2coeff20[(3,3)] = -7.028151926115957
    A2coeff21[(3,3)] = -18.232606706124482

    A2coeff00[(4,4)] = 0.7813275473485185
    A2coeff01[(4,4)] = 0.8094706044462984
    A2coeff10[(4,4)] = -5.18689829943586
    A2coeff11[(4,4)] = -5.38343327318501
    A2coeff20[(4,4)] = 14.026415859369477
    A2coeff21[(4,4)] = 0.1051625997942299
    A2coeff31[(4,4)] = 46.978434956814006

    A2coeff00[(5,5)] = 1.6763424265367357
    A2coeff01[(5,5)] = 0.4925695499534606
    A2coeff10[(5,5)] = -5.604559311983177
    A2coeff11[(5,5)] = -6.209095657439377
    A2coeff20[(5,5)] = 16.751319143123386
    A2coeff21[(5,5)] = 16.778452555342554

    out = xp.asarray([A2coeff00[(ell_i.item(),m_i.item())] + A2coeff01[(ell_i.item(),m_i.item())] * chi + A2coeff10[(ell_i.item(),m_i.item())] * eta +A2coeff11[(ell_i.item(),m_i.item())] * eta * chi + A2coeff20[(ell_i.item(),m_i.item())] * eta * eta + A2coeff21[(ell_i.item(),m_i.item())] * eta * eta * chi + A2coeff31[(ell_i.item(),m_i.item())]*eta*eta*eta*chi for (ell_i, m_i) in zip(ell, m)]).T

    return out




def EOBCalculateRDPhaseCoefficient1(ell, m,  eta,  chi, xp=None):

    if xp is None:
        xp = np

    P1coeff00={}
    P1coeff01={}
    P1coeff02={}
    P1coeff10={}
    P1coeff11={}
    P1coeff20={}
    P1coeff21={}
    P1coeff31={}
    modes = [(2,2),(2,1),(3,3),(4,4),(5,5)]
    for mode in modes:
        P1coeff00[mode] = 0.0
        P1coeff01[mode] = 0.0
        P1coeff02[mode] = 0.0
        P1coeff10[mode] = 0.0
        P1coeff11[mode] = 0.0
        P1coeff20[mode] = 0.0
        P1coeff21[mode] = 0.0
        P1coeff31[mode] = 0.0
    """
    /* Fit coefficients that enter the phase of the RD signal of SEOBNRv4 and SEOBNRv4HM*/
    /* Eq. (62) of https://dcc.ligo.org/T1600383 for SEOBNRv4 and 22 mode SEOBNRv4HM*/
    /* Eqs. (C9, C11, C13, C15) of https://arxiv.org/pdf/1803.10701.pdf for SEOBNRv4HM*/
    """

    P1coeff00[(2,2)] = 0.147584
    P1coeff01[(2,2)] = 0.00779176
    P1coeff02[(2,2)] = -0.0244358
    P1coeff10[(2,2)] = 0.263456
    P1coeff11[(2,2)] = -0.120853
    P1coeff20[(2,2)] = -0.808987

    P1coeff00[(2,1)] = 0.15601401627613815
    P1coeff01[(2,1)] = 0.10219957917717622
    P1coeff10[(2,1)] = 0.023346852452720928
    P1coeff11[(2,1)] = -0.9435308286367039
    P1coeff20[(2,1)] = 0.15326558178697175
    P1coeff21[(2,1)] = 1.7979082057513565

    P1coeff00[(3,3)] = 0.11085299117493969
    P1coeff01[(3,3)] = 0.018959099261813613
    P1coeff10[(3,3)] = 0.9999800463662053
    P1coeff11[(3,3)] = -0.729149797691242
    P1coeff20[(3,3)] = -3.3983315694441125
    P1coeff21[(3,3)] = 2.5192011762934037

    P1coeff00[(4,4)] = 0.11498976343440313
    P1coeff01[(4,4)] = 0.008389519706605305
    P1coeff10[(4,4)] = 1.6126522800609633
    P1coeff11[(4,4)] = -0.8069979888526699
    P1coeff20[(4,4)] = -6.255895564079467
    P1coeff21[(4,4)] = 7.595651881827078
    P1coeff31[(4,4)] = -19.32367406125053

    P1coeff00[(5,5)] = 0.16465380962882128
    P1coeff01[(5,5)] = -0.026574817803812007
    P1coeff10[(5,5)] = -0.19184523794508765
    P1coeff11[(5,5)] = -0.05519618962738479
    P1coeff20[(5,5)] = 0.33328424135336965
    P1coeff21[(5,5)] = 0.3194274548351241

    out = xp.asarray([P1coeff00[(ell_i.item(),m_i.item())] + P1coeff01[(ell_i.item(),m_i.item())] * chi + P1coeff02[(ell_i.item(),m_i.item())] * chi * chi + P1coeff10[(ell_i.item(),m_i.item())] * eta +P1coeff11[(ell_i.item(),m_i.item())] * eta * chi + P1coeff20[(ell_i.item(),m_i.item())] * eta * eta + P1coeff21[(ell_i.item(),m_i.item())]*eta*eta*chi + P1coeff31[(ell_i.item(),m_i.item())]*eta*eta*eta*chi for (ell_i, m_i) in zip(ell, m)]).T

    return out


def EOBCalculateRDPhaseCoefficient2(ell, m,  eta,  chi, xp=None):

    if xp is None:
        xp = np

    P2coeff00={}
    P2coeff01={}
    P2coeff02={}
    P2coeff12={}
    P2coeff10={}
    P2coeff11={}
    P2coeff20={}
    P2coeff21={}
    P2coeff31={}
    modes = [(2,2),(2,1),(3,3),(4,4),(5,5)]
    for mode in modes:
        P2coeff00[mode] = 0.0
        P2coeff01[mode] = 0.0
        P2coeff02[mode] = 0.0
        P2coeff12[mode] = 0.0
        P2coeff10[mode] = 0.0
        P2coeff11[mode] = 0.0
        P2coeff20[mode] = 0.0
        P2coeff21[mode] = 0.0
        P2coeff31[mode] = 0.0
    """
    /* Fit coefficients that enter the phase of the RD signal of SEOBNRv4 and SEOBNRv4HM*/
    /* Eq. (63) of https://dcc.ligo.org/T1600383 for SEOBNRv4 and 22 mode SEOBNRv4HM*/
    /* Eqs. (C10, C12, C14, C16) of https://arxiv.org/pdf/1803.10701.pdf for SEOBNRv4HM*/
    """
    P2coeff00[(2,2)] = 2.46654
    P2coeff01[(2,2)] = 3.13067
    P2coeff02[(2,2)] = 0.581626
    P2coeff10[(2,2)] = -6.99396
    P2coeff11[(2,2)] = -9.61861
    P2coeff20[(2,2)] = 17.5646

    P2coeff00[(2,1)] = 2.7886287922318105
    P2coeff01[(2,1)] = 4.29290053494256
    P2coeff10[(2,1)] = -0.8145406685320334
    P2coeff11[(2,1)] = -15.93796979597706
    P2coeff20[(2,1)] = 5.549338798935832
    P2coeff21[(2,1)] = 12.649775582333442
    P2coeff02[(2,1)] = 2.5582321247274726
    P2coeff12[(2,1)] = -10.232928498772893

    P2coeff00[(3,3)] = 2.7825237371542735
    P2coeff01[(3,3)] = 2.8796835808075003
    P2coeff10[(3,3)] = -7.844741660437831
    P2coeff11[(3,3)] = -34.7670039322078
    P2coeff20[(3,3)] = 27.181024362399302
    P2coeff21[(3,3)] = 127.13948436435182

    P2coeff00[(4,4)] = 3.111817347262856
    P2coeff01[(4,4)] = 5.399341180960216
    P2coeff10[(4,4)] = 15.885333959709488
    P2coeff11[(4,4)] = -87.92421137153823
    P2coeff20[(4,4)] = -79.64931908155609
    P2coeff21[(4,4)] = 657.7156442271963
    P2coeff31[(4,4)] = -1555.2968529739226
    P2coeff02[(4,4)] = 2.3832321567874686
    P2coeff12[(4,4)] = -9.532928476043567

    P2coeff00[(5,5)] = 11.102447263357977
    P2coeff01[(5,5)] = 6.015112119742853
    P2coeff10[(5,5)] = -58.605776859097084
    P2coeff11[(5,5)] = -81.68032025902797
    P2coeff20[(5,5)] = 176.60643662729498
    P2coeff21[(5,5)] = 266.47345742836745

    out = xp.asarray([P2coeff00[(ell_i.item(),m_i.item())] + P2coeff01[(ell_i.item(),m_i.item())] * chi + P2coeff02[(ell_i.item(),m_i.item())] * chi * chi + P2coeff12[(ell_i.item(),m_i.item())] * chi * chi * eta + P2coeff10[(ell_i.item(),m_i.item())] * eta +P2coeff11[(ell_i.item(),m_i.item())] * eta * chi + P2coeff20[(ell_i.item(),m_i.item())] * eta * eta + P2coeff21[(ell_i.item(),m_i.item())]*eta*eta*chi + P2coeff31[(ell_i.item(),m_i.item())]*eta*eta*eta*chi for (ell_i, m_i) in zip(ell, m)]).T

    return out

def EOBCalculateRDAmplitudeConstraintedCoefficient1(c1f,c2f,sigmaR,amp,damp,eta, xp=None):
    if xp is None:
        xp = np

    c1c = 1/(c1f*eta)*(damp-sigmaR*amp)*xp.cosh(c2f)**2
    return c1c

def EOBCalculateRDAmplitudeConstraintedCoefficient2(c1f,c2f,sigmaR,amp,damp,eta, xp=None):
    if xp is None:
        xp = np
    c2c =  amp/eta-1/(c1f*eta)*(damp-sigmaR*amp)*xp.cosh(c2f)*xp.sinh(c2f)
    return c2c

def EOBCalculateRDPhaseConstraintedCoefficient1(d1f,d2f,sigmaI,omega, xp=None):
    if xp is None:
        xp = np
    d1c = (omega - sigmaI)*(1+d2f)/(d1f*d2f)
    return d1c
