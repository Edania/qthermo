import numpy as np
import thermo_funcs_two as tf
import matplotlib.pyplot as plt

def thermal_with_lorentz(mu, T, width, height, position):
    lorentzian = lambda E: height * (width/((E-position)**2 + width**2))
    fermi = lambda E: tf.fermi_dist(E,mu,T)
    dist = lambda E: fermi(E) + lorentzian(E)
    return dist

if __name__ == "__main__":
    midT = 1
    deltaT = 0.5
    deltamu = -0.5
    muR = 0
    TR = midT-deltaT
    muL = muR + deltamu
    TL = midT + deltaT

    E_low = -1
    E_high = 5

    occupf_L = thermal_with_lorentz(muL, TL, 1 ,1, 1)

    left_virtual = tf.two_terminals(E_low, E_high, occupf_L = occupf_L)
    