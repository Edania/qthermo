import numpy as np
import thermo_funcs as tf
from scipy.optimize import minimize

def l_occup(E, muL, TL):
    return tf.fermi_dist(E,muL,TL)

def wrapper_chunk_integral(transf,E_mids,muL, TL ,muR, TR, occupf_L):
    return -tf.slice_current_integral(transf,E_mids,muL, TL ,muR, TR,occupf_L,type = "heatR")

if __name__ == "__main__":
    muR = 0
    muL = -1
    TL = 2
    TR = 1
    E_range = np.linspace(-10,10,100)
    init_transf = np.random.uniform(0,1,100)
    res = minimize(wrapper_chunk_integral, init_transf, args=(E_range,muL, TL ,muR, TR, l_occup))
