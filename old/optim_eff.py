import numpy as np
import thermo_funcs as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

kb = 1
h = 1
e = 1

muR = 0
TR = 1

def entropy_coeff(E, occup_L):
    coeff = -kb*np.log(occup_L(E)/(1-occup_L(E)))
    return coeff


def slice_current_integral(transf,E_mids, occup_L, occup_R,type = "electric", return_integrands = False, TL = TR):
    if type == "electric":
        coeff = lambda E: e
    elif type == "heat":
        coeff = lambda E: TL*entropy_coeff(E,occup_L)
    elif type == "heatR":
        coeff = lambda E: -(E-muR)
    elif type == "energy":
        coeff = lambda E: E
    elif type == "entropyL":
        coeff = lambda E: -entropy_coeff(E,occup_L)
    elif type == "entropyR":
        coeff = lambda E: entropy_coeff(E, occup_R)
    else:
        print("Invalid current type")
        return -1
    occupdiff = occup_L(E_mids)- occup_R(E_mids)
    deltaE = E_mids[1]-E_mids[0]
    integrands = 1/h*coeff(E_mids)*transf*occupdiff*deltaE
    current = np.sum(integrands)   
    if return_integrands:
        return current, integrands
    return current
def init_occup_L(E, mu, T, c, n_f = 5):
    f = tf.fermi_dist(E,mu,T)
    f_tot = np.sum((c*f.T).T, axis = 0)
    return f_tot

def occup_L_gauss(E):
    pdf = stats.norm.pdf(E, loc = 1, scale = 5)
    pdf = 0.99*pdf/np.max(pdf)
    return pdf

def wrapper_slice_integral(transf,E_mids, occup_L,occup_R):
    return -slice_current_integral(transf,E_mids,occup_L,occup_R,type = "heatR")


def entropy_COP(E, occup_L, occup_R):
    entropy_L = slice_current_integral(transf, E, occup_L, occup_R, type = "entropyL")
    entropy_R = slice_current_integral(transf, E, occup_L, occup_R, type = "entropyR")
    return -entropy_R/entropy_L

def free_energy_COP(E, occup_L, occup_R):
    heatR = slice_current_integral(transf, E, occup_L, occup_R, type = "heatR")
    energy = slice_current_integral(transf, E, occup_L, occup_R, type = "energy")
    entropy_L = slice_current_integral(transf, E, occup_L, occup_R, type = "entropyL")
    free_energy = energy-entropy_L*TR
    return - heatR/free_energy

if __name__ == "__main__":
    E_range = np.linspace(-10,20,100)
    init_transf = np.random.uniform(0,1,100)
    n_f = 5
    mu = np.random.uniform(-2,0,n_f).reshape(-1,1)#np.linspace(-5,0,n_f).reshape(-1,1)
    T = np.random.uniform(TR,2,n_f).reshape(-1,1)#np.linspace(0.1,10,n_f).reshape(-1,1)
    c = np.random.uniform(0,1,n_f)
    c = c/np.sum(c)
    

    occup_L = lambda E: init_occup_L(E, mu,T,c,n_f)
    #occup_L_func = lambda E: occup_L
    #occup_L_func = occup_L_gauss
    occup_R = lambda E: tf.fermi_dist(E, muR, TR)
    res = minimize(wrapper_slice_integral, init_transf, args=(E_range, occup_L, occup_R),bounds=(((0,1),)*len(E_range)))
    transf = res.x

    #Entropy check
    entropy_L = slice_current_integral(transf, E_range, occup_L, occup_R, type = "entropyL")
    entropy_R = slice_current_integral(transf, E_range, occup_L, occup_R, type = "entropyR")
    entropy_check = entropy_L + entropy_R >= 0
    print("Is entropy increasing?", entropy_check)

    #particle_L = slice_current_integral(transf, E_range, occup_L_func, occup_R, type = "electric")
    #particle_R = slice_current_integral(transf, E_range, occup_L_func, occup_R, type = "electric")
    heatR = slice_current_integral(transf, E_range, occup_L, occup_R, type = "heatR")
    energy = slice_current_integral(transf, E_range, occup_L, occup_R, type = "energy")

    ent_COP = entropy_COP(E_range, occup_L, occup_R)
    free_COP = free_energy_COP(E_range, occup_L, occup_R)

    print("Cooling power (heat flow from R): ", heatR)
    print("The entropy COP is", ent_COP)
    print("The free energy COP is", free_COP)
    #print("Energy flow from L to scatterer: ", energy)
    fig = plt.figure()
    plt.plot(E_range, np.array([transf, occup_L(E_range), occup_R(E_range)]).T, label = ["transf", "gL", "fR"])
    plt.legend()
    plt.show()
