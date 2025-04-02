import numpy as np
import matplotlib.pyplot as plt
import copy

from thermo_funcs_three import two_terminals
from scipy.optimize import fsolve
def thermal_with_lorentz(mu, T, width, height, position):
    lorentzian = lambda E: height * (width/((E-position)**2 + width**2))
    fermi = lambda E: two_terminals.fermi_dist(E,mu,T)
    dist = lambda E: fermi(E) + lorentzian(E)
    return dist

def buttiker_probe(system:two_terminals):
    mu_init = system.muR
    T_init = system.TR
    sys_copy = copy.deepcopy(system)
    mus = np.linspace(-1,1,10)
    def find_mu_T(params):
        mu = params[0]
        #mu = mus[np.random.randint(0,9)]
        T = params[1]
        sys_copy.muR = mu
        sys_copy.TR = T
        sys_copy.set_fermi_dist_right()
        I_E = sys_copy.calc_left_energy_current()
        I_N = sys_copy.calc_left_particle_current()
        return I_E, I_N
    params_init = [mu_init, T_init]
    res = fsolve(find_mu_T, params_init, factor=0.1)
    system.muR = res[0]
    system.TR = res[1]
    system.set_fermi_dist_right()

def check_buttiker_probe(system):
    sys_copy = copy.deepcopy(system)
    plt.plot(Es, sys_copy.occupf_R(Es), label = "start right")
    print("Old energy/particle currents: ",sys_copy.calc_left_energy_current(), sys_copy.calc_left_particle_current())
    print("Old mu/TR: ", sys_copy.muR, sys_copy.TR)
    buttiker_probe(sys_copy)
    
    print("New mu/TR: ",sys_copy.muR, sys_copy.TR)
    plt.plot(Es, sys_copy.occupf_R(Es), label = "found right")
    print("New energy/particle currents: ",sys_copy.calc_left_energy_current(), sys_copy.calc_left_particle_current())
    plt.plot(Es, sys_copy.occupf_L(Es), label = "left occup")
    plt.legend()
    plt.show()

def check_avg_optimization(system:two_terminals):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf = sys_copy.jRmax()
    target = 0.5*max_cool
    
    C = sys_copy.optimize_for_avg(0.1,target)
    transf_avg = sys_copy.transf #thermal_left._transmission_avg(0.1,thermal_left.coeff_con, thermal_left.coeff_avg)#
    #thermal_left.transf = mc_transf

    #plt.plot(Es, *)
    #plt.plot(Es,thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)))
    #plt.hlines(0, E_low, E_high)
    #plt.show()

    #transf_avg = lambda Es: np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es) - 0.1,0)*np.heaviside(thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)*np.heaviside(thermal_left.coeff_avg(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)
    print("Target: ", target)
    print("Difference from target: ", sys_copy.calc_right_heat_current()- target)
    plt.plot(Es, sys_copy.occupf_L(Es), label = "fL")
    plt.plot(Es, sys_copy.occupf_R(Es), label = "fR")
    plt.plot(Es, transf_avg(Es), label = "transf avg")
    plt.plot(Es, mc_transf(Es), label = "mc transf")
    #plt.plot(Es, np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es)-0.3,0))
    plt.legend()
    plt.show()
def check_noise_optimization(system:two_terminals):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf = sys_copy.jRmax()
    target = 0.5*max_cool
    
    C = sys_copy.optimize_for_noise(0.3, target)
    transf_avg = sys_copy.transf #thermal_left._transmission_avg(0.1,thermal_left.coeff_con, thermal_left.coeff_avg)#
    #thermal_left.transf = mc_transf

    #plt.plot(Es, *)
    #plt.plot(Es,thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)))
    #plt.hlines(0, E_low, E_high)
    #plt.show()

    #transf_avg = lambda Es: np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es) - 0.1,0)*np.heaviside(thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)*np.heaviside(thermal_left.coeff_avg(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)
    print("Target: ", target)
    print("Difference from target: ", sys_copy.calc_right_heat_current()- target)
    plt.plot(Es, sys_copy.occupf_L(Es), label = "fL")
    plt.plot(Es, sys_copy.occupf_R(Es), label = "fR")
    plt.plot(Es, transf_avg(Es), label = "transf avg")
    plt.plot(Es, mc_transf(Es), label = "mc transf")
    #plt.plot(Es, np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es)-0.3,0))
    plt.legend()
    plt.show()
def check_product_optimization(system:two_terminals):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf = sys_copy.jRmax()
    target = 0.5*max_cool
    
    C = sys_copy.optimize_for_avg(0.1,target)
    transf_avg = sys_copy.transf #thermal_left._transmission_avg(0.1,thermal_left.coeff_con, thermal_left.coeff_avg)#
    #thermal_left.transf = mc_transf

    #plt.plot(Es, *)
    #plt.plot(Es,thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)))
    #plt.hlines(0, E_low, E_high)
    #plt.show()

    #transf_avg = lambda Es: np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es) - 0.1,0)*np.heaviside(thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)*np.heaviside(thermal_left.coeff_avg(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)
    print("Target: ", target)
    print("Difference from target: ", sys_copy.calc_right_heat_current()- target)
    plt.plot(Es, sys_copy.occupf_L(Es), label = "fL")
    plt.plot(Es, sys_copy.occupf_R(Es), label = "fR")
    plt.plot(Es, transf_avg(Es), label = "transf avg")
    plt.plot(Es, mc_transf(Es), label = "mc transf")
    #plt.plot(Es, np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es)-0.3,0))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    check_probe = False
    check_avg = True
    check_noise = True
    check_product = True
    midT = 1
    deltaT = 0.5
    deltamu = -1
    muR = 0
    TR = midT-deltaT
    muL = muR + deltamu
    TL = midT + deltaT

    E_low = -0.1
    E_high = 1
    Es = np.linspace(E_low, E_high,1000)

    occupf_L_nth = thermal_with_lorentz(muL, TL, 0.2 ,0.05, 1)

    left_virtual = two_terminals(E_low, E_high, occupf_L = occupf_L_nth, muL=muL, muR=muR, TL=TL, TR=TR)

    
    if check_probe:
        check_buttiker_probe(left_virtual)

    buttiker_probe(left_virtual)

    thermal_left = two_terminals(E_low, E_high, muL=left_virtual.muR, TL = left_virtual.TR, muR = muR, TR = TR, N = 1, subdivide=False)
    if check_avg:
        check_avg_optimization(thermal_left)

    if check_noise:
        check_noise_optimization(thermal_left)

    #thermal_left = two_terminals(E_low, E_high, muL=muL, TL =TL, muR = muR, TR = TR, N = 1)
    nonthermal_left = two_terminals(E_low, E_high, occupf_L= occupf_L_nth, muL=muL, TL = TL, muR = muR, TR = TR, N = 1)

    if check_avg:
        check_avg_optimization(nonthermal_left)