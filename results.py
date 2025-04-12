import numpy as np
import matplotlib.pyplot as plt
import copy

from thermo_funcs import two_terminals
from scipy.optimize import fsolve

def thermal_with_lorentz(mu, T, width, height, position):
    lorentz_max = 2/(np.pi * width)
    height_factor = height/lorentz_max

    lorentzian = lambda E: height_factor * (width/((E-position)**2 + width**2))
    reflect = lambda E: -height_factor * (width/((E+position)**2 + width**2))
    fermi = lambda E: two_terminals.fermi_dist(E,mu,T)
    dist = lambda E: fermi(E) + lorentzian(E) + reflect(E)
    # if dist(position) < 0 or dist(position) > 1:
    #     print("Warning! Invalid occupation function", dist(position))
    
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
        
        # Make sure to update the fermi distribution in the system
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
    Es = np.linspace(sys_copy.E_low, sys_copy.E_high, 100)
    plt.plot(Es, sys_copy.occupf_R(Es), label = "probe start")
    print("Old energy/particle currents: ",sys_copy.calc_left_energy_current(), sys_copy.calc_left_particle_current())
    print("Old mu/TR: ", sys_copy.muR, sys_copy.TR)
    buttiker_probe(sys_copy)
    
    print("New mu/TR: ",sys_copy.muR, sys_copy.TR)
    plt.plot(Es, sys_copy.occupf_R(Es), label = "probe end")
    print("New energy/particle currents: ",sys_copy.calc_left_energy_current(), sys_copy.calc_left_particle_current())
    plt.plot(Es, sys_copy.occupf_L(Es), label = "left occup")
    plt.legend()
    #plt.show()

def check_avg_optimization(system:two_terminals, secondary = False):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf, roots = sys_copy.constrained_current_max()
    #max_cool, mc_transf = sys_copy.jRmax()
    target = 0.5*max_cool
    
    res = sys_copy.optimize_for_avg(target, secondary = secondary)
    transf_avg = sys_copy.transf #thermal_left._transmission_avg(0.1,thermal_left.coeff_con, thermal_left.coeff_avg)#
    #transf_avg = sys_copy._transmission_avg(1, sys_copy.coeff_con, sys_copy.coeff_avg)
    #thermal_left.transf = mc_transf

    #plt.plot(Es, *)
    #plt.plot(Es,thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)))
    #plt.hlines(0, E_low, E_high)
    #plt.show()

    #transf_avg = lambda Es: np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es) - 0.1,0)*np.heaviside(thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)*np.heaviside(thermal_left.coeff_avg(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)
    print("Target: ", target)
    print("Difference from target: ", sys_copy._current_integral(sys_copy.coeff_con)- target)
    print("Efficiency: ", sys_copy.get_efficiency())
    plt.plot(Es, transf_avg(Es), label = "transf avg", zorder = 4)
    return res
    #plt.plot(Es, np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es)-0.3,0))
    #plt.show()
def check_noise_optimization(system:two_terminals):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf, roots = sys_copy.constrained_current_max()
    target = 0.5*max_cool
    
    C = sys_copy.optimize_for_noise(target)
    transf_noise = sys_copy.transf
    #thermal_left.transf = mc_transf
    #transf_noise = sys_copy._transmission_noise(0.1, sys_copy.coeff_noise, sys_copy.coeff_con)
    #plt.plot(Es, *)
    #plt.plot(Es,thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)))
    #plt.hlines(0, E_low, E_high)
    #plt.show()

    #transf_avg = lambda Es: np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es) - 0.1,0)*np.heaviside(thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)*np.heaviside(thermal_left.coeff_avg(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)
    print("Target: ", target)
    print("Difference from target: ", sys_copy._current_integral(sys_copy.coeff_con)- target)
    plt.plot(Es, transf_noise(Es), label = "transf noise", zorder = 5)
    #plt.plot(Es, np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es)-0.3,0))
    #plt.show()
def check_product_optimization(system:two_terminals):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf, roots = sys_copy.constrained_current_max()
    target = 0.5*max_cool
    #sys_copy.debug = False
    C = sys_copy.optimize_for_product(target)#,[0.01978747, 0.00269362, 0.04305463])
    print("Thetas: ", C)
    transf_prod = sys_copy.transf #thermal_left._transmission_avg(0.1,thermal_left.coeff_con, thermal_left.coeff_avg)#
    #transf_prod = sys_copy._transmission_product([0.01978747, 0.00269362, 0.04305463])
    #print(sys_copy.optimize_for_product.opt_func([0.01978747, 0.00269362, 0.04305463]))
    #thermal_left.transf = mc_transf

    #plt.plot(Es, *)
    #plt.plot(Es,thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)))
    #plt.hlines(0, E_low, E_high)
    #plt.show()

    #transf_avg = lambda Es: np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es) - 0.1,0)*np.heaviside(thermal_left.coeff_con(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)*np.heaviside(thermal_left.coeff_avg(Es)*(thermal_left.occupf_L(Es) - thermal_left.occupf_R(Es)),0)
    print("Target: ", target)
    print("Difference from target: ", sys_copy._current_integral(sys_copy.coeff_con)- target)
    
    plt.plot(Es, transf_prod(Es), '--',label = "transf prod", zorder = 6)
    #plt.plot(Es, np.heaviside(thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es)-0.3,0))
    
    #plt.show()

def plot_max(system:two_terminals):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf, roots = sys_copy.constrained_current_max()
    plt.plot(Es, sys_copy.occupf_L(Es), label = "fL", zorder = 2)
    plt.plot(Es, sys_copy.occupf_R(Es), label = "fR", zorder = 1)
    plt.plot(Es, mc_transf(Es), label = "mc transf" ,zorder = 3)


class PowerPlot:
    def __init__(self, system_th:two_terminals, system_nth:two_terminals, verbose = False):
        self.system_th = copy.deepcopy(system_th)
        self.system_nth = copy.deepcopy(system_nth)
        self.coeff_avgs = [system_th.left_entropy_coeff, lambda E: -system_th.left_noneq_free_coeff(E)]
        #self.fig, self.axs = plt.subplots(1,3)
        self.jmax_th, self.transf_max_th, roots = self.system_th.constrained_current_max()
        self.targets_th = np.linspace(0.01*self.jmax_th,self.jmax_th,10)
        self.system_th.adjust_limits()
        self.system_nth.adjust_limits()

        self.jmax_nth, self.transf_max_nth, roots = self.system_nth.constrained_current_max()
        self.targets_nth = np.linspace(0.01*self.jmax_nth,self.jmax_nth,10)
        self.verbose = verbose
    #def define_figure(self):
    def make_figure(self, make_eff = True, make_noise = True, make_product = True):
        #make_list = [make_eff, make_noise, make_product]
        fig, axs = plt.subplots(1, 3, figsize = (10,5))
        if make_eff:
            self.eff_plot(self.system_th, axs[0], self.targets_th, label = "thermal")
            self.eff_plot(self.system_nth, axs[0], self.targets_nth, label = "nonthermal")
            axs[0].legend()
        if make_noise:
            self.noise_plot(self.system_th, axs[1], self.targets_th, label = "thermal")
            self.noise_plot(self.system_nth, axs[1], self.targets_nth, label = "nonthermal")
            axs[1].legend()
        if make_product:
            self.product_plot(self.system_th, axs[2], self.targets_th, label = "thermal")
            self.product_plot(self.system_nth, axs[2], self.targets_nth, label = "nonthermal")
            axs[2].legend()
        
        
        plt.tight_layout()
        return fig

    def make_dist_figure(self):
        Es = np.linspace(self.system_nth.E_low, self.system_nth.E_high)
        fig = plt.figure()
        plt.plot(Es, self.system_nth.occupf_L(Es), label = "Non_thermal")
        plt.plot(Es, self.system_th.occupf_L(Es), label = "Thermal probe")
        plt.plot(Es, self.system_th.occupf_R(Es), label = "Right")
        plt.legend()
        plt.xlabel("E")
        plt.ylabel("Occupation")
        plt.title("Occupation functions")
        plt.tight_layout()
        return fig

    def eff_plot(self,system, axs, targets, label):
        if self.verbose:
            print("Making efficiency plot for ", label)
        JR_list = []
        avg_list = []
        for target in targets:
            if self.verbose:
                print("On target ", target)
            system.optimize_for_avg(target)
            JR_list.append(system._current_integral(system.coeff_con))
            avg_list.append(system._current_integral(system.coeff_avg))
        JR_arr = np.array(JR_list)
        avg_arr = np.array(avg_list)
        eff_arr = JR_arr/avg_arr
        axs.plot(JR_arr, eff_arr, label=label)
        axs.set_title("Efficiency vs cooling power")
        axs.set_xlabel("J_R")
        axs.set_ylabel(r"$\eta$")

    def noise_plot(self, system, axs, targets, label):
        if self.verbose:
            print("Making noise plot for ", label)
        JR_list = []
        noise_list = []
        for target in targets:
            if self.verbose:
                print("On target ", target)

            system.optimize_for_noise(target)
            JR_list.append(system._current_integral(system.coeff_con))
            noise_list.append(system.noise_cont(system.coeff_noise))
        JR_arr = np.array(JR_list)
        noise_arr = np.array(noise_list)
        
        axs.plot(JR_arr, noise_arr, label = label)
        axs.set_title("Noise vs cooling power")
        axs.set_xlabel("J_R")
        axs.set_ylabel(r"$S$")

    def product_plot(self, system, axs, targets, label):
        if self.verbose:
            print("Making product plot for ", label)

        JR_list = []
        product_list = []
        for target in targets:
            if self.verbose:
                print("On target ", target)

            system.optimize_for_product(target)
            JR_list.append(system._current_integral(system.coeff_con))
            product_list.append(system._current_integral(system.coeff_avg)*system.noise_cont(system.coeff_noise))
        JR_arr = np.array(JR_list)
        product_arr = np.array(product_list)
        eff_arr = product_arr/JR_arr
        axs.plot(JR_arr, eff_arr, label=label)
        axs.set_title("Product vs cooling power")
        axs.set_xlabel("J_R")
        axs.set_ylabel(r"$S/ \eta$")


if __name__ == "__main__":
    check_probe= False
    check_avg = True
    check_noise = False
    check_product = False
    check_max = True
    midT = 1
    deltaT = 0.5
    deltamu = -1
    muR = 0
    TR = midT-deltaT
    muL = muR + deltamu
    TL = midT + deltaT

    E_low = -0.5
    E_high = 1.5
    Es = np.linspace(E_low, E_high,1000)

    occupf_L_nth = thermal_with_lorentz(muL, TL, 0.1 ,-0.1, 1)

    left_virtual = two_terminals(-1, 2, occupf_L = occupf_L_nth, muL=muL, muR=muR, TL=TL, TR=TR)

    
    if check_probe:
        check_buttiker_probe(left_virtual)
        plt.show()

    buttiker_probe(left_virtual)

    thermal_left = two_terminals(E_low, E_high, muL=left_virtual.muR, TL = left_virtual.TR, muR = muR, TR = TR, N = 1, subdivide=False, debug=False)
    nonthermal_left = two_terminals(E_low, E_high, occupf_L= occupf_L_nth, muL=muL, TL = TL, muR = muR, TR = TR, N = 1, subdivide=False, debug = False)

    thermal_left.coeff_con = lambda E: -thermal_left.right_entropy_coeff(E)
    thermal_left.coeff_avg = lambda E: thermal_left.left_entropy_coeff(E)#thermal_left.left_noneq_free_coeff(E)
    thermal_left.coeff_noise = lambda E: -thermal_left.right_entropy_coeff(E)

    nonthermal_left.coeff_con = lambda E: -nonthermal_left.right_entropy_coeff(E)
    nonthermal_left.coeff_avg = lambda E: nonthermal_left.left_entropy_coeff(E)#thermal_left.left_noneq_free_coeff(E)
    nonthermal_left.coeff_noise = lambda E: -nonthermal_left.right_entropy_coeff(E)


    active_system = thermal_left

    test_occup = lambda E,muL:thermal_with_lorentz(muL, TL, 0.1 ,-0.1, 1)(E)#lambda E, muL: two_terminals.fermi_dist(E,muL, TL)
    #active_system.set_occupf_L(test_occup, -1)
    #active_system.E_low = -5
    #active_system.E_high = 5
    active_system.debug = True
    if check_avg:
        #plt.plot(Es,thermal_left.coeff_con(Es)/thermal_left.coeff_avg(Es), label = "Comp")
        plt.plot(Es,active_system.coeff_con(Es)*(active_system.occupf_L(Es)- active_system.occupf_R(Es)), label = "con")
        #plt.plot(Es,thermal_left.right_heat_coeff(Es)/thermal_left.TR*(thermal_left.occupf_L(Es)- thermal_left.occupf_R(Es)), label = "con 2")
        #plt.plot(Es,thermal_left.left_heat_coeff(Es)/thermal_left.TL*(thermal_left.occupf_L(Es)- thermal_left.occupf_R(Es)), label = "avg")
        plt.plot(Es,active_system.coeff_avg(Es)*(active_system.occupf_L(Es)- active_system.occupf_R(Es)), label = "avg")
        plt.hlines(0,Es[0], Es[-1])
        plt.legend()
        plt.show()
        res = check_avg_optimization(active_system, False)
        print(res)
        #print("Efficiency old: ", active_system.get_efficiency())
        #active_system.z = res[1]
        #active_system.update_occupf_L()

    if check_noise:
        check_noise_optimization(active_system)

    if check_product:
        check_product_optimization(active_system)
    
    #thermal_left = two_terminals(E_low, E_high, muL=muL, TL =TL, muR = muR, TR = TR, N = 1)
    
    #if check_avg:
    #    check_avg_optimization(nonthermal_left)
    
    if any([check_avg, check_noise, check_product, check_max]):
        plot_max(active_system)
        
        plt.plot(Es, test_occup(Es, -0.5), label = "init")
        plt.legend()
        plt.show()

    # powerPlot = PowerPlot(thermal_left, nonthermal_left, True)
    # fig = powerPlot.make_figure(make_eff=True, make_noise=True, make_product=True)
    # plt.savefig("Test_powerplot.png")

    # fig = powerPlot.make_dist_figure()
    # plt.savefig("test_dist.png")