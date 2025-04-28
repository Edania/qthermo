import numpy as np
import matplotlib.pyplot as plt
import copy

from thermo_funcs import two_terminals
from scipy.optimize import fsolve, minimize
from scipy import integrate

def thermal_with_lorentz(mu, T, width, height, position):
    lorentz_max = 2/(np.pi * width)
    height_factor = height/lorentz_max
    rel_position = position

    lorentzian = lambda E: height_factor * (width/((E-position-mu)**2 + width**2))
    reflect = lambda E: -height_factor * (width/((E+position-mu)**2 + width**2))
    fermi = lambda E: two_terminals.fermi_dist(E,mu,T)
    dist = lambda E: fermi(E) + lorentzian(E) + reflect(E)
    # if dist(position) < 0 or dist(position) > 1:
    #     print("Warning! Invalid occupation function", dist(position))
    
    return dist

def two_thermals(mu1, T1, mu2, T2):
    dist1 = lambda E: two_terminals.fermi_dist(E,mu1,T1)
    dist2 = lambda E: two_terminals.fermi_dist(E,mu2,T2)
    return lambda E: 0.5*(dist1(E) + dist2(E))

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
    Es = np.linspace(sys_copy.E_low, sys_copy.E_high, 1000)
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

def check_avg_optimization(system:two_terminals, secondary = False, C_init = None, target = 0.1):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf, roots = sys_copy.constrained_current_max()
    #max_cool, mc_transf = sys_copy.jRmax()
    target = 0.5*max_cool
    
    res = sys_copy.optimize_for_avg(target, secondary = secondary, C_init = C_init)
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
def check_noise_optimization(system:two_terminals, target_coeff = 0.5):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf, roots = sys_copy.constrained_current_max()
    target = target_coeff*max_cool
    
    C = sys_copy.optimize_for_noise(target,C_init = 10)
    transf_noise = sys_copy.transf
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
    print("Max cool: ", max_cool)
    
    plt.plot(Es, sys_copy.occupf_L(Es), label = "fL", zorder = 2)
    plt.plot(Es, sys_copy.occupf_R(Es), label = "fR", zorder = 1)
    plt.plot(Es, mc_transf(Es), label = "mc transf" ,zorder = 3)
    #plt.plot(Es, sys_copy.occupf_L(Es)-sys_copy.occupf_R(Es), label = "occupdiff")
def check_Jrmax(system:two_terminals):
    sys_copy = copy.deepcopy(system)
    
    mus = np.linspace(-5,5,20)
    jr_list = []
    for mu in mus:
        sys_copy.muR = mu
        sys_copy.set_fermi_dist_right()
        jrmax, transf_max, roots = sys_copy.constrained_current_max()
        jr_list.append(jrmax)
    
    plt.plot(mus, jr_list)
    plt.show()
    plt.plot(Es, transf_max(Es))
    plt.plot(Es, sys_copy.occupf_L(Es))
    plt.plot(Es, sys_copy.occupf_R(Es))
    
    plt.show()

class PowerPlot:
    def __init__(self, system_th:two_terminals, system_nth:two_terminals, verbose = False, n_targets = 10):
        self.system_th = copy.deepcopy(system_th)
        self.system_nth = copy.deepcopy(system_nth)
        self.n_targets = n_targets
        #self.coeff_avgs = [system_th.left_entropy_coeff, lambda E: -system_th.left_noneq_free_coeff(E)]
        #self.fig, self.axs = plt.subplots(1,3)
        self.jmax_th, self.transf_max_th, roots = self.system_th.constrained_current_max()
        self.targets_th = np.linspace(0.01*self.jmax_th,self.jmax_th,n_targets)
        #print(self.jmax_th)
        self.system_th.adjust_limits()
        self.system_nth.adjust_limits()

        self.jmax_nth, self.transf_max_nth, roots = self.system_nth.constrained_current_max()
        self.targets_nth = np.linspace(0.01*self.jmax_nth,self.jmax_nth,n_targets)
        self.verbose = verbose
    #def define_figure(self):
    def make_figure(self, make_eff = True, make_noise = True, make_product = True,
                    filenames = [None]*6):
        #make_list = [make_eff, make_noise, make_product]
        fig, axs = plt.subplots(1, 3, figsize = (10,5))
        if make_eff:
            self.eff_plot(axs[0], label = "thermal",targets = self.targets_th,system = self.system_th,  filename = filenames[0])
            self.eff_plot(system = self.system_nth, axs = axs[0], targets = self.targets_nth, label = "nonthermal", filename=filenames[1])
            axs[0].legend()
        if make_noise:
            self.noise_plot(system=self.system_th, axs=axs[1], targets = self.targets_th, label = "thermal", filename=filenames[2])
            self.noise_plot(system=self.system_nth, axs=axs[1], targets=self.targets_nth, label = "nonthermal", filename=filenames[3])
            axs[1].legend()
        if make_product:
            self.product_plot(system=self.system_th, axs=axs[2], targets = self.targets_th, label = "thermal", filename=filenames[4])
            self.product_plot(system=self.system_nth, axs=axs[2], targets=self.targets_nth, label = "nonthermal", filename=filenames[5])
            axs[2].legend()
        
        
        plt.tight_layout()
        return fig

    def make_dist_figure(self):
        Es = np.linspace(self.system_nth.E_low, self.system_nth.E_high,1000)
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

    def eff_plot(self, axs, label,system=None, targets = None,filename = None):
        if self.verbose:
            print("Making efficiency plot for ", label)
        JR_arr, eff_arr, C_arr = self.get_eff_data(system, targets, filename)
        axs.plot(JR_arr, eff_arr, label=label)
        axs.set_title("Efficiency vs cooling power")
        axs.set_xlabel("J_R")
        axs.set_ylabel(r"$\eta$")

    def noise_plot(self,  axs, label,system=None, targets = None,filename = None):
        if self.verbose:
            print("Making noise plot for ", label)

        JR_arr, noise_arr, C_arr = self.get_noise_data(system, targets, filename)
        axs.plot(JR_arr, noise_arr, label = label)
        axs.set_title("Noise vs cooling power")
        axs.set_xlabel("J_R")
        axs.set_ylabel(r"$S$")

    def product_plot(self,  axs, label,system=None, targets = None,filename = None):
        if self.verbose:
            print("Making product plot for ", label)
        JR_arr, eff_arr, C_arr = self.get_product_data(system, targets, filename)

        axs.plot(JR_arr, eff_arr, label=label)
        axs.set_title("Product vs cooling power")
        axs.set_xlabel("J_R")
        axs.set_ylabel(r"$S/ \eta$")

    def produce_eff_data(self, system:two_terminals, targets):
        if self.verbose:
            print("Producing eff data")
        JR_list = []
        avg_list = []
        C_min = system.C_limit_avg(system.coeff_con, system.coeff_con)
        jmax,_,_ = system.constrained_current_max()
        C_max = system.optimize_for_avg(0.95*jmax, 10)
        C_list = np.geomspace(C_min, C_max, self.n_targets)
        for C in C_list:
            system.set_transmission_avg_opt(C, system.coeff_con, system.coeff_avg)
            JR_list.append(system._current_integral(system.coeff_con))
            avg_list.append(system._current_integral(system.coeff_avg))
            
        # C_list = []
        # for target in targets:
        #     if self.verbose:
        #         print("On target ", target)
        #     C = system.optimize_for_avg(target)
        #     C_list.append(C)
        #     JR_list.append(system._current_integral(system.coeff_con))
        #     avg_list.append(system._current_integral(system.coeff_avg))
        JR_arr = np.array(JR_list)
        avg_arr = np.array(avg_list)
        C_arr = np.array(C_list)
        eff_arr = JR_arr/avg_arr
        return JR_arr, eff_arr, C_arr
    
    def produce_noise_data(self, system:two_terminals, targets):
        if self.verbose:
            print("Producing noise data")

        JR_list = []
        noise_list = []
        C_min = 0.01#system.C_limit_avg(system.coeff_con, system.coeff_con)
        jmax,_,_ = system.constrained_current_max()
        C_max = system.optimize_for_noise(0.95*jmax, 10)
        C_list = np.geomspace(C_min, C_max, self.n_targets)
        for C in C_list:
            system.set_transmission_noise_opt(C)
            JR_list.append(system._current_integral(system.coeff_con))
            # noise_list.append(system._current_integral(system.coeff_avg))     
            # plot_max(system)
            
            # plt.plot(Es, system.transf(Es))
            # plt.show()   
        # for target in targets:
        #     if self.verbose:
        #         print("On target ", target)

        #     C = system.optimize_for_noise(target, targets[-1]-target)
        #     C_list.append(C)
        #     JR_list.append(system._current_integral(system.coeff_con))
            noise_list.append(system.noise_cont(system.coeff_noise))
        JR_arr = np.array(JR_list)
        noise_arr = np.array(noise_list)
        C_arr = np.array(C_list)
        return JR_arr, noise_arr, C_arr

    def produce_product_data(self, system:two_terminals, targets):
        if self.verbose:
            print("Producing product data")

        JR_list = []
        product_list = []
        theta_list = []
        C_min = 0.01#system.C_limit_avg(system.coeff_con, system.coeff_con)
        jmax,_,_ = system.constrained_current_max()
        _,_,C_max = system.optimize_for_product(0.95*jmax)
        C_list = np.geomspace(C_min, C_max, self.n_targets)
        for C in C_list:
            system.set_transmission_product_opt(C)
            JR_list.append(system._current_integral(system.coeff_con))
            # product_list.append(system._current_integral(system.coeff_avg))   
        # for target in targets:
        #     if self.verbose:
        #         print("On target ", target)

        #     thetas = system.optimize_for_product(target)
        #     theta_list.append(thetas)
        #     JR_list.append(system._current_integral(system.coeff_con))
            product_list.append(system._current_integral(system.coeff_avg)*system.noise_cont(system.coeff_noise))
        JR_arr = np.array(JR_list)
        product_arr = np.array(product_list)
        theta_arr = np.array(theta_list)
        eff_arr = product_arr/JR_arr
        return JR_arr, eff_arr, theta_arr

    def get_eff_data(self,system=None, targets = None,filename = None):
        if filename:
            eff_file = np.load(filename)
            JR_arr = eff_file["JR_arr"]
            eff_arr = eff_file["eff_arr"]
            C_arr = eff_file["C_arr"]
        elif system and targets:
            JR_arr, eff_arr, C_arr = self.produce_eff_data(system, targets)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, eff_arr, C_arr

    def get_noise_data(self, system= None, targets = None, filename = None):
        if filename:
            eff_file = np.load(filename)
            JR_arr = eff_file["JR_arr"]
            noise_arr = eff_file["noise_arr"]
            C_arr = eff_file["C_arr"]
        elif system and targets:
            JR_arr, noise_arr, C_arr = self.produce_eff_data(system, targets)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, noise_arr, C_arr

    def get_product_data(self, system = None, targets = None, filename = None):
        if filename:
            eff_file = np.load(filename)
            JR_arr = eff_file["JR_arr"]
            eff_arr = eff_file["eff_arr"]
            theta_arr = eff_file["theta_arr"]
        elif system and targets:
            JR_arr, eff_arr, theta_arr = self.produce_eff_data(system, targets)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, eff_arr, theta_arr

    def save_eff(self, system,targets, filename):
        JR_arr, eff_arr, C_arr = self.produce_eff_data(system, targets)
        np.savez(filename, JR_arr = JR_arr, eff_arr = eff_arr, C_arr = C_arr)
    
    def save_noise(self, system, targets, filename):
        JR_arr, noise_arr, C_arr = self.produce_noise_data(system, targets)
        np.savez(filename, JR_arr = JR_arr, noise_arr = noise_arr, C_arr = C_arr)

    def save_product(self, system, targets, filename):
        JR_arr, eff_arr, theta_arr = self.produce_product_data(system, targets)
        np.savez(filename, JR_arr = JR_arr, eff_arr = eff_arr, theta_arr = theta_arr)



if __name__ == "__main__":
    check_probe= True
    check_avg = True
    check_noise = True
    check_product = True
    check_max = True
    check_cond = True
    testing = False
    midT = 1
    deltaT = 0.5
    deltamu = 0
    muR = -1.5
    TR = midT-deltaT
    muL = 0#muR + deltamu
    TL = midT + deltaT

    E_low = -5
    E_high = 5
    Es = np.linspace(E_low, E_high,1000)

    save_data = True
    load_params = True
    dist_type = "lorentz_dip"

    if load_params:
        th_dist_params = np.load("data/th_params_"+dist_type+".npz")['arr_0']
        muR = th_dist_params[0]
        TR = th_dist_params[1]
        nth_dist_params = np.load("data/nth_params_"+dist_type+".npz")['arr_0']
    else:
        th_dist_params = np.array([muR, TR])
        #nth_dist_params = np.array([muL, TL, 0.1 ,0.3, 1])
        nth_dist_params = np.array([muL, TL, muR, TR])
    occupf_L_nth = thermal_with_lorentz(*nth_dist_params)
    #occupf_L_nth = two_thermals(*nth_dist_params)
    #dist_params = np.array([muL,TL,muR,TR])


    left_virtual = two_terminals(-20, 20, occupf_L = occupf_L_nth, muL=muL, muR=muR, TL=TL, TR=TR)

    if check_probe:
        check_buttiker_probe(left_virtual)
        plt.show()
    buttiker_probe(left_virtual)

    thermal_left = two_terminals(E_low, E_high, muL=left_virtual.muR, TL = left_virtual.TR, muR = muR, TR = TR, N = 1, subdivide=False, debug=False)
    nonthermal_left = two_terminals(E_low, E_high, occupf_L= occupf_L_nth, muL=muL, TL = TL, muR = muR, TR = TR, N = 1, subdivide=False, debug = False)

    thermal_left.coeff_con = lambda E: -thermal_left.right_entropy_coeff(E)
    thermal_left.coeff_avg = lambda E: thermal_left.left_entropy_coeff(E)#thermal_left.left_noneq_free_coeff(E)#
    thermal_left.coeff_noise = lambda E: -thermal_left.right_entropy_coeff(E)

    nonthermal_left.coeff_con = lambda E: -nonthermal_left.right_entropy_coeff(E)
    nonthermal_left.coeff_avg = lambda E: nonthermal_left.left_entropy_coeff(E)#nonthermal_left.left_noneq_free_coeff(E)#
    nonthermal_left.coeff_noise = lambda E: -nonthermal_left.right_entropy_coeff(E)

    #TODO: fix
    thermal_left.adjust_limits()
    nonthermal_left.adjust_limits()

    if testing:
        if check_probe:
            check_buttiker_probe(left_virtual)
            plt.show()
        active_system = nonthermal_left
        active_system.debug = False
        #active_system.muR = 1
        active_system.set_fermi_dist_right()
        active_system.set_occup_roots()
        active_system.adjust_limits()
        Es = np.linspace(active_system.E_low, active_system.E_high,1000)
        
        plt.plot(Es,active_system.coeff_con(Es)*(active_system.occupf_L(Es) - active_system.occupf_R(Es)), label = "con")
        plt.plot(Es,active_system.coeff_avg(Es)*(active_system.occupf_L(Es) - active_system.occupf_R(Es)), label = "avg")
        plt.plot(Es, active_system.left_entropy_coeff(Es)*(active_system.occupf_L(Es) - active_system.occupf_R(Es)), label = "L entropy")
        plt.plot(Es, active_system.left_energy_coeff(Es)*(active_system.occupf_L(Es) - active_system.occupf_R(Es)), label = "L energy")
        plt.hlines(0,Es[0],Es[-1], colors="red")
        plt.legend()
        plt.show()
        if check_cond:
            Cs = np.linspace(-10,10,10)
            #Cs = [0.1]
            for C in Cs:
                #[con_avg, con_noise] = active_system.calc_for_product_determined(C)
                #cond = active_system._product_condition([con_avg, con_noise, C])
                #cond = active_system._avg_condition(C, active_system.coeff_con, active_system.coeff_avg)
                cond = active_system._noise_condition(C)
                plt.plot(Es, cond(Es), label = C)
            
            C_limit = active_system.C_limit_noise()
            print("C limit: ", C_limit)
            #plt.plot(Es,-active_system.coeff_avg(Es) + 10*active_system.coeff_con(Es), label = "coeff con")
            #plt.plot(Es,active_system.coeff_avg(Es), label = "coeff avg")
            plt.hlines(0,Es[0],Es[-1], colors="red")
            plt.legend()
            plt.show()


        if check_avg:
            plot_mus = False
            check_avg_optimization(active_system)
            if plot_mus:
                        
                C_list = []
                Iy_list = []
                Ix_list = []
                dd_list = []
                mus = np.linspace(-0.1,-5,20)
                active_system.E_high = 2
                active_system.E_low = -2
                for mu in mus:
                    active_system.muR = mu
                    active_system.set_fermi_dist_right()
                    
                    #plt.show()
                    Es = np.linspace(-2,4)
                    active_system.adjust_limits()
                    print(active_system.occuproots)
                    C = active_system.optimize_for_best_avg(10)
                    Ix = active_system._current_integral(active_system.coeff_con)
                    Iy = active_system._current_integral(active_system.coeff_avg)
                    C_list.append(C)
                    Iy_list.append(Iy)
                    Ix_list.append(Ix)
                C_list = np.array(C_list)
                Ix_list = np.array(Ix_list)
                Iy_list = np.array(Iy_list)
                eff_list = Ix_list/Iy_list
                plt.plot(mus, eff_list)
                #plt.vlines(mus[np.argmax(0.1/np.array(Iy_list))], 0, 1)
                #plt.show()
                #plt.plot(mus,np.array(C_list)-1)
                #plt.plot(mus, np.array(dd_list)-1)
                plt.show()

                plt.plot(Ix_list, eff_list)
                plt.show()

        if check_noise:
            check_noise_optimization(active_system)

        if check_product:
            # jmax, transf, roots = active_system.constrained_current_max()
            # active_system.transf = transf
            # noise_max = active_system.noise_cont(active_system.coeff_noise)
            # avg_max = active_system._current_integral(active_system.coeff_avg)
            # cond = active_system._product_condition([avg_max/10, noise_max/10, 1])
            # plt.plot(Es, cond(Es))
            # plt.show()
            check_product_optimization(active_system)
        
        if any([check_avg, check_noise, check_product, check_max]):
            plot_max(active_system)
            
            #plt.plot(Es, test_occup(Es, -0.5), label = "init")
            plt.legend()
            plt.show()

    else:
        
        
        filenames = ["data/th_"+dist_type+"_eff.npz","data/nth_"+dist_type+"_eff.npz","data/th_"+dist_type+"_noise.npz",
                     "data/nth_"+dist_type+"_noise.npz","data/th_"+dist_type+"_product.npz","data/nth_"+dist_type+"_product.npz"]
        powerPlot = PowerPlot(thermal_left, nonthermal_left, True, n_targets=20)
        if save_data:
            
            # np.savez("data/th_params_"+dist_type, th_dist_params)
            # np.savez("data/nth_params_"+dist_type, nth_dist_params)
            # powerPlot.save_eff(thermal_left, powerPlot.targets_th, filenames[0])
            # powerPlot.save_eff(nonthermal_left, powerPlot.targets_nth, filenames[1])
            powerPlot.save_noise(thermal_left, powerPlot.targets_th, filenames[2])
            powerPlot.save_noise(nonthermal_left, powerPlot.targets_nth, filenames[3])            
            # powerPlot.save_product(thermal_left, powerPlot.targets_th, filenames[4])
            # powerPlot.save_product(nonthermal_left, powerPlot.targets_nth, filenames[5])
        # else:
        #     np.load()

        fig = powerPlot.make_figure(make_eff=True, make_noise=True, make_product=True, filenames=filenames)
        plt.suptitle("Optimized plots for " + dist_type + r", metric $\dot S_R/\dot S_L$")
        plt.tight_layout()
        plt.savefig("figs/"+dist_type+".png")

        # fig = plt.figure()
        JR_arr, noise_arr, C_arr =powerPlot.get_eff_data(filename=filenames[1])
        print(JR_arr)
        # plt.plot(powerPlot.targets_nth-JR_arr)
        # plt.show()
        fig = powerPlot.make_dist_figure()
        plt.savefig("figs/"+dist_type+"_dist.png")