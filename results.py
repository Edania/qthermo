import numpy as np
import matplotlib.pyplot as plt
import copy

from thermo_funcs import two_terminals
from scipy.optimize import fsolve, minimize
from scipy import integrate

### MATPLOT SETUP ###

cold = "#007EF5"
hot = "#FF4400"
nonthermal = "#9D00FF"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Segoe UI'

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

def buttiker_probe(system:two_terminals, set_right = True):
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


def plot_max(system:two_terminals):
    sys_copy = copy.deepcopy(system)
    max_cool, mc_transf, roots = sys_copy.constrained_current_max()
    sys_copy.adjust_limits(factor = 0)
    print("Max cool: ", max_cool)
    Es = np.linspace(0, 6, 10000)
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
        fig, axs = plt.subplots(1, 3, figsize = (8,4))
        if make_eff:
            self.eff_plot(axs[0], color = hot, label = "thermal",targets = self.targets_th,system = self.system_th,  filename = filenames[0])
            self.eff_plot(system = self.system_nth, color = nonthermal, axs = axs[0], targets = self.targets_nth, label = "nonthermal", filename=filenames[1])
            axs[0].legend()
        if make_noise:
            self.noise_plot(system=self.system_th, color = hot, axs=axs[1], targets = self.targets_th, label = "thermal", filename=filenames[2])
            self.noise_plot(system=self.system_nth, color = nonthermal, axs=axs[1], targets=self.targets_nth, label = "nonthermal", filename=filenames[3])
            axs[1].legend()
        if make_product:
            self.product_plot(system=self.system_th, color = hot, axs=axs[2], targets = self.targets_th, label = "thermal", filename=filenames[4])
            self.product_plot(system=self.system_nth, color = nonthermal, axs=axs[2], targets=self.targets_nth, label = "nonthermal", filename=filenames[5])
            axs[2].legend()
        
        
        plt.tight_layout()
        return fig

    def make_dist_figure(self, E_min = None, E_max = None):
        if E_min == None or E_max == None:
            Es = np.linspace(self.system_nth.E_low, self.system_nth.E_high,1000)
        else:
            Es = np.linspace(E_min, E_max, 1000)
        fig = plt.figure(figsize= (3.5, 3.5))
        plt.plot(Es, self.system_nth.occupf_L(Es), label = "Nonthermal", color = nonthermal, zorder = 3)
        plt.plot(Es, self.system_th.occupf_L(Es), label = "Thermal probe", color = hot, zorder = 2)
        plt.plot(Es, self.system_th.occupf_R(Es), label = "Cold thermal", color = cold, zorder = 1)
        plt.legend()
        plt.xlabel(r"$\varepsilon$ [$k_B T_0$]")
        plt.ylabel("Occupation probability")
        plt.title("Thermal distribution with \n Lorentzian peak and dip")
        plt.tight_layout()
        return fig

    def make_example_figure(self):
        pass

    def eff_plot(self, axs, color, label,system=None, targets = None,filename = None):
        if self.verbose:
            print("Making efficiency plot for ", label)
        JR_arr, eff_arr, C_arr = self.get_eff_data(system, targets, filename)
        axs.plot(JR_arr, eff_arr, label=label, color = color)
        axs.set_title("Efficiency vs cooling power")
        axs.set_xlabel(r"$J_R$")
        axs.set_ylabel(r"$\eta$")

    def noise_plot(self,  axs, color, label,system=None, targets = None,filename = None):
        if self.verbose:
            print("Making noise plot for ", label)

        JR_arr, noise_arr, C_arr = self.get_noise_data(system, targets, filename)
        axs.plot(JR_arr, noise_arr, label = label, color = color)
        axs.set_title("Noise vs cooling power")
        axs.set_xlabel(r"$J_R$")
        axs.set_ylabel(r"$S$")

    def product_plot(self,  axs, color, label,system=None, targets = None,filename = None):
        if self.verbose:
            print("Making product plot for ", label)
        JR_arr, eff_arr, C_arr, err_arr = self.get_product_data(system, targets, filename)
        keep_index = np.argwhere(np.abs(err_arr) < 1e-4)
        JR_arr = JR_arr[keep_index]
        eff_arr = eff_arr[keep_index]
        #s_arr = s_arr[keep_index]
        axs.plot(JR_arr, eff_arr, label=label, color = color)
        axs.set_title("Product vs cooling power")
        axs.set_xlabel(r"$J_R$")
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
        C_list = []
        targets = np.linspace(0,jmax, self.n_targets)
        err_list = []
        [_,_,C_max], err = system.optimize_for_product(0.95*jmax)
        C_list = np.geomspace(C_min, C_max, self.n_targets)
        k = 0
        for C in C_list:
            if self.verbose:
                print("On target nr ", k, "with C = ", C)
            err = system.set_transmission_product_opt(C)
            JR_list.append(system._current_integral(system.coeff_con))
            product_list.append(system._current_integral(system.coeff_avg)*system.noise_cont(system.coeff_noise))   
            err_list.append(err)
            if self.verbose:
                print("Error: ", err)
            k += 1
        # for target in targets:
        #     if self.verbose:
        #         print("On target ", target)

        #     thetas, err = system.optimize_for_product(target)
        #     theta_list.append(thetas)
        #     JR_list.append(system._current_integral(system.coeff_con))
        #     product_list.append(system._current_integral(system.coeff_avg)*system.noise_cont(system.coeff_noise))
        #     err_list.append(err)
        JR_arr = np.array(JR_list)
        product_arr = np.array(product_list)
        C_arr = np.array(theta_list)
        eff_arr = product_arr/JR_arr
        err_arr = np.array(err_list)
        return JR_arr, eff_arr, C_arr, err_arr

    def get_eff_data(self,system=None, targets = None,filename = None):
        if filename:
            file = np.load(filename)
            JR_arr = file["JR_arr"]
            eff_arr = file["eff_arr"]
            C_arr = file["C_arr"]
        elif system and targets:
            JR_arr, eff_arr, C_arr = self.produce_eff_data(system, targets)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, eff_arr, C_arr

    def get_noise_data(self, system= None, targets = None, filename = None):
        if filename:
            file = np.load(filename)
            JR_arr = file["JR_arr"]
            noise_arr = file["noise_arr"]
            C_arr = file["C_arr"]
        elif system and targets:
            JR_arr, noise_arr, C_arr = self.produce_eff_data(system, targets)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, noise_arr, C_arr

    def get_product_data(self, system = None, targets = None, filename = None):
        if filename:
            file = np.load(filename)
            JR_arr = file["JR_arr"]
            eff_arr = file["eff_arr"]
            C_arr = file["C_arr"]
            err_arr = file["err_arr"]
        elif system and targets:
            JR_arr, eff_arr, C_arr, err_arr = self.produce_eff_data(system, targets)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, eff_arr, C_arr, err_arr

    def save_eff(self, system,targets, filename):
        JR_arr, eff_arr, C_arr = self.produce_eff_data(system, targets)
        np.savez(filename, JR_arr = JR_arr, eff_arr = eff_arr, C_arr = C_arr)
    
    def save_noise(self, system, targets, filename):
        JR_arr, noise_arr, C_arr = self.produce_noise_data(system, targets)
        np.savez(filename, JR_arr = JR_arr, noise_arr = noise_arr, C_arr = C_arr)

    def save_product(self, system, targets, filename):
        JR_arr, eff_arr, C_arr, err_arr = self.produce_product_data(system, targets)
        np.savez(filename, JR_arr = JR_arr, eff_arr = eff_arr, C_arr = C_arr, err_arr = err_arr)


class SecondaryPlot:
    def __init__(self, system_th:two_terminals, system_nth:two_terminals, s_min, s_max, secondary_prop, n_points = 20, verbose = False):
        self.system_th = system_th
        self.system_nth = system_nth
        self.verbose = verbose
        self.secondary_prop = secondary_prop
        self.s_min = s_min
        self.s_max = s_max
        self.s_arr = np.linspace(s_min, s_max, n_points)

    def updater(self, s, system:two_terminals):
        if self.secondary_prop == "muR":
            system.muR = s
            system.set_fermi_dist_right()
            system.adjust_limits(factor = 0)
        elif self.secondary_prop == "TR":
            system.TR = s
            system.set_fermi_dist_right()
            system.adjust_limits(factor = 0)
        else:
            pass
    def make_figure(self, make_eff = True, make_noise = True, make_product = True,
                    filenames = [None]*6, max_filenames = [None, None]):
        #make_list = [make_eff, make_noise, make_product]
        fig, axs = plt.subplots(3, 3, figsize = (10,10))
        JR_arr_th, avg_arr_th, noise_arr_th, prod_arr_th, s_arr_th = self.get_max_data(max_filenames[0])
        JR_arr_nth, avg_arr_nth, noise_arr_nth, prod_arr_nth, s_arr_nth = self.get_max_data(max_filenames[1])
        eff_arr_th = JR_arr_th/avg_arr_th
        eff_arr_nth = JR_arr_nth/avg_arr_nth

        if make_eff:
            self.eff_plot(axs[:,0], label = "thermal",system = self.system_th,  filename = filenames[0])
            self.eff_plot(system = self.system_nth, axs = axs[:,0], label = "nonthermal", filename=filenames[1])
            axs[0,0].plot(JR_arr_nth, eff_arr_nth, label = "Max J_R nth")
            axs[1,0].plot(s_arr_nth, eff_arr_nth, label = "Max J_R nth")
            axs[2,0].plot(s_arr_nth, JR_arr_nth, label = "Max J_R nth")
            axs[0,0].plot(JR_arr_th, eff_arr_th, label = "Max J_R th")
            axs[1,0].plot(s_arr_th, eff_arr_th, label = "Max J_R th")
            axs[2,0].plot(s_arr_th, JR_arr_th, label = "Max J_R th")

            axs[0,0].legend()
            axs[1,0].legend()
            axs[2,0].legend()
        if make_noise:
            self.noise_plot(system=self.system_th, axs=axs[:,1], label = "thermal", filename=filenames[2])
            self.noise_plot(system=self.system_nth, axs=axs[:,1], label = "nonthermal", filename=filenames[3])
            axs[0,1].plot(JR_arr_th, noise_arr_th, label = "Max J_R_th")
            axs[1,1].plot(s_arr_th, noise_arr_th, label = "Max J_R_th")
            axs[2,1].plot(s_arr_th, JR_arr_th, label = "Max J_R_th")
            axs[0,1].plot(JR_arr_nth, noise_arr_nth, label = "Max J_R_nth")
            axs[1,1].plot(s_arr_nth, noise_arr_nth, label = "Max J_R_nth")
            axs[2,1].plot(s_arr_nth, JR_arr_nth, label = "Max J_R_nth")

            axs[0,1].legend()
            axs[1,1].legend()
            axs[2,1].legend()
            #axs[:,1].legend()
        if make_product:
            self.product_plot(system=self.system_th, axs=axs[:,2], label = "thermal", filename=filenames[4])
            self.product_plot(system=self.system_nth, axs=axs[:,2], label = "nonthermal", filename=filenames[5])
            
            axs[0,2].plot(JR_arr_nth, prod_arr_nth, label = "Max J_R_nth")
            axs[1,2].plot(s_arr_nth, prod_arr_nth, label = "Max J_R_nth")
            axs[2,2].plot(s_arr_nth, JR_arr_nth, label = "Max J_R_nth")
            axs[0,2].plot(JR_arr_th, prod_arr_th, label = "Max J_R_th")
            axs[1,2].plot(s_arr_th, prod_arr_th, label = "Max J_R_th")
            axs[2,2].plot(s_arr_th, JR_arr_th, label = "Max J_R_th")

            axs[0,2].legend()
            axs[1,2].legend()
            axs[2,2].legend()
            #axs[:,2].legend()
        
        
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

    def make_example_figure(self):
        pass

    def eff_plot(self, axs, label,system=None,filename = None):
        if self.verbose:
            print("Making efficiency plot for ", label)
        JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr = self.get_eff_data(system, filename)
        eff_arr = JR_arr/avg_arr
        keep_index = np.argwhere(np.abs(err_arr) < 0.01)
        JR_arr = JR_arr[keep_index]
        eff_arr = eff_arr[keep_index]
        s_arr = s_arr[keep_index]

        print(len(s_arr))
        axs[0].scatter(JR_arr, eff_arr, label=label)
        axs[0].set_title("Efficiency vs cooling power")
        axs[0].set_xlabel("J_R")
        axs[0].set_ylabel(r"$\eta$")
        
        axs[1].scatter(s_arr, eff_arr, label=label)
        axs[1].set_title("Efficiency vs cooling power")
        axs[1].set_xlabel(self.secondary_prop)
        axs[1].set_ylabel(r"$\eta$")

        axs[2].scatter(s_arr, JR_arr, label=label)
        axs[2].set_title("Efficiency vs cooling power")
        axs[2].set_xlabel(self.secondary_prop)
        axs[2].set_ylabel(r"$J_R$")

    def noise_plot(self,  axs, label,system=None,filename = None):
        if self.verbose:
            print("Making noise plot for ", label)

        JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr = self.get_noise_data(system, filename)
        keep_index = np.argwhere(np.abs(err_arr) < 0.01)
        JR_arr = JR_arr[keep_index]
        noise_arr = noise_arr[keep_index]
        s_arr = s_arr[keep_index]
        axs[0].scatter(JR_arr, noise_arr, label=label)
        axs[0].set_title("Efficiency vs cooling power")
        axs[0].set_xlabel("J_R")
        axs[0].set_ylabel(r"$\eta$")
        axs[1].scatter(s_arr, noise_arr, label=label)
        axs[1].set_title("Efficiency vs cooling power")
        axs[1].set_xlabel(self.secondary_prop)
        axs[1].set_ylabel(r"$\eta$")
        axs[2].scatter(s_arr, JR_arr, label=label)
        axs[2].set_title("Efficiency vs cooling power")
        axs[2].set_xlabel(self.secondary_prop)
        axs[2].set_ylabel(r"$J_R$")
    def product_plot(self,  axs, label,system=None,filename = None):
        if self.verbose:
            print("Making product plot for ", label)
        JR_arr, avg_arr, noise_arr, C_arr, s_arr , err_arr= self.get_product_data(system, filename)
        keep_index = np.argwhere(np.abs(err_arr) < 0.01)
        #eff_arr = eff_arr*JR_arr
        JR_arr = JR_arr[keep_index]
        # eff_arr = eff_arr[keep_index]
        s_arr = s_arr[keep_index]
        C_arr = C_arr[keep_index]
        prod_arr = avg_arr*noise_arr#C_arr[:,0]*C_arr[:,1]
        prod_arr = prod_arr[keep_index]
        axs[0].scatter(JR_arr, prod_arr, label=label)
        axs[0].set_title("Efficiency vs cooling power")
        axs[0].set_xlabel("J_R")
        axs[0].set_ylabel(r"$\eta$")
        axs[1].scatter(s_arr, prod_arr, label=label)
        axs[1].set_title("Efficiency vs cooling power")
        axs[1].set_xlabel(self.secondary_prop)
        axs[1].set_ylabel(r"$\eta$")
        axs[2].scatter(s_arr, JR_arr, label=label)
        axs[2].set_title("Efficiency vs cooling power")
        axs[2].set_xlabel(self.secondary_prop)
        axs[2].set_ylabel(r"$J_R$")

    def produce_data_wrapper(self, system:two_terminals, opt_func, set_func):
        JR_list = []
        avg_list = []
        C_list = []
        err_list = []
        noise_list = []
        for s in self.s_arr:
            self.updater(s, system)
            C, err = opt_func(10, secondary_prop=self.secondary_prop)
            set_func(C)
            C_list.append(C)
            JR_list.append(system._current_integral(system.coeff_con))
            avg_list.append(system._current_integral(system.coeff_avg))
            noise_list.append(system._current_integral(system.coeff_noise))
            err_list.append(err)

        JR_arr = np.array(JR_list)
        avg_arr = np.array(avg_list)
        C_arr = np.array(C_list)
        err_arr = np.array(err_list)
        noise_arr = np.array(noise_list)
        return JR_arr, avg_arr, noise_arr, C_arr, err_arr


    def produce_eff_data(self, system:two_terminals):
        if self.verbose:
            print("Producing eff data")

        return self.produce_data_wrapper(system, system.optimize_for_best_avg, lambda C: system.set_transmission_avg_opt(C, system.coeff_con, system.coeff_avg))


        # JR_list = []
        # avg_list = []
        # C_list = []
        # err_list = []
        # for s in self.s_arr:
        #     self.updater(s, system)
        #     C, err = system.optimize_for_best_avg(10, secondary_prop=self.secondary_prop)
        #     system.set_transmission_avg_opt(C, system.coeff_con, system.coeff_avg)
        #     C_list.append(C)
        #     JR_list.append(system._current_integral(system.coeff_con))
        #     avg_list.append(system._current_integral(system.coeff_avg))
        #     Es = np.linspace(system.E_low, system.E_high, 10000)
        #     err_list.append(err)
        #     # plot_max(system)
        #     # plt.plot(Es, system.transf(Es), label = "transf")
        #     # plt.legend()
        #     # plt.show()
        # JR_arr = np.array(JR_list)
        # avg_arr = np.array(avg_list)
        # C_arr = np.array(C_list)
        # err_arr = np.array(err_list)
        # eff_arr = JR_arr/avg_arr
        # return JR_arr, eff_arr, C_arr, err_arr
    
    def produce_noise_data(self, system:two_terminals):
        if self.verbose:
            print("Producing noise data")
        return self.produce_data_wrapper(system, system.optimize_for_best_noise, system.set_transmission_noise_opt)

        # JR_list = []
        # noise_list = []
        # C_list = []
        # err_list = []
        # avg_list = []
        # for s in self.s_arr:
        #     if self.verbose:
        #         print("On target ", s)
        #     self.updater(s,system)
        #     C, err = system.optimize_for_best_noise(10, secondary_prop=self.secondary_prop)
        #     system.set_transmission_noise_opt(C)
        #     C_list.append(C)
        #     JR_list.append(system._current_integral(system.coeff_con))
        #     noise_list.append(system.noise_cont(system.coeff_noise))
        #     Es = np.linspace(system.E_low, system.E_high, 10000)
        #     err_list.append(err)
        #     avg_list.append(system._current_integral(system.coeff_avg))
        #     # plot_max(system)
        #     # plt.plot(Es, system.transf(Es), label = "transf")
        #     # plt.legend()
        #     # plt.show()
        # avg_arr = np.array(avg_list)
        # JR_arr = np.array(JR_list)
        # noise_arr = np.array(noise_list)
        # C_arr = np.array(C_list)
        # err_arr = np.array(err_list)
        # return JR_arr, noise_arr, C_arr, err_arr

    def produce_product_data(self, system:two_terminals):
        if self.verbose:
            print("Producing product data")
        return self.produce_data_wrapper(system, system.optimize_for_best_product, system.set_ready_transmission_product)

        # JR_list = []
        # product_list = []
        # theta_list = []
        # err_list = []
        # for s in self.s_arr:
        #     self.updater(s, system)
        #     [calc_avg, calc_noise, C], err = system.optimize_for_best_product(10)
        #     system.transf = system._transmission_product([calc_avg, calc_noise, C])
        #     prod = system._current_integral(system.coeff_avg)*system.noise_cont(system.coeff_noise)
        #     if self.verbose:
        #         print("Product: ", prod)
        #     JR_list.append(system._current_integral(system.coeff_con))
        #     theta_list.append([calc_avg, calc_noise, C])
        #     product_list.append(prod)
        #     err_list.append(err)
        # JR_arr = np.array(JR_list)
        # product_arr = np.array(product_list)
        # C_arr = np.array(theta_list)
        # err_arr = np.array(err_list)
        # eff_arr = product_arr/JR_arr
        # return JR_arr, eff_arr, C_arr, err_arr

    def produce_max_data(self, system:two_terminals):
        JR_arr = []
        eff_arr = []
        noise_arr = []
        avg_arr = []
        prod_arr = []
        for s in self.s_arr:
            self.updater(s, system)
            jmax, transf, _ = system.constrained_current_max()
            JR_arr.append(jmax)
            system.transf = transf
            noise = system.noise_cont(system.coeff_noise)
            avg = system._current_integral(system.coeff_avg)
            avg_arr.append(avg)
            noise_arr.append(noise)
            prod_arr.append(noise*avg)
        prod_arr = np.array(prod_arr)
        noise_arr = np.array(noise_arr)
        eff_arr = np.array(eff_arr)
        JR_arr = np.array(JR_arr)
        return JR_arr, avg_arr, noise_arr, prod_arr
    def get_eff_data(self,system=None,filename = None):
        if filename:
            file = np.load(filename)
            JR_arr = file["JR_arr"]
            avg_arr = file["avg_arr"]
            noise_arr = file["noise_arr"]
            C_arr = file["C_arr"]
            s_arr = file["s_arr"]
            err_arr = file["err_arr"]
        elif system:
            s_arr = self.s_arr
            JR_arr, avg_arr, C_arr, err_arr = self.produce_eff_data(system)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr

    def get_noise_data(self, system= None, filename = None):
        if filename:
            file = np.load(filename)
            JR_arr = file["JR_arr"]
            noise_arr = file["noise_arr"]
            avg_arr = file["avg_arr"]
            C_arr = file["C_arr"]
            s_arr = file["s_arr"]
            err_arr = file["err_arr"]
        elif system:
            s_arr = self.s_arr
            JR_arr, noise_arr, C_arr, err_arr = self.produce_eff_data(system)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr

    def get_product_data(self, system = None, filename = None):
        if filename:
            file = np.load(filename)
            JR_arr = file["JR_arr"]
            avg_arr = file["avg_arr"]
            C_arr = file["C_arr"]
            noise_arr = file["noise_arr"]
            s_arr = file["s_arr"]
            err_arr = file["err_arr"]
        elif system:
            s_arr = self.s_arr
            JR_arr, avg_arr, C_arr, err_arr = self.produce_eff_data(system)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr

    def fix_product_data(self, filename, system:two_terminals):
        file = np.load(filename)


        #JR_arr = file["JR_arr"]
        # eff_arr = file["eff_arr"]
        C_arr = file["C_arr"]
        JR_list = []
        s_arr = file["s_arr"]

        for i,s in enumerate(s_arr):
            self.updater(s, system)
            system.set_ready_transmission_product(C_arr[i,:])
            JR = system._current_integral(system.coeff_con)
            JR_list.append(JR)
        #noise_arr = file["noise_arr"]
        JR_arr = np.array(JR_list)
        err_arr = file["err_arr"]
        avg_arr = C_arr[:,0]
        noise_arr = C_arr[:,1]
        np.savez(filename, JR_arr = JR_arr, avg_arr = avg_arr, C_arr = C_arr, s_arr = s_arr, err_arr = err_arr, noise_arr = noise_arr)

    def get_max_data(self, filename):
        file = np.load(filename)
        JR_arr = file["JR_arr"]
        avg_arr = file["avg_arr"]
        noise_arr = file["noise_arr"]
        prod_arr = file["prod_arr"]
        return JR_arr, avg_arr, noise_arr,prod_arr,self.s_arr

    def save_eff(self, system, filename):
        JR_arr, avg_arr, noise_arr, C_arr, err_arr = self.produce_eff_data(system)
        np.savez(filename, JR_arr = JR_arr, avg_arr = avg_arr, C_arr = C_arr, s_arr = self.s_arr, err_arr = err_arr, noise_arr = noise_arr)
    
    def save_noise(self, system, filename):
        JR_arr, avg_arr, noise_arr, C_arr, err_arr = self.produce_noise_data(system)
        np.savez(filename, JR_arr = JR_arr, noise_arr = noise_arr, C_arr = C_arr, s_arr = self.s_arr, err_arr = err_arr, avg_arr = avg_arr)

    def save_product(self, system, filename):
        JR_arr, avg_arr, noise_arr, C_arr, err_arr = self.produce_product_data(system)
        np.savez(filename, JR_arr = JR_arr, avg_arr = avg_arr, C_arr = C_arr, s_arr = self.s_arr, err_arr = err_arr, noise_arr = noise_arr)

    def save_max(self, system, filename):
        JR_arr, avg_arr, noise_arr, prod_arr = self.produce_max_data(system)
        np.savez(filename, JR_arr = JR_arr, avg_arr = avg_arr, noise_arr = noise_arr, prod_arr = prod_arr, s_arr = self.s_arr)

    
if __name__ == "__main__":
    midT = 1
    deltaT = 0.8
    deltamu = 0
    muR = 1.2
    TR = midT-deltaT
    muL = 0#muR + deltamu
    TL = midT

    E_low = -5
    E_high = 5
    Es = np.linspace(E_low, E_high,1000)

    save_data = False
    load_params = True
    dist_type = "lorentz_peak"

    if load_params:
        th_dist_params = np.load("data/th_params_"+dist_type+".npz")['arr_0']
        muR = th_dist_params[0]
        TR = th_dist_params[1]
        nth_dist_params = np.load("data/nth_params_"+dist_type+".npz")['arr_0']
    else:
        th_dist_params = np.array([muR, TR])
        nth_dist_params = np.array([muL, TL, 0.1 ,0.4, 1])
        #nth_dist_params = np.array([muL, TL, 1, 0.5])
    occupf_L_nth = thermal_with_lorentz(*nth_dist_params)
    #occupf_L_nth = two_thermals(*nth_dist_params)
    #dist_params = np.array([muL,TL,muR,TR])


    left_virtual = two_terminals(-20, 20, occupf_L = occupf_L_nth, muL=muL, muR=1.2, TL=TL, TR=TR)

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


    fig_type = ".svg"
    secondary = True
    if secondary:
        thermal_left.debug = False
        nonthermal_left.debug = False
        secondary_prop = "muR"
        filenames = ["data/th_"+dist_type+"_eff_"+secondary_prop + ".npz","data/nth_"+dist_type+"_eff_"+secondary_prop + ".npz","data/th_"+dist_type+"_noise_"+secondary_prop + ".npz",
                    "data/nth_"+dist_type+"_noise_"+secondary_prop + ".npz","data/th_"+dist_type+"_product_"+secondary_prop + ".npz","data/nth_"+dist_type+"_product_"+secondary_prop + ".npz"]
        max_filenames = ["data/th_max_"+dist_type+secondary_prop+".npz","data/nth_max_"+dist_type+secondary_prop+".npz"]
        secondaryPlot = SecondaryPlot(thermal_left, nonthermal_left, 1, 5, secondary_prop, n_points=50, verbose=True)
        if save_data:
            if not load_params:            
                np.savez("data/th_params_"+dist_type, th_dist_params)
                np.savez("data/nth_params_"+dist_type, nth_dist_params)
            # secondaryPlot.save_eff(thermal_left, filenames[0])
            # secondaryPlot.save_eff(nonthermal_left, filenames[1])
            # secondaryPlot.save_noise(thermal_left, filenames[2])
            # secondaryPlot.save_noise(nonthermal_left, filenames[3])            


            # secondaryPlot.save_product(thermal_left, filenames[4])
            # secondaryPlot.save_product(nonthermal_left, filenames[5])
            secondaryPlot.save_max(thermal_left, max_filenames[0])
            secondaryPlot.save_max(nonthermal_left, max_filenames[1])

            # secondaryPlot.fix_product_data(filenames[4], thermal_left)
            # secondaryPlot.fix_product_data(filenames[5], nonthermal_left)

        # else:
        #     np.load()

        fig = secondaryPlot.make_figure(make_eff=True, make_noise=True, make_product=True, filenames=filenames, max_filenames=max_filenames)
        #plt.suptitle("Optimized plots for " + dist_type + r", metric $\dot S_R/\dot S_L$")
        plt.suptitle("Optimized plots for " + dist_type + r", metric $\dot S_R/\dot S_L$")
        plt.tight_layout()
        plt.savefig("figs/"+dist_type+"_"+secondary_prop+".png")

        fig = plt.figure()
        JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr =secondaryPlot.get_product_data(filename=filenames[5])
        print(JR_arr)
        # plt.plot(avg_arr*noise_arr)
        # JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr =secondaryPlot.get_noise_data(filename=filenames[3])
        # plt.plot(avg_arr*noise_arr)
        # JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr =secondaryPlot.get_eff_data(filename=filenames[1])
        # plt.plot(avg_arr*noise_arr)
        
        # nonthermal_left.TR = s_arr[-2]
        # nonthermal_left.set_fermi_dist_right()
        # nonthermal_left.adjust_limits()
        # nonthermal_left.set_transmission_product_opt(C_arr[-2,2])
        # nonthermal_left.transf = nonthermal_left._transmission_product(C_arr[-2])
        # print(err_arr)
        # plt.plot(Es, nonthermal_left.transf(Es))
        # print(nonthermal_left.noise_cont(nonthermal_left.coeff_noise)*nonthermal_left._current_integral(nonthermal_left.coeff_avg))#/nonthermal_left._current_integral(nonthermal_left.coeff_con))
        # _,max_transf,_ = nonthermal_left.constrained_current_max()
        # nonthermal_left.transf = max_transf
        # print(nonthermal_left.noise_cont(nonthermal_left.coeff_noise)*nonthermal_left._current_integral(nonthermal_left.coeff_avg))#/nonthermal_left._current_integral(nonthermal_left.coeff_con))
        # print(C_arr[-2,0]*C_arr[-2,1])
        # print(s_arr[-2])
        # print(prod_arr*JR_arr)
        JR_arr,_,_,prod_max, _= secondaryPlot.get_max_data(max_filenames[1])
        print(JR_arr)
        # print(prod_max)
        # plt.plot(Es, max_transf(Es))
          
    else:   
        thermal_left.debug = False
        nonthermal_left.debug = False             
        filenames = ["data/th_"+dist_type+"_eff.npz","data/nth_"+dist_type+"_eff.npz","data/th_"+dist_type+"_noise.npz",
                    "data/nth_"+dist_type+"_noise.npz","data/th_"+dist_type+"_product.npz","data/nth_"+dist_type+"_product.npz"]
        powerPlot = PowerPlot(thermal_left, nonthermal_left, True, n_targets=50)
        if save_data:
            
            np.savez("data/th_params_"+dist_type, th_dist_params)
            np.savez("data/nth_params_"+dist_type, nth_dist_params)
            # powerPlot.save_eff(thermal_left, powerPlot.targets_th, filenames[0])
            # powerPlot.save_eff(nonthermal_left, powerPlot.targets_nth, filenames[1])
            # powerPlot.save_noise(thermal_left, powerPlot.targets_th, filenames[2])
            # powerPlot.save_noise(nonthermal_left, powerPlot.targets_nth, filenames[3])            
            powerPlot.save_product(thermal_left, powerPlot.targets_th, filenames[4])
            powerPlot.save_product(nonthermal_left, powerPlot.targets_nth, filenames[5])
        # else:
        #     np.load()

        fig = powerPlot.make_figure(make_eff=True, make_noise=True, make_product=True, filenames=filenames)
        #plt.suptitle("Optimized plots for " + dist_type + r", metric $\dot S_R/\dot S_L$")
        plt.suptitle("Optimized quantifiers over cooling power spectrum for thermal with Lorentzian peak and dip")
        plt.tight_layout()
        plt.savefig("figs/"+dist_type+fig_type)

        # fig = plt.figure()
        JR_arr, noise_arr, C_arr, err_arr =powerPlot.get_product_data(filename=filenames[5])
        
        # plt.plot(powerPlot.targets_nth-JR_arr)
        # plt.show()
        fig = powerPlot.make_dist_figure(-3,3)
        plt.savefig("figs/"+dist_type+"_dist"+fig_type)