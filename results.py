import numpy as np
import matplotlib.pyplot as plt
import copy
import utilities as ut

from thermo_funcs import two_terminals

from scipy.optimize import fsolve, minimize
from scipy import integrate

### MATPLOT SETUP ###

#cold = "#007EF5"
cold = "#009BFF"
#hot = "#FF4400"
hot = "#FF4C00"
nonthermal = "#A400FF"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'NewComputerModernSans08'
plt.rcParams["font.size"] = 10

plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["figure.titlesize"] = "medium"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['pdf.fonttype'] = 42


pt = 1/72
col = 246*pt


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

        self.verbose = verbose
    #def define_figure(self):
    def make_figure(self, make_eff = True, make_noise = True, make_product = True,
                    filenames = [None]*6):
        #make_list = [make_eff, make_noise, make_product]
        fig, axs = plt.subplots(1, 3, figsize = (246*2*pt,200*pt), layout = "constrained")
        #fig.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
        if make_eff:
            self.eff_plot(axs[0], color = hot, label = "thermal",system = self.system_th,  filename = filenames[0])
            self.eff_plot(system = self.system_nth, color = nonthermal, axs = axs[0], label = "nonthermal", filename=filenames[1])
            
            #axs[0].legend()
        if make_noise:
            self.noise_plot(system=self.system_th, color = hot, axs=axs[1], label = "thermal", filename=filenames[2])
            self.noise_plot(system=self.system_nth, color = nonthermal, axs=axs[1], label = "nonthermal", filename=filenames[3])
            #axs[1].legend()
        if make_product:
            self.product_plot(system=self.system_th, color = hot, axs=axs[2], label = "thermal", filename=filenames[4])
            self.product_plot(system=self.system_nth, color = nonthermal, axs=axs[2], label = "nonthermal", filename=filenames[5])
            #axs[2].legend()
        lines, labels = axs[1].get_legend_handles_labels()
        fig.legend(lines, labels, loc= "outside lower center", ncols = 2)#, bbox_to_anchor = (0.55,-0.1))
        fig.text(1,1,"x")
        #plt.subplots_adjust(left=0.1)
        #plt.tight_layout()
        return fig

    def make_dist_figure(self, E_min = None, E_max = None):
        if E_min == None or E_max == None:
            Es = np.linspace(self.system_nth.E_low, self.system_nth.E_high,1000)
        else:
            Es = np.linspace(E_min, E_max, 1000)
        fig = plt.figure(figsize= (col, 246*pt))

        
        plt.plot(Es - self.system_th.muR, self.system_nth.occupf_L(Es), label = "Nonthermal", color = nonthermal, zorder = 3)
        plt.plot(Es- self.system_th.muR, self.system_th.occupf_L(Es), label = "Thermal probe", color = hot, zorder = 2)
        plt.plot(Es- self.system_th.muR, self.system_th.occupf_R(Es), label = "Cold thermal", color = cold, zorder = 1)
        plt.legend()
        plt.xlabel(r"$\varepsilon$ [$k_B T_0$]")
        plt.ylabel("Occupation probability")
        plt.title("Thermal distribution with \n Lorentzian peak and dip")
        plt.tight_layout()
        return fig

    def make_example_figure(self, example_file, make_eff = False, make_noise = False, make_product = False):
        jmax, transf_max, _ = self.system_nth.constrained_current_max()
        self.system_nth.adjust_limits(0.5)
        Es = np.linspace(self.system_nth.E_low, self.system_nth.E_high,10000)
        
        fig, axs = plt.subplots(2,1,figsize= (col, 350*pt), layout = "constrained")
        axs[0].plot(Es- self.system_th.muR, self.system_nth.occupf_L(Es), label = "Nonthermal", color = nonthermal, zorder = 3)
        #axs[0].plot(Es, self.system_th.occupf_L(Es), label = "Thermal probe", color = hot, zorder = 2)
        axs[0].plot(Es- self.system_th.muR, self.system_th.occupf_R(Es), label = "Cold thermal", color = cold, zorder = 2)
        axs[0].vlines(self.system_nth.muR- self.system_th.muR, 0,1, colors = cold, alpha = 0.7, linestyles = "dashed")
        axs[0].annotate(r"$\mu_R$", (self.system_nth.muR- self.system_th.muR, 0.7), textcoords = "offset points", xytext = (2,0), color = cold, alpha =0.7)
        axs[0].annotate("No cooling", (0.1- self.system_th.muR, 0.7), color = "grey")
        axs[0].fill_between(Es- self.system_th.muR, 1, where = transf_max(Es) == 0, facecolor = "lightgrey", zorder = 1)

        axs[0].legend(loc = "lower left")
        axs[0].set_xlabel(r"$\varepsilon$ [$k_B T_0$]")
        axs[0].set_ylabel("Occupation probability")
        
        file = np.load(example_file)
        
        
        prod_vector = file["prod_vector"]

        if make_eff:
            C_avg = file["C_avg"]
            transf_avg = self.system_nth._transmission_avg(float(C_avg), self.system_nth.coeff_con, self.system_nth.coeff_avg)
            #axs[1].plot(Es, transf_avg(Es), color = "#c870ff", label = "Best eff.")
            axs[1].plot(Es- self.system_th.muR, transf_avg(Es), color = "#DB70FF", label = "Best eff.")

        if make_noise:
            C_noise = file["C_noise"]
            transf_noise = self.system_nth._transmission_noise(float(C_noise))
            #axs[1].plot(Es, transf_noise(Es), '--',color = "#7100b8", label = "Best precis.")
            axs[1].plot(Es- self.system_th.muR, transf_noise(Es), '--',color = "#58008F", label = "Best precis.")

        #axs[1].set_title("")
        #axs[1].plot(Es, transf_max(Es), '--',color = "#c870ff", label = "Max cooling")
        axs[1].set_xlabel(r"$\varepsilon$ [$k_B T_0$]")
        axs[1].set_ylabel("Transmission probability")
        
        axs[1].legend()
        

        fig.suptitle(r"Optimized transmission functions at 0.4 $I^{Q,R}_{\mathrm{max}}$")
        #fig.suptitle("Transmission for maximized cooling with\n arbitrary nonthermal resource")
        return fig 

    def make_crossing_figure(self, filename, system:two_terminals, type = "eff"):
        occupf_L = system.occupf_L
        occupf_R = system.occupf_R
        C_pick = 250
        if type == "eff":
            JR_arr, data_arr, C_arr = self.get_eff_data(filename=filename)
            y = system.coeff_avg
            x = system.coeff_con
            factor_two = lambda E: y(E)-C_arr.reshape(-1,1)*x(E)
            factor_one = lambda E: occupf_L(E)-occupf_R(E)
            

            Es = np.linspace(system.E_low, system.E_high, 10000)
            fig, axs = plt.subplots(1, 3, figsize = (246*2*pt, 150*pt), layout = "constrained")
            axs[0].plot(Es - system.muR, occupf_L(Es), label = "Left", color = nonthermal)
            axs[0].plot(Es - system.muR, occupf_R(Es), label = "Right", color = cold)
            #axs[0].plot(Es-system.muR, system._transmission_avg(C_arr[10], x, y)(Es), label = "transf")
            axs[0].fill_between(Es- system.muR, 1, where = factor_one(Es) > 0, facecolor = "lightgrey", zorder = 1)
            axs[0].set_xlabel("Energy")
            axs[0].set_xlabel(r"$\varepsilon - \mu$ [$k_B T_0$]")
            axs[0].set_ylabel("Factor value")
            #axs[0].legend()
            
            #axs[0].grid()
            axs[0].set_title("Occupation functions")

            axs[1].plot(Es-system.muR, -y(Es), label = "Left", color = nonthermal)
            axs[1].plot(Es-system.muR, -C_arr[C_pick]*x(Es), label = "Right", color = cold)

            #axs[1].plot(Es-system.muR, system._transmission_avg(C_arr[10], x, y)(Es), label = "transf")
            axs[1].fill_between(Es- system.muR, 5, -5, where = factor_two(Es)[C_pick,:] < 0, facecolor = "lightgrey", zorder = 1)
            axs[1].set_xlabel("Energy")
            axs[1].set_xlabel(r"$\varepsilon - \mu$ [$k_B T_0$]")
            axs[1].set_ylabel("Factor value")
            #axs[1].legend()
            axs[1].set_title("Coefficients")
            #axs[1].grid()
            axs[2].fill_between(Es- system.muR, 0, 1, where = factor_two(Es)[C_pick,:] < 0, facecolor = "gray", zorder = 1, alpha = 0.5)
            axs[2].fill_between(Es- system.muR, 0, 1, where = factor_one(Es) > 0, facecolor = "gray", zorder = 1, alpha = 0.5)
            # fig = plt.figure(figsize=(246*pt, 200*pt), layout = "constrained")
            # plt.plot(Es, factor_one(Es)[10,:], label = "Factor one")
            # plt.plot(Es, factor_two(Es), label = "Factor two")
            # plt.xlabel("Energy")
            # plt.ylabel("Factor value")
            # plt.legend()
            return fig

        elif type == "noise":
            JR_arr, data_arr, C_arr = self.get_noise_data(filename)
        elif type == "product":
            JR_arr, data_arr, C_arr, err_arr = self.get_product_data(filename)
            
        else:
            raise TypeError("Unrecognized type for crossing figure")
        
    def make_all_crossing_figure(self, filenames, system:two_terminals):
        fig, axs = plt.subplots(1,3,figsize=(246*2*pt, 200*pt))
        JR_arr, data_arr, C_eff = self.get_eff_data(filename=filenames[0])
        JR_arr, data_arr, C_noise = self.get_noise_data(filename=filenames[1])
        JR_arr, data_arr, C_prod, err_arr = self.get_product_data(filename=filenames[2])
        C_pick = int(len(C_eff)/2)
        
        # Es = np.linspace(system.E_low, system.E_high, 10000)
        Es = np.linspace(-5, 5, 10000)
        cond_eff = system._avg_condition(C_eff[C_pick], system.coeff_con, system.coeff_avg)
        cond_noise = system._noise_condition(C_noise[C_pick])
        #print(C_prod[C_pick])
        # if type(C_prod[C_pick]) != np.array:
        #     system.set_transmission_product_opt(C_prod[C_pick])
        #     nois = system.noise_cont(system.coeff_noise)
        #     avg = system._current_integral(system.coeff_avg)
        #     cond_prod = system._product_condition([avg, nois, C_prod[C_pick]])
        # else:    
        # cond_prod = system._product_condition(C_prod[C_pick,:])
        keep_index = np.argwhere(np.abs(err_arr) < 1e-4)
        # print(JR_arr)
        # JR_arr = JR_arr[keep_index]
        # eff_arr = eff_arr[keep_index]
        # print(C_prod)
        #C_prod = C_prod[keep_index]
        # print(C_prod[0])
        # for i,C in enumerate(np.linspace(0,0.1,10)):
        #     #cond_prod = system._product_condition([0.04944221649221684, 0.011044649928926384,0.09090909090909091])
        cond_prod = system._product_condition(C_prod[C_pick].flatten())
            # cond_prod = system._product_condition([0.01,0.01,C])
        axs[2].plot(Es, cond_prod(Es))
        
        # cond_prod = system._product_condition([0.041313915835068674, 0.01578651849885714,0.09131313131313132])
        # axs[2].plot(Es, cond_prod(Es), label = C)
            #axs[0].plot(Es, np.heaviside(cond_eff(Es),0))
        axs[0].plot(Es, cond_eff(Es))
        axs[0].set_title("Eff cond")
        axs[0].grid()
        axs[1].plot(Es, cond_noise(Es))
        axs[1].set_title("Noise cond")
        axs[1].grid()
        
        axs[2].set_title("Prod cond")
        #axs[2].legend()
        axs[2].grid()
        return fig

    def make_char_eff_figure(self, filename, system:two_terminals):
        occupf_L = system.occupf_L
        occupf_R = system.occupf_R
        occupdiff = lambda E: occupf_L(E) - occupf_R(E)
        C_pick = 5
        # if type == "eff":
        JR_arr, data_arr, C_arr = self.get_eff_data(filename=filename)
        y = system.coeff_avg
        x = system.coeff_con
        factor_two = lambda E: y(E)-C_arr.reshape(-1,1)*x(E)
        factor_one = lambda E: occupf_L(E)-occupf_R(E)
        

        #Es = np.linspace(system.E_low, system.E_high, 10000)
        Es = np.linspace(-5, 5, 10000)
        fig, axs = plt.subplots(1, 3, figsize = (246*2*pt, 150*pt), layout = "constrained")
        axs[0].plot(Es - system.muR, x(Es)*occupdiff(Es), label = "con", color = nonthermal)
        axs[0].plot(Es - system.muR, y(Es)*occupdiff(Es), label = "avg", color = cold)
        #axs[0].plot(Es-system.muR, system._transmission_avg(C_arr[10], x, y)(Es), label = "transf")
        # axs[0].fill_between(Es- system.muR, 1, where = factor_one(Es) > 0, facecolor = "lightgrey", zorder = 1)
        axs[0].set_xlabel("Energy")
        axs[0].set_xlabel(r"$\varepsilon - \mu$ [$k_B T_0$]")
        axs[0].set_ylabel("Factor value")
        #axs[0].legend()
        
        axs[0].grid()
        axs[0].set_title("Occupation functions")

        #axs[1].plot(Es-system.muR, np.heaviside(x(Es)/y(Es)-1/C_arr[C_pick],0), label = "frac", color = nonthermal)
        axs[1].plot(Es-system.muR, x(Es)/y(Es)*np.heaviside(x(Es)*occupdiff(Es),0), label = "frac", color = cold)
        # axs[1].plot(Es-system.muR, x(Es)/y(Es)*np.heaviside(y(Es)*occupdiff(Es),0), label = "frac", color = nonthermal)
        # axs[1].plot(Es, x(Es), color = cold)
        # axs[1].plot(Es, y(Es), color = nonthermal)
        # axs[1].plot(Es-system.muR, C_arr[C_pick], label = "Right", color = cold)
        axs[1].hlines(1/C_arr[C_pick], Es[0], Es[-1])
        print(1/C_arr[C_pick])
        # axs[1].set_ylim([0,1])

        #axs[1].plot(Es-system.muR, system._transmission_avg(C_arr[10], x, y)(Es), label = "transf")
        #axs[1].fill_between(Es- system.muR, 5, -5, where = factor_two(Es)[C_pick,:] < 0, facecolor = "lightgrey", zorder = 1)
        axs[1].set_xlabel("Energy")
        axs[1].set_xlabel(r"$\varepsilon - \mu$ [$k_B T_0$]")
        axs[1].set_ylabel("Factor value")
        #axs[1].legend()
        axs[1].set_title("Coefficients")
        axs[1].grid()
        axs[2].fill_between(Es- system.muR, 0, 1, where = factor_two(Es)[C_pick,:] < 0, facecolor = "gray", zorder = 1, alpha = 0.5)
        axs[2].fill_between(Es- system.muR, 0, 1, where = factor_one(Es) > 0, facecolor = "gray", zorder = 1, alpha = 0.5)

    def eff_plot(self, axs, color, label,system=None,filename = None):
        if self.verbose:
            print("Making efficiency plot for ", label)
        JR_arr, eff_arr, C_arr = self.get_eff_data(system, filename)

        axs.plot(JR_arr, eff_arr, label=label, color = color)
        # print(JR_arr)
        # print(JR_arr[:-1]-JR_arr[1:])
        # print(np.argmax(np.abs(JR_arr[:-1]-JR_arr[1:])))
        # print(C_arr[[112,113,114,115]])
        # print(JR_arr[[112,113,114,115]])
        #print(np.argwhere(JR_arr == 2.55085179e-02))
        axs.set_title("Highest efficency")
        axs.set_xlabel(r"$J_R$")
        axs.set_ylabel(r"$\eta$ [$\dot S_R/\dot S_L$]")
        # axs.set_xlim([0, np.max(JR_arr)])

    def noise_plot(self,  axs, color, label,system=None,filename = None):
        if self.verbose:
            print("Making noise plot for ", label)

        JR_arr, noise_arr, C_arr = self.get_noise_data(system, filename)
        axs.plot(JR_arr, noise_arr, label = label, color = color)
        axs.set_title("Lowest noise")
        axs.set_xlabel(r"$J_R$")
        axs.set_ylabel(r"$S_{\dot S_R}$")
        # axs.set_xlim([0, np.max(JR_arr)])

    def product_plot(self,  axs, color, label,system=None,filename = None):
        if self.verbose:
            print("Making product plot for ", label)
        JR_arr, eff_arr, C_arr, err_arr = self.get_product_data(system, filename)
        keep_index = np.argwhere(np.abs(err_arr) < 1e-5)
        # print(JR_arr)
        JR_arr = JR_arr[keep_index]
        eff_arr = eff_arr[keep_index]
        # s_arr = s_arr[keep_index]
        axs.plot(JR_arr, eff_arr, label=label, color = color)
        axs.set_title("Lowest noise-eff. fraction")
        axs.set_xlabel(r"$J_R$")
        axs.set_ylabel(r"$S_{\dot S_R}/ \eta$")
        # axs.set_xlim([0, np.max(JR_arr)])

    def produce_eff_data(self, system:two_terminals):
        if self.verbose:
            print("Producing eff data")
        JR_list = []
        avg_list = []
        C_min = system.C_limit_avg(system.coeff_con, system.coeff_avg)
        #print(C_min)
        #C_min = 0
        # print(system.E_low, system.E_high)
        jmax,_,_ = system.constrained_current_max()
        C_max, err = system.optimize_for_avg(0.99*jmax, 10)
        # C_list = np.logspace(C_min, C_max, self.n_targets, base = 0.001)
        # C_min = 4.11777778
        # C_max = 4.11778078
        C_list = np.linspace(C_min, C_max, self.n_targets)
        k = 0

        #print(C_min, C_max, C_list)
        for C in C_list:
            # if self.verbose:
            #     print("On target ",k)
            system.set_transmission_avg_opt(C, system.coeff_con, system.coeff_avg)
            JR = system._current_integral(system.coeff_con, cond_in=system._avg_condition(C, system.coeff_con, system.coeff_avg))
            avg = system._current_integral(system.coeff_avg, cond_in=system._avg_condition(C, system.coeff_con, system.coeff_avg))
            if JR == 0.0 or avg == 0.0:
            
                temp_E_low = system.E_low
                temp_E_high = system.E_high
                temp_Es = np.linspace(system.E_low, system.E_high, 100000)
                if any(system.transf(temp_Es) == 1):
                    limits = temp_Es[np.argwhere(system.transf(temp_Es)[1:] - system.transf(temp_Es)[:-1] != 0).flatten()]
                    system.E_low = limits[0]*0.95
                    system.E_high = limits[-1]*1.05
                    JR_list.append(system._current_integral(system.coeff_con))
                    avg_list.append(system._current_integral(system.coeff_avg))
                    system.E_low = temp_E_low
                    system.E_high = temp_E_high
                else:
                    JR_list.append(JR)
                    avg_list.append(avg)                                       
            else:
                JR_list.append(JR)
                avg_list.append(avg)
            k += 1

        JR_arr = np.array(JR_list)
        avg_arr = np.array(avg_list)
        #print(avg_arr)
        C_arr = np.array(C_list)
        eff_arr = JR_arr/avg_arr
        # eff_arr = np.nan_to_num(eff_arr)
        # print(eff_arr)
        return JR_arr, eff_arr, C_arr
    
    def produce_noise_data(self, system:two_terminals):
        if self.verbose:
            print("Producing noise data")

        JR_list = []
        noise_list = []
        C_min = 0.005#system.C_limit_avg(system.coeff_con, system.coeff_con)
        jmax,_,_ = system.constrained_current_max()
        C_max, err = system.optimize_for_noise(0.99*jmax, 10)
        # C_list = np.logspace(C_min, C_max, self.n_targets, base = 0.001)
        C_list = np.linspace(C_min, C_max, self.n_targets)
        k = 0
        for C in C_list:
            system.set_transmission_noise_opt(C)
            if self.verbose and k % 20 == 0:
                print("On target ", k)
            k += 1
            nois = system.noise_cont(system.coeff_noise, cond_in = system._noise_condition(C))
            JR = system._current_integral(system.coeff_con, cond_in = system._noise_condition(C))
            if JR == 0.0 or nois == 0.0:

                temp_E_low = system.E_low
                temp_E_high = system.E_high
                temp_Es = np.linspace(system.E_low, system.E_high, 100000)
                if any(system.transf(temp_Es) == 1):
                    limits = temp_Es[np.argwhere(system.transf(temp_Es)[1:] - system.transf(temp_Es)[:-1] != 0).flatten()]
                    system.E_low = limits[0]*0.95
                    system.E_high = limits[-1]*1.05
                    noise_list.append(system.noise_cont(system.coeff_noise, cond_in = system._noise_condition(C)))
                    JR_list.append(system._current_integral(system.coeff_con))

                    system.E_low = temp_E_low
                    system.E_high = temp_E_high            
                else:
                    JR_list.append(JR)
                    noise_list.append(nois)

            else:
                JR_list.append(JR)
                noise_list.append(nois)

            
        JR_arr = np.array(JR_list)
        noise_arr = np.array(noise_list)
        C_arr = np.array(C_list)
        return JR_arr, noise_arr, C_arr

    def produce_product_data(self, system:two_terminals):
        if self.verbose:
            print("Producing product data")

        JR_list = []
        product_list = []
        theta_list = []
        C_min = 0.001#system.C_limit_avg(system.coeff_con, system.coeff_con)
        jmax,_,_ = system.constrained_current_max()
        C_list = []
        targets = np.linspace(0,jmax, self.n_targets)
        err_list = []

        [_,_,C_max], err = system.optimize_for_product(0.99*jmax)
        # C_max = 2.2135420629378855
        print("C_min, C_max ", C_min,C_max)
        # The function of JR to C is highly non-linear, and JR increases very quickly with C, so we sample more points at low Cs to try to even out the spectrum
        base = 0.1
        C_list = np.logspace(np.emath.logn(base, C_min), np.emath.logn(base,C_max), self.n_targets, base = base)

        start_avg = 0.01
        start_noise = 0.0001

        # start_avg = None
        # start_noise = None


        # C_max = 1
        # C_list = np.linspace(C_min, C_max, self.n_targets)
        k = 0
        for C in C_list:
            if self.verbose:
                print("On target nr ", k, "with C = ", C)
            avg, nois,err = system.set_transmission_product_opt(C, start_avg = start_avg, start_noise = start_noise)
            err_list.append(err)
            JR = system._current_integral(system.coeff_con, cond_in = system._product_condition([avg, nois, C]))
            # nois = system.noise_cont(system.coeff_noise, cond_in = system._noise_condition(C))
            # avg = system._current_integral(system.coeff_avg, cond_in = system._avg_condition(C))
            product = nois*avg
            # if JR == 0.0 or product == 0.0:
            #     temp_E_low = system.E_low
            #     temp_E_high = system.E_high
            #     temp_Es = np.linspace(system.E_low, system.E_high, 100000)
            #     if any(system.transf(temp_Es) == 1):
            #         limits = temp_Es[np.argwhere(system.transf(temp_Es)[1:] - system.transf(temp_Es)[:-1] != 0).flatten()]
            #         system.E_low = limits[0]*0.95
            #         system.E_high = limits[-1]*1.05
            #         nois = system.noise_cont(system.coeff_noise)
            #         avg = system._current_integral(system.coeff_avg)
            #         JR_list.append(system._current_integral(system.coeff_con))
            #         product_list.append(avg*nois)   
            #         theta_list.append([avg, nois, C])
            #         system.E_low = temp_E_low
            #         system.E_high = temp_E_high  
                

            #     else:
            #         JR_list.append(JR)
            #         product_list.append(product)
            #         theta_list.append([avg, nois, C])
            
            
            JR_list.append(JR)
            product_list.append(product)
            theta_list.append([avg, nois, C])
        
            if self.verbose:
                print("Error: ", err)
                print("JR: ", JR_list[k])
            k += 1

        JR_arr = np.array(JR_list)
        product_arr = np.array(product_list)
        C_arr = np.array(theta_list)
        eff_arr = product_arr/JR_arr
        # eff_arr = np.nan_to_num(eff_arr)
        err_arr = np.array(err_list)
        return JR_arr, eff_arr, C_arr, err_arr

    def get_eff_data(self,system=None,filename = None):
        if filename:
            file = np.load(filename)
            JR_arr = file["JR_arr"]
            eff_arr = file["eff_arr"]
            C_arr = file["C_arr"]
        elif system:
            JR_arr, eff_arr, C_arr = self.produce_eff_data(system)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, eff_arr, C_arr

    def get_noise_data(self, system= None, filename = None):
        if filename:
            file = np.load(filename)
            JR_arr = file["JR_arr"]
            noise_arr = file["noise_arr"]
            C_arr = file["C_arr"]
        elif system:
            JR_arr, noise_arr, C_arr = self.produce_eff_data(system)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, noise_arr, C_arr

    def get_product_data(self, system = None, filename = None):
        if filename:
            file = np.load(filename)
            JR_arr = file["JR_arr"]
            eff_arr = file["eff_arr"]
            C_arr = file["C_arr"]
            err_arr = file["err_arr"]
        elif system:
            JR_arr, eff_arr, C_arr, err_arr = self.produce_eff_data(system)    
        else:
            print("Nothing to get :(")
            return
        return JR_arr, eff_arr, C_arr, err_arr

    def save_eff(self, system, filename):
        JR_arr, eff_arr, C_arr = self.produce_eff_data(system)
        np.savez(filename, JR_arr = JR_arr, eff_arr = eff_arr, C_arr = C_arr)
    
    def save_noise(self, system, filename):
        JR_arr, noise_arr, C_arr = self.produce_noise_data(system)
        np.savez(filename, JR_arr = JR_arr, noise_arr = noise_arr, C_arr = C_arr)

    def save_product(self, system, filename):
        JR_arr, eff_arr, C_arr, err_arr = self.produce_product_data(system)
        np.savez(filename, JR_arr = JR_arr, eff_arr = eff_arr, C_arr = C_arr, err_arr = err_arr)

    def save_example(self, target_factor, filename):
        jmax, _,_, = self.system_nth.constrained_current_max()
        target = target_factor*jmax
        C_avg, avg_err = self.system_nth.optimize_for_avg(target,10)
        C_noise, noise_err = self.system_nth.optimize_for_noise(target, 10)
        [prod_avg, prod_nois, C_prod], prod_err = self.system_nth.optimize_for_product(target)
        if self.verbose:
            print("Average error: ", avg_err)
            print("Noise error: ", noise_err)
            print("Product error: ", prod_err)
        prod_vector = np.array([prod_avg, prod_nois, C_prod])
        np.savez(filename, C_avg = C_avg, C_noise = C_noise, prod_vector = prod_vector)
            
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
            system.adjust_limits(E_high_start=1.5*system.E_high)
        elif self.secondary_prop == "TR":
            system.TR = s
            system.set_fermi_dist_right()
            system.adjust_limits()
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
            self.eff_plot(axs[:,0], color = hot, label = "thermal",system = self.system_th,  filename = filenames[0])
            self.eff_plot(system = self.system_nth, color = nonthermal, axs = axs[:,0], label = "nonthermal", filename=filenames[1], style = "--")
            axs[0,0].plot(JR_arr_nth, eff_arr_nth, label = "Max J_R nth", color = nonthermal, alpha = 0.5)
            axs[1,0].plot(s_arr_nth, eff_arr_nth, label = "Max J_R nth", color = nonthermal, alpha = 0.5)
            axs[2,0].plot(s_arr_nth, JR_arr_nth, label = "Max J_R nth", color = nonthermal, alpha = 0.5)
            axs[0,0].plot(JR_arr_th, eff_arr_th, label = "Max J_R th", color = hot, alpha = 0.5)
            axs[1,0].plot(s_arr_th, eff_arr_th, label = "Max J_R th", color = hot, alpha = 0.5)
            axs[2,0].plot(s_arr_th, JR_arr_th, label = "Max J_R th", color = hot, alpha = 0.5)

            axs[0,0].legend()
            axs[1,0].legend()
            axs[2,0].legend()
        if make_noise:
            self.noise_plot(system=self.system_th, color = hot, axs=axs[:,1], label = "thermal", filename=filenames[2])
            self.noise_plot(system=self.system_nth, color = nonthermal,axs=axs[:,1], label = "nonthermal", filename=filenames[3])
            axs[0,1].plot(JR_arr_th, noise_arr_th, label = "Max J_R_th", color = hot, alpha = 0.5)
            axs[1,1].plot(s_arr_th, noise_arr_th, label = "Max J_R_th", color = hot, alpha = 0.5)
            axs[2,1].plot(s_arr_th, JR_arr_th, label = "Max J_R_th", color = hot, alpha = 0.5)
            axs[0,1].plot(JR_arr_nth, noise_arr_nth, label = "Max J_R_nth", color = nonthermal, alpha = 0.5)
            axs[1,1].plot(s_arr_nth, noise_arr_nth, label = "Max J_R_nth", color = nonthermal, alpha = 0.5)
            axs[2,1].plot(s_arr_nth, JR_arr_nth, label = "Max J_R_nth", color = nonthermal, alpha = 0.5)

            axs[0,1].legend()
            axs[1,1].legend()
            axs[2,1].legend()
            #axs[:,1].legend()
        if make_product:
            self.product_plot(system=self.system_th, axs=axs[:,2], color = hot, label = "thermal", filename=filenames[4])
            self.product_plot(system=self.system_nth, axs=axs[:,2], color = nonthermal, label = "nonthermal", filename=filenames[5])
            
            axs[0,2].plot(JR_arr_nth, prod_arr_nth, label = "Max J_R_nth", color = nonthermal, alpha = 0.5)
            axs[1,2].plot(s_arr_nth, prod_arr_nth, label = "Max J_R_nth", color = nonthermal, alpha = 0.5)
            axs[2,2].plot(s_arr_nth, JR_arr_nth, label = "Max J_R_nth", color = nonthermal, alpha = 0.5)
            axs[0,2].plot(JR_arr_th, prod_arr_th, label = "Max J_R_th", color = hot, alpha = 0.5)
            axs[1,2].plot(s_arr_th, prod_arr_th, label = "Max J_R_th", color = hot, alpha = 0.5)
            axs[2,2].plot(s_arr_th, JR_arr_th, label = "Max J_R_th", color = hot, alpha = 0.5)

            axs[0,2].legend()
            axs[1,2].legend()
            axs[2,2].legend()
            #axs[:,2].legend()
        
        
        plt.tight_layout()
        return fig

    def make_eff_figure(self):
        fig, axs = plt.subplots(1, 3, figsize = (col*2,250*pt), layout = "constrained")
        self.eff_plot(axs, color = hot, label = "thermal", system = self.system_th,  filename = filenames[0])
        self.eff_plot(system = self.system_nth, color = nonthermal, axs = axs, label = "nonthermal", filename=filenames[1], style = "--")
        # axs[0,0].plot(JR_arr_nth, eff_arr_nth, label = "Max J_R nth", color = nonthermal, alpha = 0.5)
        # axs[1,0].plot(s_arr_nth, eff_arr_nth, label = "Max J_R nth", color = nonthermal, alpha = 0.5)
        # axs[2,0].plot(s_arr_nth, JR_arr_nth, label = "Max J_R nth", color = nonthermal, alpha = 0.5)
        # axs[0,0].plot(JR_arr_th, eff_arr_th, label = "Max J_R th", color = hot, alpha = 0.5)
        # axs[1,0].plot(s_arr_th, eff_arr_th, label = "Max J_R th", color = hot, alpha = 0.5)
        # axs[2,0].plot(s_arr_th, JR_arr_th, label = "Max J_R th", color = hot, alpha = 0.5)

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()       

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

    def eff_plot(self, axs, color,label,system=None,filename = None, style = '-'):
        if self.verbose:
            print("Making efficiency plot for ", label)
        JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr = self.get_eff_data(system, filename)
        eff_arr = JR_arr/avg_arr
        keep_index = np.argwhere(np.abs(err_arr) < 1e-5)
        JR_arr = JR_arr[keep_index]
        eff_arr = eff_arr[keep_index]
        s_arr = s_arr[keep_index]

        axs[0].plot(JR_arr, eff_arr, style,label=label, color = color)
        # axs[0].scatter(JR_arr, eff_arr, label=label, color = color, s = 1)

        axs[0].set_title("Efficiency vs cooling power")
        axs[0].set_xlabel("J_R")
        axs[0].set_ylabel(r"$\eta$")
        
        axs[1].plot(s_arr, eff_arr, style,label=label, color = color)
        # axs[1].scatter(s_arr, eff_arr, label=label, color = color, s = 1)
        axs[1].set_title("Efficiency vs muR")
        axs[1].set_xlabel(self.secondary_prop)
        axs[1].set_ylabel(r"$\eta$")

        axs[2].plot(s_arr, JR_arr, style,label=label, color = color)
        # axs[2].scatter(s_arr, JR_arr, label=label, color = color, s = 1)
        axs[2].set_title("Cooling power vs muR")
        axs[2].set_xlabel(self.secondary_prop)
        axs[2].set_ylabel(r"$J_R$")

    def noise_plot(self,  axs, color, label,system=None,filename = None):
        if self.verbose:
            print("Making noise plot for ", label)

        JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr = self.get_noise_data(system, filename)
        keep_index = np.argwhere(np.abs(err_arr) < 0.01)
        JR_arr = JR_arr[keep_index]
        noise_arr = noise_arr[keep_index]
        s_arr = s_arr[keep_index]
        axs[0].plot(JR_arr, noise_arr, label=label, color = color)
        axs[0].set_title("Efficiency vs cooling power")
        axs[0].set_xlabel("J_R")
        axs[0].set_ylabel(r"$\eta$")
        axs[1].plot(s_arr, noise_arr, label=label, color = color)
        axs[1].set_title("Efficiency vs cooling power")
        axs[1].set_xlabel(self.secondary_prop)
        axs[1].set_ylabel(r"$\eta$")
        axs[2].plot(s_arr, JR_arr, label=label, color = color)
        axs[2].set_title("Efficiency vs cooling power")
        axs[2].set_xlabel(self.secondary_prop)
        axs[2].set_ylabel(r"$J_R$")
    def product_plot(self,  axs, color,label,system=None,filename = None):
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
        axs[0].plot(JR_arr, prod_arr, label=label, color = color)
        axs[0].set_title("Efficiency vs cooling power")
        axs[0].set_xlabel("J_R")
        axs[0].set_ylabel(r"$\eta$")
        axs[1].plot(s_arr, prod_arr, label=label, color = color)
        axs[1].set_title("Efficiency vs cooling power")
        axs[1].set_xlabel(self.secondary_prop)
        axs[1].set_ylabel(r"$\eta$")
        axs[2].plot(s_arr, JR_arr, label=label, color = color)
        axs[2].set_title("Efficiency vs cooling power")
        axs[2].set_xlabel(self.secondary_prop)
        axs[2].set_ylabel(r"$J_R$")

    def produce_data_wrapper(self, system:two_terminals, opt_func, set_func, cond_in):
        JR_list = []
        avg_list = []
        C_list = []
        err_list = []
        noise_list = []
        k = 0
        for s in self.s_arr:
            if self.verbose:
                print("On point ", k)
            self.updater(s, system)
            C, err = opt_func(10, secondary_prop=self.secondary_prop)
            set_func(C)
            C_list.append(C)
            JR_list.append(system._current_integral(system.coeff_con,cond_in = cond_in(C)))
            avg_list.append(system._current_integral(system.coeff_avg, cond_in = cond_in(C)))
            noise_list.append(system._current_integral(system.coeff_noise, cond_in = cond_in(C)))
            err_list.append(err)
            
            if self.verbose:
                print("Error: ", err)
                print("JR: ", JR_list[k])
                print("Avg: ", avg_list[k])

            k += 1

        JR_arr = np.array(JR_list)
        avg_arr = np.array(avg_list)
        C_arr = np.array(C_list)
        err_arr = np.array(err_list)
        noise_arr = np.array(noise_list)
        return JR_arr, avg_arr, noise_arr, C_arr, err_arr


    def produce_eff_data(self, system:two_terminals):
        if self.verbose:
            print("Producing eff data")

        return self.produce_data_wrapper(system, system.optimize_for_best_avg, lambda C: system.set_transmission_avg_opt(C, system.coeff_con, system.coeff_avg), lambda C: system._avg_condition(C, system.coeff_con, system.coeff_avg))

    
    def produce_noise_data(self, system:two_terminals):
        if self.verbose:
            print("Producing noise data")
        return self.produce_data_wrapper(system, system.optimize_for_best_noise, system.set_transmission_noise_opt, system._noise_condition)


    def produce_product_data(self, system:two_terminals):
        if self.verbose:
            print("Producing product data")
        return self.produce_data_wrapper(system, system.optimize_for_best_product, system.set_ready_transmission_product, system._product_condition)



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
            noise = system.noise_cont(system.coeff_noise, cond_in = system.coeff_con)
            avg = system._current_integral(system.coeff_avg, cond_in = system.coeff_con)
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
    # midT = 1
    # deltaT = 0.8
    # deltamu = 0
    # muR = 1.2
    # TR = midT-deltaT
    # muL = 0#muR + deltamu
    # TL = midT

    midT = 1
    deltaT = 2
    deltamu = -1.5
    muR = 0
    TR = midT
    muL = muR + deltamu
    TL = midT+deltaT

    E_low = -5
    E_high = 5
    Es = np.linspace(E_low, E_high,1000)

    save_data = False
    load_params = True
    dist_type = "lorentz_peak_linear_norm"

    if load_params:
        th_dist_params = np.load("data/th_params_"+dist_type+".npz")['arr_0']
        muR = th_dist_params[0]
        TR = th_dist_params[1]
        nth_dist_params = np.load("data/nth_params_"+dist_type+".npz")['arr_0']
    else:
        th_dist_params = np.array([muR, TR])
        # nth_dist_params = np.array([muL, TL, 0.1 ,0.3, 1])
        nth_dist_params = np.array([-2, 5, -0.5, 1.2])
    occupf_L_nth = thermal_with_lorentz(*nth_dist_params)
    # occupf_L_nth = two_thermals(*nth_dist_params)
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

    thermal_left.debug = False
    nonthermal_left.debug = False
    #TODO: fix
    thermal_left.adjust_limits()
    nonthermal_left.adjust_limits()

    thermal_left.subdivide = True
    nonthermal_left.subdivide = True  

    fig_type = ".png"
    secondary = True
    if secondary:
        secondary_prop = "muR"
        filenames = ["data/th_"+dist_type+"_eff_"+secondary_prop + ".npz","data/nth_"+dist_type+"_eff_"+secondary_prop + ".npz","data/th_"+dist_type+"_noise_"+secondary_prop + ".npz",
                    "data/nth_"+dist_type+"_noise_"+secondary_prop + ".npz","data/th_"+dist_type+"_product_"+secondary_prop + ".npz","data/nth_"+dist_type+"_product_"+secondary_prop + ".npz"]
        max_filenames = ["data/th_max_"+dist_type+secondary_prop+".npz","data/nth_max_"+dist_type+secondary_prop+".npz"]
        secondaryPlot = SecondaryPlot(thermal_left, nonthermal_left, 0, 10, secondary_prop, n_points=500, verbose=True)
        if save_data:
            if not load_params:            
                np.savez("data/th_params_"+dist_type, th_dist_params)
                np.savez("data/nth_params_"+dist_type, nth_dist_params)
            secondaryPlot.save_eff(thermal_left, filenames[0])
            secondaryPlot.save_eff(nonthermal_left, filenames[1])
            # secondaryPlot.save_noise(thermal_left, filenames[2])
            # secondaryPlot.save_noise(nonthermal_left, filenames[3])            
            # secondaryPlot.save_product(thermal_left, filenames[4])
            # secondaryPlot.save_product(nonthermal_left, filenames[5])
            # secondaryPlot.save_max(thermal_left, max_filenames[0])
            # secondaryPlot.save_max(nonthermal_left, max_filenames[1])

            # secondaryPlot.fix_product_data(filenames[4], thermal_left)
            # secondaryPlot.fix_product_data(filenames[5], nonthermal_left)

        # else:
        #     np.load()
        JR_arr_th, avg_arr, noise_arr, C_arr, s_arr, err_arr = secondaryPlot.get_eff_data(thermal_left, filenames[0])
        JR_arr_nth, avg_arr, noise_arr, C_arr, s_arr, err_arr = secondaryPlot.get_eff_data(nonthermal_left, filenames[1])
        print(np.max(np.abs(JR_arr_nth[250:]-JR_arr_th[250:])))
        # fig = secondaryPlot.make_figure(make_eff=True, make_noise=True, make_product=True, filenames=filenames, max_filenames=max_filenames)
        # plt.suptitle("Optimized plots for " + dist_type + r", metric $\dot S_R/\dot S_L$")
        # plt.suptitle("Optimized plots for " + dist_type + r", metric $\dot S_R/\dot S_L$")
        # plt.tight_layout()
        # plt.savefig("figs/"+dist_type+"_"+secondary_prop+fig_type, dpi = 500)

        fig = secondaryPlot.make_eff_figure()
        plt.savefig("figs/"+dist_type+"_"+secondary_prop+"_only_eff_"+fig_type, dpi = 500)

        # JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr =secondaryPlot.get_product_data(filename=filenames[5])
        # print(JR_arr)
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
        # JR_arr,_,_,prod_max, _= secondaryPlot.get_max_data(max_filenames[1])
        # print(JR_arr)
        # print(prod_max)
        # plt.plot(Es, max_transf(Es))
          
    else:   

           
        powerPlot = PowerPlot(thermal_left, nonthermal_left, True, n_targets=500)

        fig = powerPlot.make_dist_figure(-5,5)
        plt.savefig("figs/"+dist_type+"_dist"+fig_type, dpi = 500)        

        filenames = ["data/th_"+dist_type+"_eff.npz","data/nth_"+dist_type+"_eff.npz","data/th_"+dist_type+"_noise.npz",
                    "data/nth_"+dist_type+"_noise.npz","data/th_"+dist_type+"_product.npz","data/nth_"+dist_type+"_product.npz"]
        
        if save_data:
            if not load_params:            
                np.savez("data/th_params_"+dist_type, th_dist_params)
                np.savez("data/nth_params_"+dist_type, nth_dist_params)
            # powerPlot.save_eff(thermal_left, filenames[0])
            # powerPlot.save_eff(nonthermal_left, filenames[1])
            # powerPlot.save_noise(thermal_left, filenames[2])
            # powerPlot.save_noise(nonthermal_left, filenames[3])            
            # powerPlot.save_product(thermal_left, filenames[4])
            powerPlot.save_product(nonthermal_left, filenames[5])
            powerPlot.save_example(0.4, "data/"+dist_type+"_example.npz")
        # else:
        #     np.load()
        

        fig = powerPlot.make_example_figure("data/"+dist_type+"_example.npz", make_eff=True, make_noise = True)
        plt.savefig("figs/"+dist_type+"_example"+fig_type, dpi = 500)

        fig = powerPlot.make_figure(make_eff=True, make_noise=True, make_product=True, filenames=filenames)
        #plt.suptitle("Optimized plots for " + dist_type + r", metric $\dot S_R/\dot S_L$")
        plt.suptitle("Optimized quantifiers over cooling power spectrum")
        #plt.tight_layout()
        plt.savefig("figs/"+dist_type+fig_type, dpi = 500)
        fig = powerPlot.make_crossing_figure(filenames[1], nonthermal_left)
        plt.savefig("figs/"+dist_type+"_crossing"+fig_type, dpi = 500)
        
        fig = powerPlot.make_all_crossing_figure([filenames[1],filenames[3],filenames[5]], nonthermal_left)
        plt.savefig("figs/"+dist_type+"_crossing_all"+fig_type, dpi = 500)

        fig = powerPlot.make_char_eff_figure(filenames[1], nonthermal_left)
        plt.savefig("figs/"+dist_type+"_char_eff"+fig_type, dpi = 500)
        # # fig = plt.figure()
        # #JR_arr, noise_arr, C_arr, err_arr =powerPlot.get_product_data(filename=filenames[5])
        # JR_arr, eff_arr, C_arr = powerPlot.get_eff_data(filename=filenames[1])
        # idx = [94]
        # # print(JR_arr)
        # # print(JR_arr[:-1]-JR_arr[1:])
        # print(np.argmax(np.abs(JR_arr[:-1]-JR_arr[1:])))
        # print(C_arr[idx])
        # print(JR_arr[idx])

        # # C_res,err = nonthermal_left.optimize_for_avg((JR_arr[593]+JR_arr[592])/2,5)
        # # C_res,err = nonthermal_left.optimize_for_avg(JR_arr[93],5)
        # #print(C_res,err)
        # Es = np.linspace(nonthermal_left.E_low, nonthermal_left.E_high, 100000)
        # # nonthermal_left.transf = nonthermal_left._transmission_avg(C_res, nonthermal_left.coeff_con, nonthermal_left.coeff_avg)
        # # print(nonthermal_left._current_integral(nonthermal_left.coeff_con))
        # #print((JR_arr[593]+JR_arr[592])/2)
        # #print((JR_arr[593]+JR_arr[592])/2 - nonthermal_left._current_integral(nonthermal_left.coeff_con))
        # test_transf = nonthermal_left._transmission_avg(C_arr[94], nonthermal_left.coeff_con, nonthermal_left.coeff_avg)
        # test_cond = nonthermal_left._avg_condition(C_arr[94], nonthermal_left.coeff_con, nonthermal_left.coeff_avg)
        # signed = np.sign(test_transf(Es))
        # roots_init = Es[np.argwhere(signed[1:] - signed[:-1] != 0).flatten()]
        # roots = ut.root_finder_guesses(test_cond,roots_init, tol = 1e-8)
        # roots = np.sort(roots)
        # for i in range(0,len(roots),2):
        #     print(i)

        # #test_transf = lambda E: np.heaviside(E - roots[0],0)*np.heaviside(roots[3]-E,0)+ np.heaviside(E - roots[4],0)*np.heaviside(roots[5]-E,0)
        # #test_transf = lambda E: np.heaviside(E - roots[0],0)*np.heaviside(roots[1]-E,0)+ np.heaviside(E - roots[2],0)*np.heaviside(roots[3]-E,0)+ np.heaviside(E - roots[4],0)*np.heaviside(roots[3]-E,0)
        
        # integrand = lambda E: nonthermal_left.coeff_con(E)*(nonthermal_left.occupf_L(E)- nonthermal_left.occupf_R(E))
        # # urrent= integrate.quad(integrand, roots[0], roots[1], limit = 1000)[0] + integrate.quad(integrand, roots[2], roots[3], limit = 1000)[0] + integrate.quad(integrand, roots[4], roots[5], limit = 1000)[0]
        
        # # nonthermal_left.transf = test_transf
        # # print(nonthermal_left._current_integral(nonthermal_left.coeff_con))
        # # print(current)
        # fig = plt.figure()
        
        # #plt.plot(Es, nonthermal_left.occupf_L(Es), label = "Nonthermal", color = nonthermal, zorder = 3)
        # #axs[0].plot(Es, self.system_th.occupf_L(Es), label = "Thermal probe", color = hot, zorder = 2)
        # #plt.plot(Es, nonthermal_left.occupf_R(Es), label = "Cold thermal", color = cold, zorder = 2)
        # plt.plot(Es, nonthermal_left.coeff_con(Es)*(nonthermal_left.occupf_L(Es)-nonthermal_left.occupf_R(Es)))
        # for C_avg in C_arr[idx]:
        #     transf_avg = nonthermal_left._transmission_avg(float(C_avg), nonthermal_left.coeff_con, nonthermal_left.coeff_avg)
        #     #axs[1].plot(Es, transf_avg(Es), color = "#c870ff", label = "Best eff.")
        #     plt.plot(Es- nonthermal_left.muR, transf_avg(Es), label = C_avg)
        # plt.plot(Es, test_transf(Es))
        # plt.legend()


