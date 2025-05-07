
import numpy as np
import matplotlib.pyplot as plt
import results
import copy
from thermo_funcs import two_terminals
from scipy import integrate
def check_buttiker_probe(system):
    sys_copy = copy.deepcopy(system)
    Es = np.linspace(sys_copy.E_low, sys_copy.E_high, 1000)
    plt.plot(Es, sys_copy.occupf_R(Es), label = "probe start")
    print("Old energy/particle currents: ",sys_copy.calc_left_energy_current(), sys_copy.calc_left_particle_current())
    print("Old mu/TR: ", sys_copy.muR, sys_copy.TR)
    results.buttiker_probe(sys_copy)
    
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

if __name__ == "__main__":
    midT = 1
    deltaT = 0.8
    deltamu = 0
    muR = 3
    TR = midT-deltaT
    muL = 0#muR + deltamu
    TL = midT

    E_low = -5
    E_high = 5
    Es = np.linspace(E_low, E_high,1000)
    check_probe= True
    check_avg = False
    check_noise = False
    check_product = False
    check_max = True
    check_cond = False
    load_params = False
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
    occupf_L_nth = results.thermal_with_lorentz(*nth_dist_params)
    #occupf_L_nth = results.two_thermals(*nth_dist_params)
    #dist_params = np.array([muL,TL,muR,TR])


    left_virtual = two_terminals(-20, 20, occupf_L = occupf_L_nth, muL=muL, muR=muR, TL=TL, TR=TR)
    if check_probe:
        check_buttiker_probe(left_virtual)
        plt.show()
    results.buttiker_probe(left_virtual)

    thermal_left = two_terminals(E_low, E_high, muL=left_virtual.muR, TL = left_virtual.TR, muR = muR, TR = TR, N = 1, subdivide=False, debug=True)
    nonthermal_left = two_terminals(E_low, E_high, occupf_L= occupf_L_nth, muL=muL, TL = TL, muR = muR, TR = TR, N = 1, subdivide=False, debug = True)

    thermal_left.coeff_con = lambda E: -thermal_left.right_entropy_coeff(E)
    thermal_left.coeff_avg = lambda E: thermal_left.left_entropy_coeff(E)#thermal_left.left_noneq_free_coeff(E)#
    thermal_left.coeff_noise = lambda E: -thermal_left.right_entropy_coeff(E)

    nonthermal_left.coeff_con = lambda E: -nonthermal_left.right_entropy_coeff(E)
    nonthermal_left.coeff_avg = lambda E: nonthermal_left.left_entropy_coeff(E)#nonthermal_left.left_noneq_free_coeff(E)#
    nonthermal_left.coeff_noise = lambda E: -nonthermal_left.right_entropy_coeff(E)

    #TODO: fix
    thermal_left.adjust_limits()
    nonthermal_left.adjust_limits()




    active_system = nonthermal_left
    active_system.debug = True
    active_system.muR = 1
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

        # check_avg_optimization(active_system)
        if plot_mus:
                    
            C_list = []
            Iy_list = []
            Ix_list = []
            dd_list = []
            mus = np.linspace(0.4,1.4,20)
            active_system.E_high = 2
            active_system.E_low = -2
            for mu in mus:
                active_system.TR = mu
                active_system.set_fermi_dist_right()
        # C_list = []
        # Iy_list = []
        # dd_list = []
        # mus = np.linspace(3,5,20)
        # for mu in mus:
                # active_system.muR = mu
                # active_system.set_fermi_dist_right()
                
                #plt.show()
                active_system.adjust_limits()
                C = active_system.optimize_for_avg(0.01, 10)
                
                Es = np.linspace(active_system.E_low, active_system.E_high,1000)
                
        #         print(active_system.E_high)
        #         print(active_system.E_low)
                
        #         print(active_system._current_integral(active_system.coeff_con))
        #         print(mu)
                
        #         plot_max(active_system)
                
        #         plt.plot(Es, active_system.transf(Es), label = "Opt transf")
        # #plt.plot(Es, test_occup(Es, -0.5), label = "init")
        #         plt.legend()
        #         plt.show()
                
                # res = check_avg_optimization(active_system, False,10)
                # plt.show()
                Iy = active_system._current_integral(active_system.coeff_avg)
                davg_integrand = active_system.dSL_dTR(active_system.transf)#lambda E: active_system.coeff_avg(E)*active_system.transf(E)*(active_system.occupf_R(E)**2 *(np.exp((E-active_system.muR)/active_system.TR))/active_system.TR)
                dcon_integrand = active_system.dSR_dTR(active_system.transf)#lambda E: active_system.transf(E)*((active_system.occupf_L(E)- active_system.occupf_R(E))/TR - active_system.coeff_con(E)*(active_system.occupf_R(E)**2 *(np.exp((E-active_system.muR)/active_system.TR))/active_system.TR))
                davg_current, err = integrate.quad(davg_integrand, active_system.E_low, active_system.E_high, args=(), points=active_system.occuproots, limit = 100)
                dcon_current, err = integrate.quad(dcon_integrand, active_system.E_low, active_system.E_high, args=(), points=active_system.occuproots, limit = 100)
                dd_list.append(davg_current/dcon_current if dcon_current != 0 else 0)
                
                C_list.append(C)
                Iy_list.append(Iy)
                #plt.show()
                # Es = np.linspace(-2,4)
                # active_system.adjust_limits()
                # print(active_system.occuproots)
                
                # Ix = active_system._current_integral(active_system.coeff_con)
                # Iy = active_system._current_integral(active_system.coeff_avg)
                # C_list.append(C)
                # Iy_list.append(Iy)
                # Ix_list.append(Ix)
            #active_system.muR = 4.5
            #active_system.set_fermi_dist_right()
            
            #active_system.adjust_limits()
            #C = active_system.optimize_for_best_avg(10)
            #print(active_system._current_integral(active_system.coeff_con))
            C_list = np.array(C_list)
            Ix_list = np.array(Ix_list)
            # Iy_list = np.array(Iy_list)
                            # eff_list = Ix_list/Iy_list
            plt.plot(mus, 0.01/np.array(Iy_list), label = "effs")
            plt.vlines(mus[np.argmax(0.01/np.array(Iy_list))], 0.9, 1, colors="red")
            #plt.show()
            plt.plot(mus,np.array(C_list), label = "Cs")
            plt.plot(mus, np.array(dd_list), label = "dds")
            plt.legend()
            plt.show()                    

            # plt.plot(mus, eff_list)
            #plt.vlines(mus[np.argmax(0.1/np.array(Iy_list))], 0, 1)
            #plt.show()
            #plt.plot(mus,np.array(C_list)-1)
            #plt.plot(mus, np.array(dd_list)-1)
            # plt.show()

            # plt.plot(Ix_list, eff_list)
            # plt.show()

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
