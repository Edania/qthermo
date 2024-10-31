import thermo_funcs as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.size'] = '15'

#plt.rcParams['font.sans-serif'] = 'Arial'
#plt.rcParams["mathtext.default"] = 'rm'


from scipy.optimize import minimize, fsolve
def produce_graphs(occupf_L, suffix, filetype):
    # Calculate assuming band pass
    E0_band = tf.E_max(init_muL,TL,muR,TR)
    #E1_band = tf.correct_E1(E_range, muL, TL, muR, TR, occupf_L, target_power)
    #            res = minimize(maxf, [0.5*E_max(muL, TL, muR, TR),1.5*E_max(muL, TL, muR, TR)], args = (muL, TL, muR, TR), 
    #                    constraints= [{'type':'ineq', 'fun': lambda E: E[1] - E[0]},
    #                                  {'type': 'eq', 'fun':constrf, 'args':(muL, TL, muR, TR,target_eff)}]).x
    [muL, E1_band] = minimize(tf.opt_for_eff, [init_muL, E0_band*2], args = (TL, muR,TR,occupf_L), constraints={'type':'eq', 'fun': tf.constrain_pgen, 'args': (TL,muR,TR,occupf_L,target_power)}).x
    E0_band = tf.E_max(muL,TL,muR,TR)
    #print(E1_band)
    heatL_band = tf.current_integral(E0_band, E1_band,muL,TL,muR,TR, lambda E: N, occupf_L, type = "heat")
    heatR_band = tf.current_integral(E0_band, E1_band,muL,TL,muR,TR, lambda E: N, occupf_L, type = "heatR")
    electric_band = tf.current_integral(E0_band, E1_band,muL,TL,muR,TR, lambda E: N, occupf_L, type = "electric")
    transf_band = N*np.heaviside(E_range-E0_band,1) * np.heaviside(E1_band-E_range,1)
    #heatL_band = tf.slice_current_integral(transf_band, E_range, muL, TL ,muR, TR, deltaE, occupf_L, type = "heat")
    #heatR_band = tf.slice_current_integral(transf_band, E_range, muL, TL ,muR, TR, deltaE, occupf_L, type = "heatR")
    
    
    power_band = heatL_band+heatR_band
    eff_band = power_band/heatL_band
    Jprim, Pprim, dJdT,dPdT = tf.calc_dJR_dP_dmu(transf_band,E_range,muL,TL,muR,TR,deltaE,occupf_L)
    print(E1_band-(muL*(1 - Jprim/Pprim)))

    # Optimize with arbitrary transmission function
    init_transf = np.random.uniform(0,1,len(small_E_range))
    res = minimize(tf.slice_maximize_eff, init_transf, args = (small_E_range, muL, TL ,muR, TR, small_deltaE,occupf_L ), bounds=(((0,1),)*len(small_E_range)),
            constraints = [{'type':'eq', 'fun': tf.slice_pow_constraint, 'args':(small_E_range, muL, TL ,muR, TR, small_deltaE,target_power,occupf_L)}])
    transf_gen = res.x
    heatL_gen = tf.slice_current_integral(transf_gen,small_E_range,muL,TL,muR,TR, small_deltaE,occupf_L, type = "heat")
    heatR_gen = tf.slice_current_integral(transf_gen,small_E_range,muL,TL,muR,TR, small_deltaE,occupf_L, type = "heatR")
    electric_gen = tf.slice_current_integral(transf_gen,small_E_range,muL,TL,muR,TR, small_deltaE,occupf_L, type = "electric")
    power_gen = heatL_gen+heatR_gen
    eff_gen = power_gen/heatL_gen
    el_power_gen = -(muL-muR)*electric_gen
    # Optimize by solving equation for constant

    res1 = fsolve(tf.opt_transf, 0.4, args=(E_range, muL, TL, muR, TR, target_power, deltaE, occupf_L), factor = 1, maxfev=1000)

    c_eta = res1[0]

    mom_etas = (E_range-tf.entropy_coeff(E_range, muL, TL, occupf_L))/(-tf.entropy_coeff(E_range, muL, TL, occupf_L))
    occupdiff = occupf_L(E_range,muL,TL)- tf.fermi_dist(E_range,muR,TR)
    transf_ceta = np.zeros_like(mom_etas)
    transf_ceta[mom_etas > c_eta] = 1
    transf_ceta[occupdiff < 0] = 0
    #transf_ceta = np.ones_like(E_range)
    electric_ceta, integrands = tf.slice_current_integral(transf_ceta,E_range, muL, TL ,muR, TR, deltaE, occupf_L,type = "electric", return_integrands=True)
    #print(integrands)
    heatL_ceta, heatL_inters = tf.slice_current_integral(transf_ceta, E_range, muL, TL ,muR, TR, deltaE, occupf_L, type = "heat", return_integrands=True)
    heatR_ceta, heatR_inters = tf.slice_current_integral(transf_ceta, E_range, muL, TL ,muR, TR, deltaE, occupf_L, type = "heatR", return_integrands=True)
    power_ceta = heatL_ceta+heatR_ceta
    eff_ceta = power_ceta/heatL_ceta
    el_power_ceta = -(muL-muR)*electric_ceta
    
    #mom_etas *= np.sign(heatL_inters)

    E1_stuff = -tf.entropy_coeff(E_range, muL, TL, occupf_L)/(E_range-tf.entropy_coeff(E_range, muL, TL, occupf_L)) - 1/c_eta
    #E1_stuff = -(E_range-tf.entropy_coeff(E_range, muL, TL, occupf_L))/tf.entropy_coeff(E_range, muL, TL, occupf_L) - c_eta

    nonzero_idxs = np.argwhere(transf_ceta > 1e-5)
    mom_etas = np.ones_like(E_range)*np.nan
    mom_etas[nonzero_idxs] = (heatR_inters[nonzero_idxs] + heatL_inters[nonzero_idxs])/heatL_inters[nonzero_idxs]
    disc_E_range = np.ones_like(E_range)*np.nan
    disc_E_range[nonzero_idxs] = E_range[nonzero_idxs]
    mom_etas_diff = lambda E: (E-tf.entropy_coeff(E, muL, TL, occupf_L))/(-tf.entropy_coeff(E, muL, TL, occupf_L))
    occupdiff = lambda E: occupf_L(E,muL,TL)- tf.fermi_dist(E,muR,TR)
    occup_zero = E_range[np.where(np.sign(occupdiff(E_range))[:-1] - np.sign(occupdiff(E_range))[1:] < 0)[0]]
    occup_zero = np.array([occup_zero])
    E_starts = np.linspace(E_range[0], E_range[-1], 10)
    mom_etas_zero = fsolve(mom_etas_diff, E_starts, factor=0.1)
    mom_etas_zero = np.unique(mom_etas_zero.round(decimals=3))

    norm = True
    dP_dtau = tf.calc_dP_dtau(E_range, muL,TL,muR,TR,occupf_L)
    zero_area = E_range[np.argwhere(np.abs(heatL_inters)<1e-8)]
    dJR_dtau = E1_stuff*dP_dtau


    print(f"c_eta = {c_eta}")
    print(f"Pprim/Jprim = {Pprim/Jprim}")
    print(f"Temperature bias: {TL-TR}")
    print(f"Potential bias: {muL-muR}")
    print(f"Exchange energy: {E0_band}")
    
    print(f"\nFrom chunk optimization:")
    print(f"Power output: {power_gen}, electric = {el_power_gen}")
    print(f"Efficiency: {eff_gen} ")
    print(f"(Target: {target_power})")

    print(f"\nFrom c_eta optimization:")
    print(f"Power output: {power_ceta}, electric = {el_power_ceta}")
    print(f"Efficiency: {eff_ceta} ")
    print(f"(Target: {target_power})")

    print(f"\nFrom regular optimization")
    print(f"Power: {power_band}")
    print(f"Eff: {eff_band}")

    fig = plt.figure(figsize=(4.5,4))

    plt.bar(small_E_range, transf_gen, align='edge', width=small_deltaE, label = "Gen. opt.", zorder =1)
    plt.plot(E_range, transf_ceta, 'red', zorder =2, label = fr"$c_\eta$ opt.")
    plt.plot(E_range, transf_band, '--',color = 'black', alpha = 1, label = "BP opt.", zorder = 3)
    plt.title(f"Transmission for\n maximized efficiency, {suffix}")
    plt.ylabel("Transmission function")
    plt.xlabel(fr"Electron energy/$k_B \bar T$")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"figs/transmission_{suffix}.{filetype}", dpi=1200)

    fig, axs = plt.subplots(1, figsize = (4,3.5))
    axs.plot(E_range, tf.fermi_dist(E_range, muR, TR), label = "R dist.", zorder = 1)
    if suffix == "thermal":
        axs.plot(E_range, tf.fermi_dist(E_range, muL, TL), label = "L dist.", zorder = 2)
    else:
        axs.plot(E_range, tf.fermi_dist(E_range, muL, TL), '--', label = "L Fermi", zorder = 2)
        axs.plot(E_range, occupf_L(E_range,muL, TL), label = "L dist.", zorder = 3)
    
    axs.set_title(fr"Distribution functions, $\Delta \mu/k_b \bar T = ${muL:.3f}")
    axs.set_xlabel(fr"Electron energy/$k_B \bar T$")
    axs.set_ylabel("Occupation probability")
    axs.legend()
    
    plt.tight_layout()
    #axs[1].plot(E_range, tf.entropy_coeff(E_range, muL, TL, occupf_L), label = "perturbed")
    #axs[1].plot(E_range, E_range - muL, label = "thermal")
    #axs[1].set_title("Comparison of coefficient")
    #axs[1].legend()
    plt.savefig(f"figs/dist_{suffix}.{filetype}", dpi=1200)


    fig, axs = plt.subplots(1,1, figsize=(4.5,4))
    #axs[0].plot(E_range, heatL_inters, label = "heat L")
    #axs[0].plot(E_range, heatR_inters, label = "heat R")
    #axs[0].plot(E_range, heatR_inters + heatL_inters, label = "power")
    #axs[0].set_title("Momentary heat and power")
    #axs[0].legend()
    nC = tf.carnot(TL,TR)
    axs.plot(disc_E_range, mom_etas/nC, '--', color = 'b', zorder = 3, label = rf"$\tau_\gamma = 1$")
    if suffix == "thermal":
        axs.plot(E_range, 1/nC*1/(1-E_range/muL),zorder = 1, label = rf"$\eta_\gamma/\eta_C$")
    else:
        axs.plot(E_range, 1/nC*1/(1-E_range/muL),zorder = 1, label = rf"Thermal $\eta_\gamma/\eta_C$")
        axs.plot(E_range, 1/nC*(E_range-tf.entropy_coeff(E_range, muL, TL, occupf_L))/(-tf.entropy_coeff(E_range, muL, TL, occupf_L)), zorder = 2, label = rf"Nonthermal $\eta_\gamma/\eta_C$")
    axs.set_ylim([0,1])
    axs.set_xlim([E0_band, E_range[-1]])
    axs.hlines(c_eta/nC, E_range[0], E_range[-1], 'r', label = fr"$c_\eta/\eta_C$")
    #axs.hlines(tf.carnot(TL,TR), E_range[0], E_range[-1], 'g', label = fr"$Carnot$")
    #axs.vlines(occup_zero, 0, 1)
    axs.set_xlabel(fr"Electron energy/$k_B \bar T$")
    axs.set_ylabel(fr"$\eta_\gamma/\eta_C$")
    axs.legend()
    axs.set_title(rf"Efficiency spectrum for $\epsilon_\gamma > \epsilon_0$, {suffix}")
    plt.tight_layout()
    plt.savefig(f"figs/sliced_eff_{suffix}.{filetype}", dpi=1200)

    fig, axs = plt.subplots(1,2, figsize = (10,10))
    if norm:
        axs[0].plot(E_range, dP_dtau/np.max(np.abs(dP_dtau)), label = "dP/dtau|mu")
        #axs[0].plot(E_range, tf.fermi_dist(E_range, muL, TL) + fpertub(E_range), label = "L perturb")

        #axs[0].plot(E_range, tf.fermi_dist(E_range, muR, TR), label = "R")

        axs[0].plot(E_range, dJR_dtau/np.max(np.abs(dJR_dtau)), label = "dJL/dtau|Pgen")
        #print(E_range/(E_range+TL*kb*tf.entropy_coeff(E_range, muL, TL, occupf_L)) - Jprim/Pprim)
        axs[0].plot(E_range, E1_stuff/np.max(np.abs(E1_stuff)), label = "E1")
        axs[0].scatter(zero_area, np.zeros_like(zero_area), c = 'green', s = 5, zorder = 2, label = "JL = 0")
        axs[0].hlines(0, E_range[0], E_range[-1], colors = 'red', zorder = 1)
        #axs[0].vlines(E1,-1,1, colors='red', label = "E1")

    else:
        axs[0].plot(E_range, dP_dtau, label = "dJL/dtau|mu")
        axs[0].plot(E_range, dJR_dtau, label = "dJL/dtau|Pgen")
        axs[0].plot(E_range, E1_stuff, label = "E1")
    axs[0].set_title("dJ/dtau at fixed Pgen, normalized")
    axs[0].legend()
    plt.savefig(f"figs/dJdTau_{suffix}.{filetype}", dpi=1200)
if __name__ == "__main__":
    # Set up variables
    T = 1
    deltaT = 0.5
    TL = T+deltaT
    TR = T-deltaT
    muR = 0
    init_muL = -1
    deltamu = init_muL-muR
    N = 1
    tf.N = N
    target_power = 0.6*tf.pmax(TL,TR)
    small_E_range = np.linspace(-1,5,100)
    small_deltaE = small_E_range[1]-small_E_range[0]
    
    E_range = np.linspace(-1,5,100000)
    deltaE = E_range[1]-E_range[0]
    
    ## THERMAL CASE ##
    th_occupf_L = tf.fermi_dist
    th_suffix = "thermal"
    
    ## NON-THERMAL CASE ## 
    
    def def_pert(E, muL):
        E0 = tf.E_max(muL,TL,muR,TR)
        #pert = 0.5*fermi_dist(E, muL+0.1, TL + 0.1)  - 0.5*fermi_dist(E,muL,TL)
        #ratio = ((-1-E0)/(5-E0)).as_integer_ratio()
        pert = 0.1*np.sin((E - E0)*8*np.pi/(5-E0))*np.exp(-0.5*np.abs(E-E0))
        #print(type(E))
        if type(E) == np.ndarray:
            pert[E > 5] = 0
            pert[E < -1] = 0
        else:
            if E > 5:
                pert = 0
            if E < -1:
                pert = 0
        #pert = np.ones_like(E)*0.2
        #pert[E<0.2] = 0
        #pert[E>1] = 0 
        #pert[pert > 1] = 1
        #pert[pert < 0] = 0
        return pert
    fpertub = lambda E, muL: def_pert(E, muL)
    nth_foccup_L = lambda E, muL, TL: tf.pertub_fermi(E, muL, TL, fpertub(E,muL))
    nth_suffix = "nonthermal"


    produce_graphs(th_occupf_L, th_suffix, "pdf")
    produce_graphs(nth_foccup_L,nth_suffix,"pdf")
