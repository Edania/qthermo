#Define functions
import numpy as np
import mpmath as mp
from scipy import integrate
from scipy.optimize import minimize, fsolve

## GLOBAL CONSTANTS ##
h = 1
kb = 1
e = 1
N = 1
## FUNCTION DEFINITIONS ##
def fermi_dist(E, mu, T):
    f_dist = 1/(1+np.exp((E-mu)/(T*kb)))
    return f_dist

def current_integral(E_low, E_high, muL, TL, muR, TR, transf, occupf_L = fermi_dist, occupf_R = fermi_dist, type = "electric"):
    if type == "electric":
        coeff = lambda E: e
    elif type == "heat":
        if occupf_L != fermi_dist:
            coeff = lambda E: entropy_coeff(E,muL,TL,occupf_L)
        else:
            coeff = lambda E: (E-muL)
    elif type == "heatR":
        coeff = lambda E: -(E-muR)
    elif type == "energy":
        coeff = lambda E: E
    else:
        print("Invalid current type")
        return -1
    occupdiff = lambda E: occupf_L(E, muL, TL) - occupf_R(E, muR, TR)
    
    integrand = lambda E: 1/h*coeff(E)*transf(E)*(occupdiff(E))
    current, err = integrate.quad(integrand, E_low, E_high, args=())
    return current

# To make sure that we are in the heat engine regime (optimizing gets weird otherwise.)
# However this constraint seems to lead to a badly behaved problem. It could be better/easier to filter the output data afterwards to remove the results where power < 0
def carnot(TL,TR):
    return (1-TR/TL)

def cop(TL, TR):
    return TR/(TL-TR)

def pmax(TL, TR):
    A = 0.0321
    p = A * np.pi**2/h * N * kb**2 *(TL-TR)**2
    return p

def E_max(muL, TL, muR, TR):
    if TL-TR == 0:
        return 0
    #print(((deltaT+T)*mu - T*(deltamu+mu))/deltaT)
    return (TL*muR - TR*muL)/(TL-TR)

def opt_for_eff(theta, TL, muR, TR, occupf_L):
    muL = theta[0]
    E1 = theta[1]
    E0 = E_max(muL,TL,muR,TR)
    heatL = current_integral(E0, E1,muL,TL,muR,TR, lambda E: N, occupf_L, type = "heat")
    return heatL
    #heatR = current_integral(E0, E1,muL,TL,muR,TR, lambda E: N, occupf_L, type = "heatR")
    #pgen = heatL + heatR

def constrain_pgen(theta, TL, muR, TR, occupf_L, target_p):
    muL = theta[0]
    E1 = theta[1]
    E0 = E_max(muL,TL,muR,TR)
    heatL = current_integral(E0, E1,muL,TL,muR,TR, lambda E: N, occupf_L, type = "heat")
    heatR = current_integral(E0, E1,muL,TL,muR,TR, lambda E: N, occupf_L, type = "heatR")
    pgen = heatL + heatR
    return pgen - target_p


def correct_E1(E_range, muL, TL, muR, TR, occupf_L, target_p):
    def GL(E):
        return kb*TL/h * np.log(1+np.exp(-(E-muL)/(kb*TL)))
    def GR(E):
        return kb*TR/h * np.log(1+np.exp(-(E-muR)/(kb*TR)))
    E0 = E_max(muL, TL, muR, TR)
    GL_0 = GL(E0)
    GR_0 = GR(E0)
    #print(FL_prim_0)
    def e_solver(E1):
        #transf = np.zeros_like(E_range)
        #transf[E_range > E0] = 1
        #transf[E_range > E1] = 0
        #heatL = slice_current_integral(transf, E_range, muL, TL ,muR, TR, E_range[1]-E_range[0], occupf_L, type = "heat")
        #heatR = slice_current_integral(transf, E_range, muL, TL ,muR, TR, E_range[1]-E_range[0], occupf_L, type = "heatR")
        heatL = current_integral(E0, E1,muL,TL,muR,TR, lambda E: N, occupf_L, type = "heat")
        heatR = current_integral(E0, E1,muL,TL,muR,TR, lambda E: N, occupf_L, type = "heatR")
        pgen = heatL + heatR
        #pgen = muL*(-GL_0+GR_0+GL(E1)-GR(E1))
        #print(pgen)
        return pgen - target_p
        #J_prim = FL_prim_0 - FL_prim(E1)
        #P_prim = e*(GL_0+GR_0-GL(E1)-GR(E1)) + muL*(GL_prim_0-GL_prim(E1))
        #if P_prim == 0:
        #    return E0
        #print(GL_0+GR_0-GL(E1)-GR(E1))
        #print((J_prim/P_prim))
        #return float((-(1-J_prim/P_prim)*muL - E1)[0])

    res = fsolve(e_solver, E0*5)
    return res[0] 

def correct_Jprim_Pprim(E0,E1,muL_real, TL, muR_real, TR):
    deltamu = muL_real - muR_real
    muR = 0
    muL = deltamu
    polylog = np.frompyfunc(mp.polylog, 2, 1)
    if muR != 0:
        print("Only for muR = 0 for now :(")
        return -1
    def GL(E):
        return kb*TL/h * np.log(1+np.exp(-(E-muL)/(kb*TL)))
    def GR(E):
        return kb*TR/h * np.log(1+np.exp(-(E)/(kb*TR)))
    def FL(E):
        t_one = E*GL(E)
        t_two = - (kb*TL)**2/h * polylog(2, -np.exp(-(E-muL)/(kb*TL)))
        t_three = muL*(E - kb*TL*np.log(np.exp(E/(kb*TL)) + np.exp(muL/(kb*TL))))
        return t_one + t_two + t_three
    def FR(E):
        t_one = (E-muL)*GR(E)
        t_two = - (kb*TR)**2/h * polylog(2, -np.exp(-(E)/(kb*TR)))
        return t_one + t_two
    
    def GL_prim(E):
        return 1/h * np.exp((-E+muL)/(kb*TL))/(1+np.exp((-E+muL)/(kb*TL)))
    def FL_prim(E):
        t_one = E*GL_prim(E)
        t_two = kb*TL/h * np.log(1+np.exp((-E+muL)/(kb*TL)))
        t_three = - kb*TL *np.log(np.exp(E/(kb*TL)) + np.exp(muL/(kb*TL)))
        t_four = E - muL*np.exp(muL/(kb*TL))/(np.exp(E/(kb*TL))+np.exp(muL/(kb*TL)))
        return t_one + t_two + t_three + t_four
    def FR_prim(E):
        return - GR(E)
    GL_0 = GL(E0)
    GL_prim_0 = GL_prim(E0)
    GR_0 = GR(E0)
    FL_prim_0 = FL_prim(E0)
    Jprim = FL_prim_0 - FL_prim(E1) - FR_prim(E0) + FR_prim(E1)
    Pprim = -(GL_0-GR_0-GL(E1)+GR(E1) + muL*(GL_prim_0-GL_prim(E1)))

    JL = FL(E0) - FL(E1) - FR(E0) + FR(E1)
    P = muL*(-GL(E0) + GL(E1) + GR(E0) - GR(E1))

    return (float(Jprim), float(Pprim), float(JL), float(P)) 

def entropy_coeff(E, muL, TL, occupf_L = fermi_dist):
    coeff = -kb*TL*np.log(occupf_L(E, muL, TL)/(1-occupf_L(E, muL, TL)))
    return coeff

def calc_dP_dtau(E, muL, TL, muR, TR, occupf_L = fermi_dist, occupf_R= fermi_dist):
    coeff = -entropy_coeff(E, muL, TL, occupf_L)
    dP_dtau = -1/h * (E - muR + coeff) * (occupf_L(E, muL, TL) - occupf_R(E,muR,TR))
    return dP_dtau


def slice_current_integral(transf,E_mids,muL, TL ,muR, TR, occupf_L = fermi_dist, occupf_R=fermi_dist,type = "electric", return_integrands = False):
    if type == "electric":
        coeff = lambda E: e
    elif type == "heat":
        if occupf_L != fermi_dist:
            coeff = lambda E: entropy_coeff(E,muL,TL,occupf_L)
        else:
            coeff = lambda E: (E-muL)
    elif type == "heatR":
        coeff = lambda E: -(E-muR)
    elif type == "energy":
        coeff = lambda E: E
    else:
        print("Invalid current type")
        return -1

    #if type == "heatR":
    #    occupdiff = -occupf_L(E_mids,muL,TL)+ occupf_R(E_mids,muR,TR)
    #else:    
    occupdiff = occupf_L(E_mids,muL,TL)- occupf_R(E_mids,muR,TR)
    #print(coeff(E_mids))
    #print(transf)
    #print(occupdiff)
    deltaE = E_mids[1]-E_mids[0]
    integrands = 1/h*coeff(E_mids)*transf*occupdiff*deltaE
    #if np.any(integrands < 0):
    #    current = -100
    #else:
    current = np.sum(integrands)   
    if return_integrands:
        return current, integrands
    return current
    
def slice_maximize_eff(transf, E_mids, muL, TL ,muR, TR,occupf_L = fermi_dist, occupf_R=fermi_dist):
    #electric, integrands = slice_current_integral(transf, E_mids, muL, TL ,muR, TR, deltaE, occupf_L, occupf_R,type = "electric", return_integrands=True)

    heat = slice_current_integral(transf, E_mids, muL, TL ,muR, TR, occupf_L,occupf_R,type = "heat")
    heatR = slice_current_integral(transf, E_mids, muL, TL ,muR, TR,occupf_L,occupf_R,type = "heatR")
    power = heat + heatR#-e*(muL-muR)*electric
    #if heat == 0:
    #    return np.inf
    eff = power/heat
    #if np.abs(eff) > 1:
    #    return np.inf
    return -eff


def slice_maximize_power(transf, E_mids, muL, TL ,muR, TR,occupf_L = fermi_dist):
    electric = slice_current_integral(transf, E_mids, muL, TL ,muR, TR, occupf_L,type = "electric")
    power = -e*(muL-muR)*electric
    #if target_eff > 0:
    #    heat = w_simple_current(E, deltamu, deltaT, type = "heat")
    #    eff = power/heat
    return -power

def slice_eff_constraint(transf, E_mids, muL, TL ,muR, TR, deltaE, target_eff,occupf_L = fermi_dist):
#    electric = slice_current_integral(transf,E_mids, muL, TL ,muR, TR, deltaE, type = "electric")
    heat = slice_current_integral(transf, E_mids, muL, TL ,muR, TR, deltaE, occupf_L,type = "heat")
    heatR = slice_current_integral(transf, E_mids, muL, TL ,muR, TR, deltaE,occupf_L,type = "heatR")
    power = heat + heatR#-e*(muL-muR)*electricpower = -e*(muL-muR)*electric
    eff = power/heat if heat != 0 else -1
    return target_eff - eff

def slice_pow_constraint(transf, E_mids, muL, TL, muR, TR, target_pow, occupf_L = fermi_dist):
#    electric = slice_current_integral(transf,E_mids, muL, TL ,muR, TR, deltaE, type = "electric")
#    power = -e*(muL-muR)*electric
    heat = slice_current_integral(transf, E_mids, muL, TL ,muR, TR, occupf_L,type = "heat")
    heatR = slice_current_integral(transf, E_mids, muL, TL ,muR, TR,occupf_L,type = "heatR")
    power = heat + heatR#-e*(muL-muR)*electricpower = -e*(muL-muR)*electric
    return target_pow-power

def slice_constraint(transf, E_mids, muL, TL, muR, TR, deltaE):
    electric, integrands = slice_current_integral(transf, E_mids, muL, TL ,muR, TR, deltaE,type = "heat", return_integrands=True)
    sgn_integrands = np.sign(integrands[np.argwhere(integrands != 0)]).flatten()-1#np.sign(np.where(integrands != 0)[0])-1
    #print(sgn_integrands)
    print(np.sum(sgn_integrands))
    return np.sum(sgn_integrands)

def pertub_fermi(E, muL, TL, pertub):
    fermi = fermi_dist(E,muL,TL)
    #pertub = fpertub(E)
    dist = fermi + pertub
    if type(dist) == np.ndarray:
        dist[dist < 0] = 0
        dist[dist > 1] = 1
    else:
        if dist < 0:
            dist = 0
        elif dist > 1:
            dist = 1
    return dist

def calc_dJR_dP_dmu(transf, E, muL, TL, muR, TR,occupf_L = fermi_dist, h = 0.01):
    if muR != 0:
        print("only for muR = 0 right now :(")
        return -1,-1
    mus = [muL-h, muL+h]
    [heatL_lowmu, heatL_highmu] = [slice_current_integral(transf, E, mus[0], TL, muR, TR, occupf_L, type = "heat"),slice_current_integral(transf, E, mus[1], TL, muR, TR, occupf_L, type = "heat")]
    [heatR_lowmu, heatR_highmu] = [slice_current_integral(transf, E, mus[0], TL, muR, TR, occupf_L, type = "heatR"),slice_current_integral(transf, E, mus[1], TL, muR, TR, occupf_L, type = "heatR")]
    TLs = [TL-h/2, TL+h/2]
    TRs = [TR+h/2, TR-h/2]
    [heatL_lowT, heatL_highT] = [slice_current_integral(transf, E, muL, TLs[0], muR, TRs[0], occupf_L, type = "heat"),slice_current_integral(transf, E, muL, TLs[1], muR, TRs[1], occupf_L, type = "heat")]
    [heatR_lowT, heatR_highT] = [slice_current_integral(transf, E, muL, TLs[0], muR, TRs[0], occupf_L, type = "heatR"),slice_current_integral(transf, E, muL, TLs[1], muR, TRs[1], occupf_L, type = "heatR")]
    #print(heatR_high)
    #print(heatR_low)
    #print(heatL_high+heatR_high)
    #print(heatL_low+heatR_low)
    #print(heatL_low)
    #print(heatR_low)
    dJ_dmu = (heatL_highmu - heatL_lowmu)/(2*h)
    dP_dmu = (heatR_highmu- heatR_lowmu)/(2*h) + dJ_dmu
    dJ_dT = (heatL_highT - heatL_lowT)/(2*h)
    dP_dT = (heatR_highT- heatR_lowT)/(2*h) + dJ_dT
    
    #dJ_dmu = (heatR_high - heatR_low)/(h)
    #dP_dmu = (heatL_high- heatL_low)/(h) + dJ_dmu
    return dJ_dmu, dP_dmu, dJ_dT, dP_dT

def calc_dJR_dtau_at_P(E, muL, TL,  dP_dtau, Jprim, Pprim,occupf_L=fermi_dist):
    coeff = entropy_coeff(E,muL,TL,occupf_L)
    #print((E/(E+TL*kb*coeff) - Jprim/Pprim))
    E1 = -coeff/(E-coeff) - Jprim/Pprim
    dJR_dtau = (E1)*dP_dtau
    #dJR_dtau = (E/(muL) - Jprim/Pprim)*dP_dtau
    #dJR_dtau = (1-Jprim/Pprim*(1+E/(-E+muL)))*dP_dtau
    return dJR_dtau

def opt_transf_weird(c_eta, r_E, muL, TL, muR, TR, target_p, deltaE, foccup_L):
    mom_etas_diff = lambda E: (E-entropy_coeff(E, muL, TL, foccup_L))/(-entropy_coeff(E, muL, TL, foccup_L)) - c_eta
    #print(mom_etas_diff(np.array([1,2])))
    occupdiff = lambda E: foccup_L(E,muL,TL)- fermi_dist(E,muR,TR)
    occup_zero = fsolve(occupdiff, r_E[0])[0]
    #print((occup_zero))
    #print(occupdiff(occup_zero + deltaE) -  occupdiff(occup_zero - deltaE))
    #if type(occup_zero) == float:
    occup_zero = np.array([occup_zero])
    E_starts = np.linspace(r_E[0], r_E[-1], 10)
    #for E_start in E_starts:
    #    print(fsolve(mom_etas_diff, E_start))
    mom_etas_zero = np.sort(fsolve(mom_etas_diff, E_starts, factor = 0.1))
    #print(mom_etas_zero)
    cutoff = np.argwhere(mom_etas_zero > r_E[-1])
    #print(cutoff)
    if len(cutoff) != 0:
        mom_etas_zero = mom_etas_zero[:cutoff[0,0]]
    print(mom_etas_zero)
    mom_etas_zero = mom_etas_zero[np.isclose(mom_etas_diff(mom_etas_zero), np.zeros_like(mom_etas_zero), atol = 1e-3)]
    mom_etas_zero = np.unique(mom_etas_zero)#.round(decimals=6))
    #print(np.unique(mom_etas_zero.round(decimals=3)))
#print(np.isclose(mom_etas_diff(mom_etas_zero), np.zeros_like(mom_etas_zero), atol = 1e-3))
#print(mom_etas_diff(mom_etas_zero))
    #print(mom_etas_zero)
   
    #if type(mom_etas_zero) == float:
    mom_etas_zero = np.array([mom_etas_zero])

    occup_sign = np.sign(occupdiff(occup_zero + deltaE) -  occupdiff(occup_zero - deltaE))
    mom_etas_sign = np.sign(mom_etas_diff(mom_etas_zero + deltaE) -  mom_etas_diff(mom_etas_zero - deltaE))
    #print(occup_zero)
    #print(mom_etas_zero)
    flip_to_zero = np.sort(np.append(occup_zero[occup_sign < 0], mom_etas_zero[mom_etas_sign < 0]))
    flip_to_one = np.sort(np.append(occup_zero[occup_sign > 0], mom_etas_zero[mom_etas_sign > 0]))

    # The flips should be monotonically increasing by checking first the index 0 in flip_to_zero, then index 0 in flip_to_one, then 1 in flip_to_zero, then 1 in flip_to_one and so on. 
    # The arrays might be differently sized at first, but if we start and and at zero transf, then they should be of equal length
    print(f"before one:{flip_to_one}")
    print(f"before zero:{flip_to_zero}")

    max_len = len(flip_to_zero) if len(flip_to_zero) >= len(flip_to_one) else len(flip_to_one)
    tmp_flip_zero = []
    tmp_flip_one = []
    for i in range(0,max_len - 1):
        check_zero = False
        check_one = False
        while (not check_one or not check_zero):
            if len(flip_to_one) <= i+1:
                break
            if len(flip_to_zero) <= i+1:
                break
            #print("in while loop")
            check_zero = flip_to_zero[i] > flip_to_one[i]
            #print(flip_to_zero[i])
            #print(max_len-2)
            if i != max_len-2:
                check_one = flip_to_one[i+1] > flip_to_zero[i]
            else:
                check_one = True
            #print(check_zero)
            #print(check_one)
            #if check_one and check_zero:
            #    continue
                #if i != 0:
                #    tmp_flip_one.append(flip_to_one[i+1])
                #    tmp_flip_zero.append(flip_to_zero[i])
                #else:
                #    tmp_flip_one.append(flip_to_one[i])
                #    tmp_flip_one.append(flip_to_one[i+1])
                #    tmp_flip_zero.append(flip_to_zero[i])
            if check_zero and not check_one:
                flip_to_one = np.delete(flip_to_one, i+1)
    #            tmp_flip_zero.append(flip_to_zero[i])
            if check_one and not check_zero:
                flip_to_zero = np.delete(flip_to_zero,i)
        
        #print(i)
        #print(flip_to_one)
        #print(flip_to_zero)
        if len(flip_to_one) <= i+1:
            break
        if len(flip_to_zero) <= i+1:
            break
        #if check_one:
                #tmp_flip_zero.append(flip_to_zero[i])
        #    tmp_flip_one.append(flip_to_one[i+1])
    
    if len(flip_to_zero) != len(flip_to_one):
        
        
        if len(flip_to_one) > len(flip_to_zero) and len(flip_to_zero) != 0:
            flip_to_one = flip_to_one[:len(flip_to_zero)]
        if len(flip_to_zero) > len(flip_to_one) and len(flip_to_one) != 0:
            flip_to_zero = flip_to_zero[:len(flip_to_one)]

    #        while len(flip_to_one) != len(flip_to_zero):
    #            check_zero = flip_to_zero[-1] > flip_to_one[-1]
    #            if len(flip_to_zero) > 1:
    #                check_one = flip_to_zero[-2] < flip_to_one[-1]
    #            else:
    #                check_one = True
    #            if not check_one or not check_zero:
    #                flip_to_one = np.delete(flip_to_one, -1)


    #flip_to_one = np.array(tmp_flip_one)
    #flip_to_zero = np.array(tmp_flip_zero)


    print(f"after one:{flip_to_one}")
    print(f"after zero: {flip_to_zero}")
    if len(flip_to_one) != 0 and len(flip_to_zero) != 0:
        if np.min(flip_to_one) < np.min(flip_to_zero):
            start_with_zero = True
        else:
            start_with_zero = False
    else: 
        start_with_zero = True    
    if start_with_zero:
        if len(flip_to_zero) != 0:
            def wrapper_transf(E):
                transf_func = 0
                for flip_i in range(len(flip_to_zero)):
                    #print(np.heaviside(E-flip_to_one[flip_i],1)*np.heaviside(flip_to_zero[flip_i]-E,1))
                    transf_func += np.heaviside(E-flip_to_one[flip_i],1)*np.heaviside(flip_to_zero[flip_i]-E,1)
                return transf_func
            transf_func = lambda E: wrapper_transf(E)#lambda E: np.prod((np.heaviside(E-flip_to_one+1,1)*np.heaviside(flip_to_zero-E,1)))
            #transf_func = lambda E: np.prod((np.heaviside(E - flip_to_one, 1) * np.heaviside(flip_to_zero - E,1)))
        else:
            transf_func = lambda E: np.prod((np.heaviside(E - flip_to_one, 1)))
    else: 
        transf_func = lambda E: np.prod((np.heaviside(E - flip_to_zero, 1) * np.heaviside(flip_to_one - E,1)))
    
    heatL = current_integral(r_E[0], r_E[-1], muL, TL, muR, TR, transf_func, foccup_L, type = "heat")
    heatR = current_integral(r_E[0], r_E[-1], muL, TL, muR, TR, transf_func, foccup_L, type = "heatR")
    #print(heatL)
    #heatL = slice_current_integral(transf_func(r_E), r_E, muL, TL, muR, TR, deltaE, foccup_L, type = "heat")
    #heatR = slice_current_integral(transf_func(r_E), r_E, muL, TL, muR, TR, deltaE, foccup_L, type = "heatR")
    power = heatL+heatR
    #print(power)
    print(c_eta)
    return power - target_p
def opt_transf(c_eta, r_E, muL, TL, muR, TR, target_p, foccup_L):
    mom_etas = (r_E-entropy_coeff(r_E, muL, TL, foccup_L))/(-entropy_coeff(r_E, muL, TL, foccup_L))
    
    occupdiff = foccup_L(r_E,muL,TL)- fermi_dist(r_E,muR,TR)
    #rint(occupdiff)
    transf = np.zeros_like(mom_etas)
    transf[mom_etas > c_eta] = 1
    transf[occupdiff < 0] = 0
    heatL = slice_current_integral(transf, r_E, muL, TL, muR, TR, foccup_L, type = "heat")
    heatR = slice_current_integral(transf, r_E, muL, TL, muR, TR, foccup_L, type = "heatR")
    power = heatL+heatR
    return power - target_p