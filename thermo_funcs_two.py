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

def entropy_coeff(E, occupf_L):
    coeff = -kb*np.log(occupf_L(E)/(1-occupf_L(E)))
    return coeff

def heat_L_coeff(E,muL):
    return E-muL


def slice_current_integral(transf,E_mids,occupf_L, occupf_R, coeff, return_integrands = False):

    occupdiff = occupf_L(E_mids)- occupf_R(E_mids)
    deltaE = E_mids[1]-E_mids[0]
    integrands = 1/h*coeff(E_mids)*transf*occupdiff*deltaE

    current = np.sum(integrands)   
    if return_integrands:
        return current, integrands
    return current
    
def slice_maximize_eff(transf, E_mids,occupf_L, occupf_R, coeff_in, coeff_out):
    input = slice_current_integral(transf, E_mids,occupf_L, occupf_R, coeff_in)
    output = slice_current_integral(transf, E_mids,occupf_L, occupf_R, coeff_out)
    eff = output/input
    return -eff

def slice_eff_constraint(transf, E_mids, muL, TL ,muR, TR, deltaE, target_eff,occupf_L = fermi_dist):
#    electric = slice_current_integral(transf,E_mids, muL, TL ,muR, TR, deltaE, type = "electric")
    heat = slice_current_integral(transf, E_mids, muL, TL ,muR, TR, deltaE, occupf_L,type = "heat")
    heatR = slice_current_integral(transf, E_mids, muL, TL ,muR, TR, deltaE,occupf_L,type = "heatR")
    power = heat + heatR#-e*(muL-muR)*electricpower = -e*(muL-muR)*electric
    eff = power/heat if heat != 0 else -1
    return target_eff - eff

def slice_pow_constraint(transf, E_mids,occupf_L, occupf_R, coeff, target):
    current = slice_current_integral(transf, E_mids,occupf_L, occupf_R, coeff)
    return target-current


def pertub_dist(E, dist, pertub):
    fermi = dist(E)
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

def transmission_avg_avg(C, E_mids, occupf_L, occupf_R, coeff_in, coeff_out):
    in_integrands = coeff_in(E_mids)*(occupf_L(E_mids)- occupf_R(E_mids))
    out_integrands = coeff_out(E_mids)*(occupf_L(E_mids)- occupf_R(E_mids))
    transf = np.heaviside(coeff_out(E_mids)/coeff_in(E_mids) - C, 0)*np.heaviside(out_integrands, 0)*np.heaviside(in_integrands, 0)
    return transf

def transmission_avg_noise(C, E_mids, occupf_L, occupf_R, coeff):
    integrands = coeff(E_mids)*(occupf_L(E_mids)- occupf_R(E_mids))
    transf = np.heaviside((occupf_L(E_mids) - occupf_R(E_mids))/(coeff(E_mids)*(occupf_L(E_mids) *(1-occupf_L(E_mids)) + occupf_R(E_mids)*(1-occupf_R(E_mids)))) - C, 0)*np.heaviside(integrands, 0)
    return transf

def general_opt_avg_avg(C_init,target,E_mids,occupf_L, occupf_R, coeff_in, coeff_out, fixed = "out"):
    '''
    Make sure coeffs are defined such that positive contributions to currents are desirable and negative suppressed
    '''
    transf = lambda C: transmission_avg_avg(C, E_mids, occupf_L, occupf_R, coeff_in, coeff_out)
    if fixed == "out":
        coeff = coeff_out
    else:
        coeff = coeff_in

    fixed_current_eq = lambda C: slice_current_integral(transf(C),E_mids,occupf_L, occupf_R,coeff_out) - target

    res = fsolve(fixed_current_eq,C_init, factor = 0.1)

    return res[0]

def general_opt_avg_noise(C_init, target,E_mids,occupf_L, occupf_R, coeff):
    transf = lambda C: transmission_avg_noise(C, E_mids, occupf_L, occupf_R, coeff)
    fixed_current_eq = lambda C: slice_current_integral(transf(C),E_mids,occupf_L, occupf_R,coeff) - target
    res = fsolve(fixed_current_eq,C_init)
    return res[0]

