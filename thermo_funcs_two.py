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


class two_terminals:
    def __init__(self, E_low, E_high, transf = None, occupf_L = None, occupf_R = None,  muL = 0, TL = 1, muR = 0, TR = 1):
        '''
        occupf_L, occupf_R and transf must be functions of E, energy
        '''

        self.E_low = E_low
        self.E_high = E_high
        self.muL = muL
        self.muR = muR
        self.TL = TL
        self.TR = TR
        if transf == None:
            self.set_full_transmission()
        else:
            self.transf = transf
        if occupf_L == None:
            self.set_fermi_dist_left()
        else:
            self.occupf_L = occupf_L
        if occupf_R == None:
            self.set_fermi_dist_right()
        else:
            self.occupf_R = occupf_R
    
    def set_full_transmission(self):
        self.transf = lambda E: 1

    def set_fermi_dist_left(self):
        self.occupf_L = lambda E: fermi_dist(E, self.muL, self.TL)
    
    def set_fermi_dist_right(self):
        self.occupf_R = lambda E: fermi_dist(E, self.muR, self.TR)

    def _current_integral(self, coeff):
        integrand = lambda E: 1/h*coeff(E)*self.transf(E)*(self.occupf_L(E)- self.occupf_R(E))
        current, err = integrate.quad(integrand, self.E_low, self.E_high, args=())
        return current

    def noise_cont(self, coeff):
        thermal = lambda E: coeff(E)**2*self.transf(E)*(self.occupf_L(E)*(1-self.occupf_L(E))+ self.occupf_R(E)*(1-self.occupf_R(E)))
        shot = lambda E: coeff(E)**2 * self.transf(E)*(1-self.transf(E))*(self.occupf_L(E)+self.occupf_R(E))**2
        integrand = lambda E: thermal(E) + shot(E)
        current, err = integrate.quad(integrand, self.E_low, self.E_high, args=())
        return current
    
    def _transmission_avg(self, C, coeff_in, coeff_out):
        in_integrands = lambda E: coeff_in(E)*(self.occupf_L(E)- self.occupf_R(E))
        out_integrands = lambda E: coeff_out(E)*(self.occupf_L(E)- self.occupf_R(E))
        #transf = lambda E: np.heaviside(coeff_out(E)/coeff_in(E) - C, 0)*np.heaviside(out_integrands(E)*in_integrands(E), 0)+np.heaviside(-coeff_out(E)/coeff_in(E) - C, 0)*np.heaviside(out_integrands(E)*in_integrands(E), 0)
        transf = lambda E: np.heaviside(coeff_out(E)/coeff_in(E) - C, 0)*np.heaviside(out_integrands(E),0)* np.heaviside(in_integrands(E), 0)
        return transf

    def _transmission_noise(self, C, coeff):
        integrands = lambda E: coeff(E)*(self.occupf_L(E)- self.occupf_R(E))
        comp = lambda E: (self.occupf_L(E) - self.occupf_R(E))/(coeff(E)*(self.occupf_L(E) *(1-self.occupf_L(E)) + self.occupf_R(E)*(1-self.occupf_R(E))))
        transf = lambda E: np.heaviside(comp(E) - C, 0)*np.heaviside(integrands, 0) \
                #+np.heaviside(- (comp - C), 0)*np.heaviside(-integrands, 0)
        return transf

    def general_opt_avg(self, C_init,target,E_low, E_high,occupf_L, occupf_R, coeff_in, coeff_out, fixed = "out"):
        '''
        Make sure coeffs are defined such that positive contributions to currents are desirable and negative suppressed
        '''
        transf = lambda C: self._transmission_avg(C, coeff_in, coeff_out)
        if fixed == "out":
            coeff = coeff_out
        else:
            coeff = coeff_in

        fixed_current_eq = lambda C: current_integral(E_low, E_high,occupf_L, occupf_R,coeff_out,transf(C)) - target

        res = fsolve(fixed_current_eq,C_init, factor = 0.1)

        return res[0]

    def general_opt_noise(C_init, target,E_low, E_high,occupf_L, occupf_R, coeff):
        transf = lambda C,E: transmission_noise(C, E, occupf_L, occupf_R, coeff)
        fixed_current_eq = lambda C: current_integral(E_low, E_high,occupf_L, occupf_R,coeff,lambda E:transf(C,E)) - target
        res = fsolve(fixed_current_eq,C_init)
        return res[0]






def fermi_dist(E, mu, T):
    f_dist = 1/(1+np.exp((E-mu)/(T*kb)))
    return f_dist

def current_integral_old(E_low, E_high, muL, TL, muR, TR, transf, occupf_L = fermi_dist, occupf_R = fermi_dist, type = "electric"):
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
def current_integral(E_low,E_high,occupf_L, occupf_R, coeff, transf):
    integrand = lambda E: 1/h*coeff(E)*transf(E)*(occupf_L(E)- occupf_R(E))
    current, err = integrate.quad(integrand, E_low, E_high, args=())
    return current

def noise_cont(E_low,E_high,occupf_L, occupf_R, coeff, transf):
    thermal = lambda E: coeff(E)**2*transf(E)*(occupf_L(E)*(1-occupf_L(E))+ occupf_R(E)*(1-occupf_R(E)))
    shot = lambda E: coeff(E)**2 * transf(E)*(1-transf(E))*(occupf_L(E)+occupf_R(E))**2
    integrand = lambda E: thermal(E) + shot(E)
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

def transmission_avg(C, E, occupf_L, occupf_R, coeff_in, coeff_out):
    in_integrands = coeff_in(E)*(occupf_L(E)- occupf_R(E))
    out_integrands = coeff_out(E)*(occupf_L(E)- occupf_R(E))
    transf = np.heaviside(coeff_out(E)/coeff_in(E) - C, 0)*np.heaviside(out_integrands*in_integrands, 0)+np.heaviside(-coeff_out(E)/coeff_in(E) - C, 0)*np.heaviside(out_integrands*in_integrands, 0)
    return transf

def transmission_noise(C, E, occupf_L, occupf_R, coeff):
    integrands = coeff(E)*(occupf_L(E)- occupf_R(E))
    comp = (occupf_L(E) - occupf_R(E))/(coeff(E)*(occupf_L(E) *(1-occupf_L(E)) + occupf_R(E)*(1-occupf_R(E))))
    transf = np.heaviside(comp - C, 0)*np.heaviside(integrands, 0) \
            #+np.heaviside(- (comp - C), 0)*np.heaviside(-integrands, 0)
    return transf

def general_opt_avg(C_init,target,E_low, E_high,occupf_L, occupf_R, coeff_in, coeff_out, fixed = "out"):
    '''
    Make sure coeffs are defined such that positive contributions to currents are desirable and negative suppressed
    '''
    transf = lambda C,E: transmission_avg(C, E, occupf_L, occupf_R, coeff_in, coeff_out)
    if fixed == "out":
        coeff = coeff_out
    else:
        coeff = coeff_in

    fixed_current_eq = lambda C: current_integral(E_low, E_high,occupf_L, occupf_R,coeff_out,lambda E:transf(C,E)) - target

    res = fsolve(fixed_current_eq,C_init, factor = 0.1)

    return res[0]

def general_opt_noise(C_init, target,E_low, E_high,occupf_L, occupf_R, coeff):
    transf = lambda C,E: transmission_noise(C, E, occupf_L, occupf_R, coeff)
    fixed_current_eq = lambda C: current_integral(E_low, E_high,occupf_L, occupf_R,coeff,lambda E:transf(C,E)) - target
    res = fsolve(fixed_current_eq,C_init)
    return res[0]

