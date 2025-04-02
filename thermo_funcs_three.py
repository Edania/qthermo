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

#TODO: implement secondary variable optimization 

class two_terminals:
    def __init__(self, E_low, E_high, transf = lambda E: 1, occupf_L = None, occupf_R = None,  muL = 0, TL = 1, muR = 0, TR = 1,
                 coeff_avg = None, coeff_noise = None, coeff_con = None, N = 1, subdivide = False):
        '''
        occupf_L, occupf_R, transf, coeff_avg, coeff_noise, and coeff_con must be functions of E, energy
        '''

        self.E_low = E_low
        self.E_high = E_high
        self.muL = muL
        self.muR = muR
        self.TL = TL
        self.TR = TR
        self.transf = transf
        self.N = N
        self.subdivide = subdivide
        # Standard case is two thermal functions
        if occupf_L == None:
            self.set_fermi_dist_left()
        else:
            self.occupf_L = occupf_L
        if occupf_R == None:
            self.set_fermi_dist_right()
        else:
            self.occupf_R = occupf_R

        # Set standard currents we want to compare. We choose the efficiency -J_R/P with J_R fixed as standard, and the output noise considered
        if coeff_avg == None:
            self.coeff_avg = lambda E: -self.power_coeff(E)
        else:
            self.coeff_avg = coeff_avg
        if coeff_con == None:
            self.coeff_con = self.right_heat_coeff
        else:
            self.coeff_con = coeff_con
        if coeff_noise == None:
            self.coeff_noise = lambda E: -self.power_coeff(E)
        else:
            self.coeff_noise = coeff_noise

        print("Have you defined the coefficients to be positive?")
    
    

    # @property
    # def muL(self):
    #     return self._muL

    # @muL.setter
    # def value(self, mu):
    #     self._value = mu
    #     self.set_fermi_dist_left()

    def set_full_transmission(self):
        self.transf = lambda E: 1

    def set_fermi_dist_left(self):
        self.occupf_L = lambda E: two_terminals.fermi_dist(E, self.muL, self.TL)
    
    def set_fermi_dist_right(self):
        self.occupf_R = lambda E: two_terminals.fermi_dist(E, self.muR, self.TR)

    def set_transmission_noise_opt(self, C, coeff):
        self.transf = self._transmission_noise(C, coeff)

    def set_transmission_avg_opt(self, C, coeff_in, coeff_out):
        self.transf = self._transmission_avg(C, coeff_in, coeff_out)
    
    def entropy_coeff(E, occupf):
        coeff = -kb*np.log(occupf(E)/(1-occupf(E)))
        return coeff

    def left_entropy_coeff(self,E):
        return self.entropy_coeff(E, self.occupf_L)

    def right_entropy_coeff(self,E):
        return self.entropy_coeff(E, self.occupf_R)

    def left_heat_coeff(self,E):
        return E-self.muL

    def right_heat_coeff(self,E):
        return -E+self.muR

    def left_particle_coeff(self,E):
        return 1

    def right_particle_coeff(self,E):
        return -1

    def left_electric_coeff(self,E):
        return self.muL

    def right_electric_coeff(self,E):
        return -self.muR
    
    def power_coeff(self,E):
        return (self.muR-self.muL)

    def left_energy_coeff(self,E):
        return E

    def right_energy_coeff(self, E):
        return -E
    
    def left_noneq_free_coeff(self,E):
        return -E - self.TR*self.left_entropy_coeff

    def calc_left_particle_current(self):
        return self._current_integral(self.left_particle_coeff)

    def calc_right_particle_current(self):
        return self._current_integral(self.right_particle_coeff)

    def calc_left_energy_current(self):
        return self._current_integral(self.left_energy_coeff)
    
    def calc_right_energy_current(self):
        return self._current_integral(self.right_energy_coeff)
    
    def calc_left_heat_current(self):
        return self._current_integral(self.left_heat_coeff)
    
    def calc_right_heat_current(self):
        return self._current_integral(self.right_heat_coeff)
    
    def calc_left_entropy_current(self):
        return self._current_integral(self.left_entropy_coeff)
    
    def calc_right_entropy_current(self):
        return self._current_integral(self.right_entropy_coeff)
    
    def calc_right_entropy_current(self):
        return self._current_integral(self.left_noneq_free_coeff)
    

    def _current_integral(self, coeff, transf_in = None):
        if transf_in == None:
            #print("In wrong place")
            transf = self.transf
        else:
            transf = transf_in     
        integrand = lambda E: self.N*1/h*coeff(E)*transf(E)*(self.occupf_L(E)- self.occupf_R(E))
        
        if self.subdivide:
            coarse_Es = np.linspace(self.E_low, self.E_high,10000)
            coarse_E_lows = coarse_Es[np.where(transf(coarse_Es[1:])- transf(coarse_Es[:-1]) == 1)]
            #print(coarse_E_lows)
            coarse_E_highs = coarse_Es[np.where(transf(coarse_Es[1:])- transf(coarse_Es[:-1]) == -1)]
            
        # if len(coarse_E_highs) == 0 and len(coarse_E_lows) == 0:
        #     coarse_Es = np.linspace(np.sign(self.E_low)*1.5*self.E_low, 0.2*self.E_high,10000)
        #     coarse_E_lows = coarse_Es[np.where(transf(coarse_Es[1:])- transf(coarse_Es[:-1]) == 1)]
        #     #print(coarse_E_lows)
        #     coarse_E_highs = coarse_Es[np.where(transf(coarse_Es[1:])- transf(coarse_Es[:-1]) == -1)]
                
            #print(len(coarse_E_highs))
            #print(coarse_E_highs)
            #g = 0.01
            E_lows = 0.8*coarse_E_lows
            E_highs = 1.2*coarse_E_highs
            #E_lows = [fsolve(lambda E: transf(E-g) + transf(E+g) - 1 , 0.9*coarse_E_low, factor=1) for coarse_E_low in coarse_E_lows]
            #E_highs = [fsolve(lambda E: transf(E-g) + transf(E+g) + 1, 0.9*coarse_E_high,factor=1) for coarse_E_high in coarse_E_highs]
            #print(E_lows)
            #print(E_highs)
            current = 0
            for i in range(len(E_lows)):
                current_i, err = integrate.quad(integrand, E_lows[i], E_highs[i], args=())
                current += current_i
        else:
            current, err = integrate.quad(integrand, self.E_low, self.E_high, args=())
        return current

    def noise_cont(self, coeff):
        thermal = lambda E: self.N*coeff(E)**2*self.transf(E)*(self.occupf_L(E)*(1-self.occupf_L(E))+ self.occupf_R(E)*(1-self.occupf_R(E)))
        shot = lambda E: coeff(E)**2 * self.N**2*self.transf(E)*(1-self.transf(E))*(self.occupf_L(E)+self.occupf_R(E))**2
        integrand = lambda E: thermal(E) + shot(E)
        current, err = integrate.quad(integrand, self.E_low, self.E_high, args=())
        return current
    
    def _transmission_avg(self, C, coeff_x, coeff_y):
        x_integrands = lambda E: coeff_x(E)*(self.occupf_L(E)- self.occupf_R(E))
        y_integrands = lambda E: coeff_y(E)*(self.occupf_L(E)- self.occupf_R(E))
        #transf = lambda E: np.heaviside(coeff_out(E)/coeff_in(E) - C, 0)*np.heaviside(out_integrands(E)*in_integrands(E), 0)+np.heaviside(-coeff_out(E)/coeff_in(E) - C, 0)*np.heaviside(out_integrands(E)*in_integrands(E), 0)
        transf = lambda E: np.heaviside(coeff_x(E)/coeff_y(E) - C, 0.5)*np.heaviside(y_integrands(E),0.5)* np.heaviside(x_integrands(E), 0.5)
        #print("C",C)
        #print("Con current", self._current_integral(coeff_x,transf))
        return transf

    def _transmission_noise(self, C, coeff):
        integrands = lambda E: coeff(E)*(self.occupf_L(E)- self.occupf_R(E))
        comp = lambda E: (self.occupf_L(E) - self.occupf_R(E))/(coeff(E)*(self.occupf_L(E) *(1-self.occupf_L(E)) + self.occupf_R(E)*(1-self.occupf_R(E))))
        transf = lambda E: np.heaviside(comp(E) - C, 0.5)*np.heaviside(integrands(E), 0.5) \
                #+np.heaviside(- (comp - C), 0)*np.heaviside(-integrands, 0)
        return transf

    def optimize_for_avg(self, C_init,target, fixed = "nominator"):
        '''
        Make sure coeffs are defined such that positive contributions to currents are desirable and negative suppressed
        '''


        if fixed == "nominator":
            coeff_nom = self.coeff_con
            coeff_denom = self.coeff_avg
            
        else:
            coeff_denom = self.coeff_con
            coeff_nom = self.coeff_avg
            
        transf = lambda C: self._transmission_avg(C, coeff_nom, coeff_denom)
        fixed_current_eq = lambda C: self._current_integral(self.coeff_con,transf(C)) - target

        res = fsolve(fixed_current_eq,C_init, factor = 1)
        self.transf = self._transmission_avg(res[0], coeff_nom, coeff_denom)
        return res[0]

    def optimize_for_noise(self,C_init, target):
        transf = lambda C: self._transmission_noise(C, self.coeff_noise)
        fixed_current_eq = lambda C: self._current_integral(self.coeff_noise, transf(C)) - target
        res = fsolve(fixed_current_eq,C_init)
        self.transf = self._transmission_noise(res[0],self.coeff_noise)
        return res[0]

    def carnot(self):
        return (1-self.TR/self.TL)

    def cop(self):
        return self.TR/(self.TL-self.TR)

    def pmax(self):
        A = 0.0321
        p = A * np.pi**2/h * N * kb**2 *(self.TL-self.TR)**2
        return p

    def jRmax(self):
        coeff = lambda E:-E+self.muR
        transf = lambda E: np.heaviside(coeff(E)*(self.occupf_L(E) - self.occupf_R(E)),0)
        current = self._current_integral(coeff, transf)
        return current, transf

    def E_max(self):
        if self.TL-self.TR == 0:
            return 0
        #print(((deltaT+T)*mu - T*(deltamu+mu))/deltaT)
        return (self.TL*self.muR - self.TR*self.muL)/(self.TL-self.TR)


    def fermi_dist(E, mu, T):
        f_dist = 1/(1+np.exp((E-mu)/(T*kb)))
        return f_dist
    
    
    ## SLICED FUNCTIONS ##
    
    def slice_current_integral(self, coeff, transf_in = None, return_integrands = False, n_Es = 100):
        E_mids = np.linspace(self.E_low, self.E_high, n_Es)
        occupdiff = self.occupf_L(E_mids)- self.occupf_R(E_mids)
        deltaE = E_mids[1]-E_mids[0]

        if transf_in == None:
            transf = self.transf
        integrands = 1/h*coeff(E_mids)*transf*occupdiff*deltaE

        current = np.trapz(integrands, E_mids)
        if return_integrands:
            return current, integrands
        return current
    
    def slice_constraint(self, coeff, target, transf = None):
        current = self.slice_current_integral(coeff, transf)
        return target-current
    
    def slice_maximize_eff(self, coeff_in, coeff_out,transf= None):
        input = self.slice_current_integral(coeff_in, transf)
        output = self.slice_current_integral(coeff_out,transf)
        eff = output/input
        return -eff


    

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


