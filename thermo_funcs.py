#Define functions
import numpy as np
import copy
import utilities as ut
 
from scipy import integrate
from scipy.optimize import minimize, fsolve
from inspect import signature

## GLOBAL CONSTANTS ##
h = 1
kb = 1
e = 1
# N = 1
## FUNCTION DEFINITIONS ##

#TODO: implement secondary variable optimization 

class two_terminals:

    def __init__(self, E_low, E_high, transf = lambda E: 1, occupf_L = None, diff_occupf_L = None,occupf_R = None,  muL = 0, TL = 1, muR = 0, TR = 1,
                 coeff_avg = None, coeff_noise = None, coeff_con = None, N = 1, subdivide = False, debug = False):
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
        self.debug = debug
        # Standard case is two thermal functions
        if occupf_L == None:
            self.set_fermi_dist_left()
        else:
            self.set_occupf_L(occupf_L)
        self.diff_occupf_L = diff_occupf_L
        
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
            self.coeff_noise = self.right_heat_coeff
        else:
            self.coeff_noise = coeff_noise


        self.set_occup_roots()
        #print("Have you defined the coefficients to be positive?")
    

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

    def set_occupf_L(self, occupf_L, z = 0):
        if len(signature(occupf_L).parameters) == 2:
            self.z = z#signature(occupf_L).parameters['z'].default
            self.wrapper_occupf_L = lambda E, z: occupf_L(E,z)
            self.occupf_L = lambda E: occupf_L(E,self.z)
        else:
            self.occupf_L = occupf_L

    def update_occupf_L(self):
        
        self.occupf_L = lambda E: self.wrapper_occupf_L(E, self.z)

    def entropy_coeff(E, occupf):
        coeff = kb*np.log(occupf(E)/(1-occupf(E)))
        return coeff

    def left_entropy_coeff(self,E):
        return two_terminals.entropy_coeff(E, self.occupf_L)

    def right_entropy_coeff(self,E):
        return  -two_terminals.entropy_coeff(E, self.occupf_R)

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
        return E - self.TR*self.left_entropy_coeff(E)

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
    
    def get_efficiency(self):
        return self._current_integral(self.coeff_con)/self._current_integral(self.coeff_avg)


    def find_occup_roots(self, step = 0.1, tol = 1e-3):
        occupdiff = lambda E: self.occupf_L(E) - self.occupf_R(E)
        roots = ut.root_finder(occupdiff, self.E_low, self.E_high, step, tol)
        return roots
    
    def set_occup_roots(self, step = 0.1, tol = 1e-3):
        self.occuproots = self.find_occup_roots(step, tol)

    # def find_constraint_roots(self, step = 0.1, tol = 1e-3):
        
        # return roots
    def constrained_current_max(self, step = 0.1, tol = 1e-3):
        func = lambda E:self.coeff_con(E)*(self.occupf_L(E) - self.occupf_R(E))
        roots = np.array(ut.root_finder(func, self.E_low, self.E_high, step, tol))
        roots = np.sort(roots)
        # Assert that we truly only have roots
        roots[func(roots) < 1e-6]
        transf = lambda E: np.heaviside(func(E),0)

        # h = 0.001
        # first = func(roots[0]+ h) - func(roots[0] - h)
        # if self.debug:
        #     print("Roots: ", roots)
        #     print("Changes: ", func(roots+ h) - func(roots - h))
        # #print(first)
        # transf_list = []
        # if first > 0:
        #     root_end = len(roots) - 1 if len(roots) % 2 == 0 else len(roots) - 2
            
        #     for i in range(0,root_end,2):
        #         print(i) 
        #         #root_one = copy.copy(roots[i])
        #         #root_two = copy.copy(roots[i+1])
        #         temp_transf = lambda E: np.heaviside(E - roots[i], 0)*np.heaviside(roots[i+1]-E, 0)
        #         print(temp_transf(0.2))
        #         transf_list.append(temp_transf)
        #     if len(roots) % 2 != 0:
                
        #         transf_list.append(lambda E: np.heaviside(roots[-1] - E,0))
        # else:
        #     transf_list.append(lambda E: np.heaviside(E - roots[0], 0))
        #     root_end = len(roots) - 1 if len(roots) % 2 != 0 else len(roots) - 2
        #     for i in range(1,root_end,2): 
        #         temp_transf = lambda E: np.heaviside(E - roots[i], 0)*np.heaviside(roots[i+1]-E, 0)
        #         transf_list.append(temp_transf)
        #     if len(roots) % 2 != 0:
        #         transf_list.append(lambda E: np.heaviside(roots[-1] - E,0))
        # print(transf_list[0](0.2), transf_list[1](0.2))
        # transf = lambda E: sum([x(E) for x in transf_list])
        
        #if len(roots) % 2 == 0:
        #print(transf(0.2))
        jmax = self._current_integral(self.coeff_con, transf)
        #print(transf(np.linspace(-0.5,1)))
        return jmax, transf, roots

    def adjust_limits(self):
        jmax, transf, roots = self.constrained_current_max()
        lowest = np.min(roots)
        highest = np.max(roots)
        self.E_min = lowest - 0.5*lowest
        self.E_max = highest + 0.5*highest

    #TODO: Fix subdivide. Just change E_low and E_high according to the limits of the cooling regime for now.
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
                current_i, err = integrate.quad(integrand, E_lows[i], E_highs[i], args=(), points=self.occuproots)
                current += current_i
        else:
            current, err = integrate.quad(integrand, self.E_low, self.E_high, args=(), points=self.occuproots, limit = 100)
        return current

    def noise_cont(self, coeff, transf = None):
        if transf == None:
            transf = self.transf
        thermal = lambda E: self.N*coeff(E)**2*transf(E)*(self.occupf_L(E)*(1-self.occupf_L(E))+ self.occupf_R(E)*(1-self.occupf_R(E)))
        shot = lambda E: coeff(E)**2 * self.N**2*transf(E)*(1-transf(E))*(self.occupf_L(E)+self.occupf_R(E))**2
        integrand = lambda E: thermal(E) + shot(E)
        current, err = integrate.quad(integrand, self.E_low, self.E_high, args=(), points = self.occuproots, limit = 100)
        return current
    
    def _transmission_avg(self, C, coeff_x, coeff_y):
        x_integrands = lambda E: coeff_x(E)*(self.occupf_L(E)- self.occupf_R(E))
        y_integrands = lambda E: coeff_y(E)*(self.occupf_L(E)- self.occupf_R(E))
        #transf = lambda E: np.heaviside(coeff_out(E)/coeff_in(E) - C, 0)*np.heaviside(out_integrands(E)*in_integrands(E), 0)+np.heaviside(-coeff_out(E)/coeff_in(E) - C, 0)*np.heaviside(out_integrands(E)*in_integrands(E), 0)
        transf = lambda E: np.heaviside(-(coeff_y(E) - C*coeff_x(E))*(self.occupf_L(E) - self.occupf_R(E)), 0.5)
        #transf = lambda E: np.heaviside(coeff_x(E)/coeff_y(E) - C, 0.5)*np.heaviside(y_integrands(E),0.5)* np.heaviside(x_integrands(E), 0.5)
        # if self.debug:
        #     print("C",C)
            # print("Con current", self._current_integral(coeff_x,transf))
        return transf
    def optimize_for_avg(self,target, C_init = None, fixed = "nominator", secondary = False):
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

        if C_init == None:
            temp_Es = np.linspace(self.E_low, self.E_high, 100)
            x_integrands = lambda E: coeff_nom(E)*(self.occupf_L(E)- self.occupf_R(E))
            y_integrands = lambda E: coeff_denom(E)*(self.occupf_L(E)- self.occupf_R(E))
            func_C = coeff_nom(temp_Es)/coeff_denom(temp_Es)*np.heaviside(y_integrands(temp_Es),0.5)* np.heaviside(x_integrands(temp_Es), 0.5)
            C_max = np.max(func_C)
            C_init = 0.9*C_max
            if self.debug:
                print("C_init: ", C_init)
            
            test_current = self._current_integral(self.coeff_con, transf(C_init))
            # while(test_current == 0):
            #     if self.debug:
            #         print(("Entering while loop"))
            #     C_init /= 2
                # test_current = self._current_integral(self.coeff_con, transf(C_init))
            if self.debug:
                print("C_init: ", C_init)

        if secondary:
            def fixed_current_eq(thetas, h = 0.01):
                C = thetas[0]
                z = thetas[1]
                self.z = z
                self.update_occupf_L()
                self.set_occup_roots()
                current = self._current_integral(self.coeff_con,transf(C))
                print("Current: ", current)
                print("C: ",C)
                if current == 0:
                    C = 0.5*C               
                
                
                self.z = z + h
                self.update_occupf_L()
                self.set_occup_roots()

                avg_high = self._current_integral(self.coeff_avg,transf(C))
                con_high = self._current_integral(self.coeff_con,transf(C))

                self.z = z - h
                self.update_occupf_L()
                self.set_occup_roots()

                avg_low = self._current_integral(self.coeff_avg,transf(C))
                con_low = self._current_integral(self.coeff_con,transf(C))


                if self.debug:
                    print(avg_high-avg_low)
                    print(con_high-con_low)
                    print("Diff div: ",(avg_high-avg_low)/(con_high-con_low))

                return [self._current_integral(self.coeff_con,transf(C)) - target, (avg_high-avg_low)/(con_high-con_low) - C]
            res = fsolve(fixed_current_eq,[C_init, self.z], factor = 0.1, xtol = 1e-6)
            self.z = res[1]
            self.update_occupf_L()
            self.set_occup_roots()
            
            self.transf = self._transmission_avg(res[0], coeff_nom, coeff_denom)
            return res


        else:
            fixed_current_eq = lambda C: self._current_integral(self.coeff_con,transf(C)) - target
            res = fsolve(fixed_current_eq,C_init, factor = 0.1, xtol = 1e-6)
            self.transf = self._transmission_avg(res[0], coeff_nom, coeff_denom)
            return res[0]

    
    def _noise_condition(self):
        div = lambda E: self.coeff_noise(E)**2*(self.occupf_L(E) *(1-self.occupf_L(E)) + self.occupf_R(E)*(1-self.occupf_R(E)))
        return lambda E: self.coeff_con(E)*(self.occupf_L(E) - self.occupf_R(E))/div(E)

    def _transmission_noise(self, C):
        integrands = lambda E: self.coeff_con(E)*(self.occupf_L(E)- self.occupf_R(E))
        comp = self._noise_condition()
        transf = lambda E: np.heaviside(comp(E) - C, 0)*np.heaviside(integrands(E), 0) \
                #+np.heaviside(- (comp - C), 0)*np.heaviside(-integrands, 0)
        if self.debug:
            print("C",C)       
            print("Con current", self._current_integral(self.coeff_con,transf))
        
        return transf

    def optimize_for_noise(self, target, C_init = None, secondary = False):
        
        transf = lambda C: self._transmission_noise(C)
        if C_init == None:
            temp_Es = np.linspace(self.E_low, self.E_high, 100)
            con_integrands = lambda E: self.coeff_con(E)*(self.occupf_L(E)- self.occupf_R(E))
            func_C = self._noise_condition()(temp_Es)*np.heaviside(con_integrands(temp_Es),0)
            func_C = func_C[~np.isnan(func_C)]
            C_max = np.max(func_C)
            C_init = 0.9*C_max
            if self.debug:
                print("C_init: ", C_init)

            test_current = self._current_integral(self.coeff_con, transf(C_init))
            while(test_current == 0):
                if self.debug:
                    print(("Entering while loop"))
                C_init /= 2
                test_current = self._current_integral(self.coeff_con, transf(C_init))
            if self.debug:
                print("C_init: ", C_init)
        
        fixed_current_eq = lambda C: self._current_integral(self.coeff_noise, transf(C)) - target
        res = fsolve(fixed_current_eq,C_init, factor = 0.1, xtol=1e-6)
        self.transf = self._transmission_noise(res[0])
        return res[0]


    def _product_condition(self,thetas):
        div = lambda E: self.coeff_con(E)*(self.occupf_L(E)-self.occupf_R(E))
        term_one = lambda E: -thetas[0]*self.coeff_noise(E)**2*(self.occupf_L(E)*(1-self.occupf_L(E))+self.occupf_R(E)*(1-self.occupf_R(E)))#/div(E)
        term_two = lambda E: -thetas[1]*self.coeff_avg(E)*(self.occupf_L(E)-self.occupf_R(E))#/div(E)
        term_three = lambda E: thetas[2]*div(E)
        opt_func = lambda E: term_one(E)+term_two(E)+term_three(E)
        return opt_func

    def _transmission_product(self,thetas):
        con_integrands = lambda E: self.coeff_con(E)*(self.occupf_L(E)- self.occupf_R(E))
        avg_integrands = lambda E: self.coeff_avg(E)*(self.occupf_L(E)- self.occupf_R(E))
            
        #print(-thetas[0]*thetas[1] + max_prod)
        transf = lambda E: np.heaviside(self._product_condition(thetas)(E), 0)*np.heaviside(avg_integrands(E),0)*np.heaviside(con_integrands(E),0)
        return transf
        
    def optimize_for_product(self, target, thetas_init = None, alpha = 0.5, secondary = False):
        if thetas_init == None:
            max_transf = lambda E: np.heaviside(self.coeff_con(E)*(self.occupf_L(E) - self.occupf_R(E)),0)
            self.transf = max_transf
            max_noise = self.noise_cont(self.coeff_noise)
            max_avg = self._current_integral(self.coeff_avg)
            temp_Es = np.linspace(self.E_low, self.E_high, 100)
            con_integrands = lambda E: self.coeff_con(E)*(self.occupf_L(E)- self.occupf_R(E))
            avg_integrands = lambda E: self.coeff_avg(E)*(self.occupf_L(E)- self.occupf_R(E))
            div = self.coeff_con(temp_Es)*(self.occupf_L(temp_Es)-self.occupf_R(temp_Es))
            zero_index = np.argwhere(div == 0)
            div = np.delete(div, zero_index)
            temp_Es = np.delete (temp_Es, zero_index)
            func_C = self._product_condition([max_avg/2, max_noise/2, 0])(temp_Es)/(div)*np.heaviside(con_integrands(temp_Es),0)*np.heaviside(avg_integrands(temp_Es),0)
            func_C = func_C[~np.isnan(func_C)]
            C_init = -np.min(func_C)/10
            thetas_init = [max_avg/10, max_noise/10, C_init]
            #print(C_max)
            test_current = self._current_integral(self.coeff_con, self._transmission_product(thetas_init))
            while(test_current == 0):
                if self.debug:
                    print(("Entering while loop"))
                C_init *= 2
                test_current = self._current_integral(self.coeff_con, self._transmission_product(thetas_init))
            if self.debug:
                print("C_init: ", C_init)
            thetas_init = [max_avg/10, max_noise/10, C_init]
            #thetas_init = [max_avg/2, max_noise/2, 0.1]
            if self.debug:
                print("Thetas init: ", thetas_init)

        def opt_func(thetas):
            transf = self._transmission_product(thetas)
            nois = self.noise_cont(self.coeff_noise, transf)
            avg = self._current_integral(self.coeff_avg, transf)
            con = self._current_integral(self.coeff_con, transf)
            if self.debug: 
                print("Thetas: ", thetas)
                print("opt func: ", avg - thetas[0], nois - thetas[1], con-target)
            return avg - thetas[0], nois - thetas[1], con-target
        
        res = fsolve(opt_func, thetas_init, factor=1, xtol=1e-6)
        self.transf = self._transmission_product(res)
        return res

    def carnot(self):
        return (1-self.TR/self.TL)

    def cop(self):
        return self.TR/(self.TL-self.TR)

    def pmax(self):
        A = 0.0321
        p = A * np.pi**2/h * self.N * kb**2 *(self.TL-self.TR)**2
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


