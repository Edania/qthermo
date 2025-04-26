import numpy as np
from scipy.optimize import fsolve

#TODO: fix this :)
def root_finder(func, x_min, x_max, step, tol):
    x = x_min
    roots = []
    while(x <= x_max):
        res = fsolve(func, x, factor = 0.1)[0]
        # print("Root: ",res)
        # print("All roots: ",roots)
        # print("Current search: ",E_search)
        
        x += step
        if any(np.abs(roots - res) < tol):
            continue
        roots.append(res)
        
    return roots

def root_finder_guesses(func, x_inits, tol = 1e-3):
    roots = []
    for x in x_inits:
        res = fsolve(func, x, factor = 0.1)[0]
        if any(np.abs(roots - res) < tol):
            continue
        roots.append(res)
    
    return roots