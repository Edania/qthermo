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

def root_finder_guesses(func, x_low, x_high, tol = 1e-8):
    roots = []
    xs = np.linspace(x_low, x_high, 100000)
    signed = np.sign(func(xs))
    x_inits = xs[np.argwhere(signed[1:] - signed[:-1] != 0).flatten()]
    for x in x_inits:
        res = fsolve(func, x, factor = 0.1)[0]
        if any(np.abs(roots - res) < tol):
            continue
        roots.append(res)
    
    return roots