import numpy as np
from scipy.optimize import fsolve


def root_finder(func, x_min, x_max, step, tol):
    x = x_min
    roots = []
    while(x <= x_max):
        res = fsolve(func, x)[0]
        # print("Root: ",res)
        # print("All roots: ",roots)
        # print("Current search: ",E_search)
        
        x += step
        if any(np.abs(roots - res) < tol):
            continue
        roots.append(res)
        
    return roots