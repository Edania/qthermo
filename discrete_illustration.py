import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Segoe UI'

plt.rcParams["mathtext.fontset"] = "cm"

#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color = plt.get_cmap("tab20c").colors)

ds = np.linspace(0,1,1000)
ks = np.array([-0.5,0.4]).reshape(-1,1)
#ks = np.array([4,5]).reshape(-1,1)
ms = np.array([0.7,0.3]).reshape(-1,1)
bs = np.array([-5,-3]).reshape(-1,1)
Is = ds*ks + ms
#Is = bs*ds**2 + ds*ks + ms
fig = plt.figure(figsize = (2.7,2.3), layout = "constrained")
plt.plot(ds, Is.T[:,0],"tab:blue")
plt.plot(ds, Is.T[:,1],"tab:cyan")
plt.scatter(1, float(Is.T[-1,0]), color="tab:blue")
plt.scatter(0, float(Is.T[0,1]), color="tab:cyan")
plt.title("Illustration of how average current \n changes with transmission height")#, fontname = "Segoe UI")
plt.xlabel(r"$d_\gamma$", size = 12)
plt.ylabel(r"$I_y$", size = 12)
#plt.ylabel(r"$S_x$", size = 12)
plt.yticks([])
plt.text(0.8, Is.T[-1,0] + 0.1, r"$d_1$", color = "tab:blue", size = 12)
plt.text(0.1, 0.4, r"$d_2$", color = "tab:cyan", size = 12)
plt.text(0.7, 0.4, r"$\left.\frac{\partial I_y}{\partial d_1}\right|_{I_x} < 0$", color = "tab:blue", size = 12)
plt.text(0.25, 0.3, r"$\left.\frac{\partial I_y}{\partial d_2}\right|_{I_x} > 0$", color = "tab:cyan", size = 12)
#plt.legend()
plt.savefig("figs/linear_illus.pdf")