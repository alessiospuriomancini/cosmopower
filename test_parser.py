import numpy as np
import matplotlib.pyplot as plt
import cosmopower as cp

parser = cp.YAMLParser("example.yaml")
networks = parser.restore_networks()
print(networks)

for quantity in networks:
    network = networks[quantity]
    
    print(quantity, network.modes)

fiducial_params = {
    # Planck 2018 best-fitting theory
    "omega_b" : 0.022383,
    "omega_cdm" : 0.12011,
    "tau_reio" : 0.0543,
    "ln10^{10}A_s" : 3.0448,
    "n_s" : 0.96605,
    "h" : 0.6732
}

import camb

pars = camb.set_params(
    ns = 0.96605, As = 1e-10 * np.exp(3.0448), ombh2 = 0.022383, omch2 = 0.12011, tau = 0.0543, H0 = 67.32, lens_potential_accuracy = 1, lmax = 2508
)
res = camb.get_results(pars)
spec = res.get_cmb_power_spectra(CMB_unit = None)["total"]

fig, axes = plt.subplots(2, 2, figsize = (12, 8), sharex = True)

fig.delaxes(axes[0,1])

def plot_quantity(ax, quantity, log = True, c = "C0"):
    network = networks[quantity]
    ls = network.modes
    cls = network.predictions_np(fiducial_params)[0,:]
    if log:
        cls = network.ten_to_predictions_np(fiducial_params)[0,:]
        ax.semilogy()
    
    cls = cls * ls * (ls + 1.0) / (2.0 * np.pi)
    
    ax.plot(ls, cls, lw = 2, c = c)
    ax.set_xlim(2, ls.max())

plot_quantity(axes[0,0], "Cl/tt", c = "C0")
plot_quantity(axes[1,0], "Cl/te", log = False, c = "C1")
plot_quantity(axes[1,1], "Cl/ee", c = "C2")

axes[0,0].plot(np.arange(spec.shape[0]), spec[:,0], c = "k", ls = "--", lw = 2)
axes[1,0].plot(np.arange(spec.shape[0]), spec[:,3], c = "k", ls = "--", lw = 2)
axes[1,1].plot(np.arange(spec.shape[0]), spec[:,1], c = "k", ls = "--", lw = 2)

axes[0,0].set_title("TT")
axes[1,0].set_title("TE")
axes[1,1].set_title("EE")

plt.show()
