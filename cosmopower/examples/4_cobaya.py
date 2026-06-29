#!/usr/bin/env/python

# Author: H. T. Jense

"""
Cosmopower comes with a built-in wrapper for cobaya. This allows for very fast
parameter inference from cosmological data.

In this example, we will run a quick cosmological chain using cobaya to infer
cosmological parameters from the Planck 2018 data. Make sure you have cobaya
installed in your python environment.

We provide the `planck_cmb.yaml` yaml file to prescribe some simple parameter
ranges for a CMB emulator that computes the TT, TE, and EE CMB power spectra
that are measured by Planck.

Using the tutorials provided before, you can train your own emulator to compute
these quantities. This can take a while, so alternatively you can take a look
at the pre-generated emulators provided at
<https://github.com/cosmopower-organization>, which can be used with the
framework provided here.
"""

from cobaya.yaml import yaml_load
from cobaya.install import install
from cobaya.model import get_model
from cobaya.output import get_output
from cobaya.sampler import get_sampler
import numpy as np
import matplotlib.pyplot as plt

model = r"""
likelihood:
  planck_2018_lowl.EE:
    stop_at_error: true
  planck_2018_lowl.TT:
    stop_at_error: true
  planck_2018_highl_plik.TTTEEE_lite_native:
    stop_at_error: true

theory:
  cosmopower:
    package_file: planck_cmb.yaml
    stop_at_error: true

params:
  ombh2:
    prior:
      min: 0.020
      max: 0.024
    ref: 0.022
  omch2:
    prior:
      min: 0.10
      max: 0.14
    ref: 0.12
  cosmomc_theta:
    prior:
      min: 103.7e-4
      max: 104.4e-4
    ref: 104.1e-4
  logA:
    prior:
      min: 2.9
      max: 3.2
    ref: 3.05
  ns:
    prior:
      min: 0.92
      max: 1.01
    ref: 0.96
  tau:
    prior:
      min: 0.01
      max: 0.12
    ref: 0.05
  h:
    prior:
      min: 0.6
      max: 0.8
    ref: 0.67

"""

yaml = yaml_load(model)

install(yaml)

# Create the cobaya model.
model = get_model(yaml)

# The best-fit values from the Planck paper.
plik_best_fit = {
    "ombh2": 22.383e-3,
    "omch2": 12.011e-2,
    "cosmomc_theta": 104.0909e-4,
    "h": 0.67,
    "logA": 3.0448,
    "ns": 0.96605,
    "tau": 0.0543,
    "A_planck": 1.0,
}

# Compute the log-posterior at best-fit.
logpost = model.logposterior(plik_best_fit, as_dict=True)
print(logpost)

"""
Make a plot of the spectra and datapoints.
"""
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

cl = model.provider.get_Cl(ell_factor=True)
plik = model.likelihood["planck_2018_highl_plik.TTTEEE_lite_native"]

fig.delaxes(axes[0, 1])

ax = axes[0, 0]
ax.plot(cl["ell"], cl["tt"], lw=2, c="C0")

lav = plik.lav[plik.used_bins[0]]
Xvec = plik.X_data[plik.used_bins[0]] * (lav * (lav + 1)) / 2. / np.pi
Xerr = np.sqrt(np.diag(plik.cov))[plik.used_bins[0]] * \
       (lav * (lav + 1)) / 2. / np.pi
i0 = len(plik.used_bins[0])

ax.errorbar(lav, Xvec, yerr=Xerr, c="k", marker=".", lw=0, elinewidth=1)

ax.set_xlim(0, 2508)
ax.set_title("TT")


ax = axes[1, 0]
ax.plot(cl["ell"], cl["te"], lw=2, c="C1")


lav = plik.lav[plik.used_bins[1]]
Xvec = plik.X_data[plik.used_bins[1] + i0] * (lav * (lav + 1)) / 2. / np.pi
Xerr = np.sqrt(np.diag(plik.cov))[plik.used_bins[1] + i0] * \
       (lav * (lav + 1)) / 2. / np.pi
i0 = i0 + len(plik.used_bins[1])

ax.errorbar(lav, Xvec, yerr=Xerr, c="k", marker=".", lw=0, elinewidth=1)

ax.set_title("TE")

ax = axes[1, 1]
ax.plot(cl["ell"], cl["ee"], lw=2, c="C2")

lav = plik.lav[plik.used_bins[2]]
Xvec = plik.X_data[plik.used_bins[2] + i0] * (lav * (lav + 1)) / 2. / np.pi
Xerr = np.sqrt(np.diag(plik.cov))[plik.used_bins[2] + i0] * \
       (lav * (lav + 1)) / 2. / np.pi
i0 = i0 + len(plik.used_bins[2])

ax.errorbar(lav, Xvec, yerr=Xerr, c="k", marker=".", lw=0, elinewidth=1)

ax.set_title("EE")

plt.show()

# Write to some output file
output = get_output(prefix="chains/example_planck", resume=False, force=True)

# Create an MCMC sampler for cobaya.
sampler = get_sampler({"mcmc": None}, model=model, output=output)

# Run the sampler!
sampler.run()
