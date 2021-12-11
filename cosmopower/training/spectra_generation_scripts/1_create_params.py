#!/usr/bin/env/python

# Author: A. Spurio Mancini


import numpy as np
import pyDOE as pyDOE

# number of parameters and samples

n_params = 8
n_samples = 400000

# parameter ranges

obh2 =      np.linspace(0.01865, 0.02625, n_samples)

omch2 =     np.linspace(0.05,    0.255,   n_samples)

h =         np.linspace(0.64,    0.82,    n_samples)

ns =        np.linspace(0.84,    1.1,    n_samples)

lnAs =      np.linspace(1.61,    3.91,    n_samples)

cmin =      np.linspace(2.,      4.,    n_samples)

eta0 =      np.linspace(0.5,     1.,    n_samples)

z =         np.linspace(0,       5.,      n_samples)

# LHS grid

AllParams = np.vstack([obh2, omch2, h, ns, lnAs, cmin, eta0, z])
lhd = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
idx = (lhd * n_samples).astype(int)

AllCombinations = np.zeros((n_samples, n_params))
for i in range(n_params):
    AllCombinations[:, i] = AllParams[i][idx[:, i]]

# saving

params = {'omega_b': AllCombinations[:, 0],
          'omega_cdm': AllCombinations[:, 1],
          'h': AllCombinations[:, 2],
          'n_s': AllCombinations[:, 3],
          'ln10^{10}A_s': AllCombinations[:, 4],
          'c_min': AllCombinations[:, 5],
          'eta_0': AllCombinations[:, 6],
          'z': AllCombinations[:, 7],
           }

np.savez('your_LHS_parameter_file.npz', **params)
