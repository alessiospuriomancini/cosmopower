#!/usr/bin/env/python

# Author: A. Spurio Mancini

import numpy as np
import classy
import sys

krange1 = np.logspace(np.log10(1e-5), np.log10(1e-4), num=20, endpoint=False) 
krange2 = np.logspace(np.log10(1e-4), np.log10(1e-3), num=40, endpoint=False)
krange3 = np.logspace(np.log10(1e-3), np.log10(1e-2), num=60, endpoint=False)
krange4 = np.logspace(np.log10(1e-2), np.log10(1e-1), num=80, endpoint=False)
krange5 = np.logspace(np.log10(1e-1), np.log10(1), num=100, endpoint=False)
krange6 = np.logspace(np.log10(1), np.log10(10), num=120, endpoint=False)

k = np.concatenate((krange1, krange2, krange3, krange4, krange5, krange6))
num_k = len(k)  # 420 k-modes
np.savetxt('k_modes.txt', k)

cosmo = classy.Class()

params_lhs = np.load('your_LHS_parameter_file.npz')


def spectra_generation(i):
    print('parameter set ', i)

    # Define your cosmology (what is not specified will be set to CLASS default parameters)
    params = {'output': 'tCl mPk',
             'non linear':'hmcode',
             'z_max_pk': 5,
             'P_k_max_1/Mpc': 10.,
             'nonlinear_min_k_max': 100.,
             'N_ncdm' : 0,
             'N_eff' : 3.046,
             'omega_b': params_lhs['omega_b'][i],
             'omega_cdm': params_lhs['omega_cdm'][i],
             'h': params_lhs['h'][i],
             'n_s': params_lhs['n_s'][i],
             'ln10^{10}A_s': params_lhs['ln10^{10}A_s'][i],
             'c_min': params_lhs['c_min'][i],
             'eta_0': params_lhs['eta_0'][i], 
             }

    # Set the parameters to the cosmological code
    cosmo.set(params)

    try:
        cosmo.compute()
        z = params_lhs['z'][i]

        # non linear power spectrum
        Pnonlin = np.array([cosmo.pk(ki, z) for ki in k])

        # linear power spectrum
        Plin = np.array([cosmo.pk_lin(ki, z) for ki in k])
        cosmo_array = np.hstack(([params_lhs[k][i] for k in params_lhs], Plin))
        f=open('./linear.dat','ab')
        np.savetxt(f, [cosmo_array])
        f.close()

        # non linear boost
        ratio = Pnonlin/Plin
        cosmo_array = np.hstack(([params_lhs[k][i] for k in params_lhs], ratio))
        f=open('./boost.dat','ab')
        np.savetxt(f, [cosmo_array])
        f.close()


    # parameter set rejected by Class
    except classy.CosmoComputationError as failure_message:
        print(str(failure_message)+'\n')
        cosmo.struct_cleanup()
        cosmo.empty()

    # something wrong in Class init
    except classy.CosmoSevereError as critical_message:
        print("Something went wrong when calling CLASS" + str(critical_message))
        cosmo.struct_cleanup()
        cosmo.empty()
    return

# loop over parameter sets
for i in range(len(params_lhs['omega_b'])):
    spectra_generation(i)
