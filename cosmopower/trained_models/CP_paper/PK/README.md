# Matter power spectrum emulators

## Parameter ranges

| Parameter  | Range |
| ---------  | ----- |
| <img src="https://latex.codecogs.com/gif.latex?\omega_{\mathrm{b}}"> | [0.01875, 0.02625] |
| <img src="https://latex.codecogs.com/gif.latex?\omega_{\mathrm{cdm}}"> | [0.05, 0.255] |
| <img src="https://latex.codecogs.com/gif.latex?h"> | [0.64, 0.82] |
| <img src="https://latex.codecogs.com/gif.latex?n_{\mathrm{s}}"> | [0.84, 1.1] |
| <img src="https://latex.codecogs.com/gif.latex?\mathrm{ln}10^{10}A_{\mathrm{s}}"> | [1.61, 3.91] |
| <img src="https://latex.codecogs.com/gif.latex?c_\mathrm{min}"> | [2, 4] |
| <img src="https://latex.codecogs.com/gif.latex?\eta_0"> | [0.5, 1] |
|  <img src="https://latex.codecogs.com/gif.latex?z"> | [0, 5] |


## Features

- PKLIN_NN: ``cosmopower_NN`` mapping parameters to _log_-spectra (units: <img src="https://latex.codecogs.com/gif.latex?\mathrm{Mpc}^3">); 

- PKNLBOOST_NN: ``cosmopower_NN`` mapping parameters to _log_-spectra (=log-ratio of nonlinear and linear power). Nonlinear power spectrum obtained using [HMcode](https://github.com/alexander-mead/HMcode) ([Mead et al. 2020](https://arxiv.org/abs/2009.01858));

Sampled k-modes: see [k_modes.txt](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/trained_models/CP_paper/PK/k_modes.txt) (units:  <img src="https://latex.codecogs.com/gif.latex?\mathrm{Mpc}^{-1}">)

Note redshift <img src="https://latex.codecogs.com/gif.latex?z"> is treated as an additional input parameter for both emulators.


## Usage

```python
import cosmopower as cp

lin_emu = cp.cosmopower_NN(restore=True,
                           restore_filename='PKLIN_NN')

nlboost_emu = cp.cosmopower_NN(restore=True,
                               restore_filename='PKNLBOOST_NN')

# batch evaluation for e.g. three random sets of parameters
omega_b = [0.0223, 0.0221, 0.0243]
omega_cdm = [0.112, 0.132, 0.142]
h = [0.67, 0.69, 0.73]
n_s = [0.94, 0.97, 0.81]
lnAs = [3.05, 3.07, 2.91]
c_min = [2.3, 3.7, 2.5]
eta_0 = [0.7, 0.6, 0.9]
z = [0.3, 1., 0.7]

batch_params_lin = {'omega_b': omega_b,
                    'omega_cdm': omega_cdm,
                    'h': h,
                    'n_s': n_s,
                    'ln10^{10}A_s': lnAs,
                    'z': z,
                    }

batch_params_hmcode = {'c_min': c_min,
                       'eta_0': eta_0}

batch_params_nlboost = {**batch_params_lin, **batch_params_hmcode}

# (log_lin + log_boost)
total_log_power = lin_emu.predictions_np(batch_params_lin) + nlboost_emu.predictions_np(batch_params_nlboost)
# 10*(log_lin + log_boost)
total_power = 10.**(total_log_power)
```
