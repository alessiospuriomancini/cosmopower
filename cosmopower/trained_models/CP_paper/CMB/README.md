# CMB emulators

## Parameter ranges

| Parameter  | Range |
| ---------  | ----- |
| <img src="https://latex.codecogs.com/gif.latex?\omega_{\mathrm{b}}"> | [0.005, 0.04] |
| <img src="https://latex.codecogs.com/gif.latex?\omega_{\mathrm{cdm}}"> | [0.001, 0.99] |
| <img src="https://latex.codecogs.com/gif.latex?h"> | [0.2, 1.0] |
| <img src="https://latex.codecogs.com/gif.latex?\tau_{\mathrm{reio}}"> | [0.01, 0.8] |
| <img src="https://latex.codecogs.com/gif.latex?n_{\mathrm{s}}"> | [0.7, 1.3] |
| <img src="https://latex.codecogs.com/gif.latex?\mathrm{ln}10^{10}A_{\mathrm{s}}"> | [1.61, 5] |


## Features

- cmb_TT_NN: ``cosmopower_NN`` mapping parameters to _log_-spectra;

- cmb_TE_NN: ``cosmopower_PCAplusNN`` mapping parameters to spectra;

- cmb_EE_NN: ``cosmopower_NN`` mapping parameters to log-spectra;

- cmb_EE_NN: ``cosmopower_PCAplusNN`` mapping parameters to log-spectra;

For all of them: 

- units: dimensionless;

- sampled multipoles: each multipole <img src="https://render.githubusercontent.com/render/math?math=\ell \in [2, 2508]">.


## Usage

```python
import cosmopower as cp

tt_emu = cp.cosmopower_NN(restore=True, 
                          restore_filename='cmb_TT_NN')
te_emu = cp.cosmopower_PCAplusNN(restore=True, 
                                 restore_filename='cmb_TE_PCAplusNN')
ee_emu = cp.cosmopower_NN(restore=True, 
                          restore_filename='cmb_EE_NN')
pp_emu = cp.cosmopower_PCAplusNN(restore=True, 
                                 restore_filename='cmb_PP_PCAplusNN')

# batch evaluation for e.g. three random sets of parameters
batch_params = {'omega_b': [0.0223, 0.0221, 0.0243],
                'omega_cdm': [0.112, 0.132, 0.142],
                'h': [0.67, 0.69, 0.73],
                'tau_reio': [0.06, 0.057, 0.08],
                'n_s': [0.94, 0.97, 0.81],
                'ln10^{10}A_s': [3.05, 3.07, 2.91],
                }

tt_spectra = tt_emu.ten_to_predictions_np(batch_params)
te_spectra = te_emu.predictions_np(batch_params)
ee_spectra = ee_emu.ten_to_predictions_np(batch_params)
pp_spectra = pp_emu.ten_to_predictions_np(batch_params)
```
