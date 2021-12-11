Here we simply show how to format the parameter and (log)-spectra data so that they can be read by ``cosmopower_NN`` and ``cosmopower_PCAplusNN`` during training.

### Preparing the files for use in training

```py
import numpy as np

# redefine 
k_modes = np.loadtxt('k_modes.txt')

# load predictions from Class: concatenations of parameters and spectra
linear_spectra_and_params = np.loadtxt('./linear.dat')
boost_spectra_and_params = np.loadtxt('./boost.dat')

# clean NaN's if any
rows = np.where(np.isfinite(linear_spectra_and_params).all(1))
linear_spectra_and_params = linear_spectra_and_params[rows]

rows = np.where(np.isfinite(boost_spectra_and_params).all(1))
boost_spectra_and_params = boost_spectra_and_params[rows]

# here the ordering should match the one used in `1_create_params.py`
params = ['omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s', 'c_min', 'eta_0', 'z']
n_params = len(params)

# separate parameters from spectra, take log
linear_parameters = linear_spectra_and_params[:, :n_params]
linear_log_spectra = np.log10(linear_spectra_and_params[:, n_params:])

boost_parameters = boost_spectra_and_params[:, :n_params]
boost_log_spectra = np.log10(boost_spectra_and_params[:, n_params:])

linear_parameters_dict = {params[i]: linear_parameters[:, i] for i in range(len(params))}
linear_log_spectra_dict = {'modes': k_modes,
                           'features': linear_log_spectra}

boost_parameters_dict = {params[i]: boost_parameters[:, i] for i in range(len(params))}
boost_log_spectra_dict = {'modes': k_modes,
                          'features': boost_log_spectra}

# save
np.savez('class_linear_params.npz', **linear_parameters_dict)
np.savez('class_linear_logpower.npz', **linear_log_spectra_dict)
np.savez('class_boost_params.npz', **boost_parameters_dict)
np.savez('class_boost_logpower.npz', **boost_log_spectra_dict)
```
