# spectra_generation

This folder contains scripts to:

- generate a Latin Hypercube Sampling grid of parameters ([1_create_params.py](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/training/spectra_generation_scripts/1_create_params.py));

- produce power spectra at each node of the Latin Hypercube ([2_create_spectra.py](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/training/spectra_generation_scripts/2_create_spectra.py));

- re-arrange ([3_postprocess.py](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/training/spectra_generation_scripts/3_postprocess.)) the output files in a format compatible with that required by the training notebooks in the [training_notebooks](https://github.com/alessiospuriomancini/cosmopower/blob/main/notebooks/training_notebooks) folder.
