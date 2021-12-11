The notebook [tf_planck2018_lite.ipynb](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/notebooks/likelihoods_notebooks/tf_planck2018_lite.ipynb) shows an example of how to run a complete inference pipeline with power spectra sourced from ``CosmoPower``. The notebooks runs a version of the _Planck_ 2018 ``lite`` likelihood rewritten to be fully implemented in [TensorFlow](https://www.tensorflow.org/): [tf_planck2018_lite.py](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/likelihoods/tf_planck2018_lite). The ``lite`` version of the _Planck_ likelihood is pre-marginalised over a set of nuisance parameters. This TensorFlow version of the _Planck_ lite likelihood, provided as part of ``CosmoPower``, is an adaptation for TensorFlow of the [planck-lite-py](https://github.com/heatherprince/planck-lite-py) likelihood written by H. Prince and J. Dunkley.

If you use ``tf_planck2018_lite``, _in addition_ to the ``CosmoPower`` [release paper](https://arxiv.org/abs/2106.03846) please also cite [Prince & Dunkley (2019)](https://arxiv.org/abs/1909.05869) and [Planck (2018)](https://arxiv.org/abs/1907.12875).

The notebook [tf_planck2018_lite.ipynb](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/notebooks/likelihoods_notebooks/tf_planck2018_lite.ipynb) can also be run on Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TUDp1MWe0nU79JJXlsHBuMszWVpOLg7S?usp=sharing)


# ``tf_planck2018_lite`` instantiation

Her we will simply show how to instantiate the ``tf_planck2018_lite`` likelihood, referring to the [tf_planck2018_lite.ipynb](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/notebooks/likelihoods_notebooks/tf_planck2018_lite.ipynb) notebook for a more detailed example of how to run it for inference.

The ``tf_planck2018_lite`` likelihood requires emulators for the TT, TE, EE power spectra. In the [tf_planck2018_lite.ipynb](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/notebooks/likelihoods_notebooks/tf_planck2018_lite.ipynb) notebook we use the pre-trained models from the ``CosmoPower`` [release paper](https://arxiv.org/abs/2106.03846), available [in the ``CosmoPower`` repository](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/trained_models/CP_paper/CMB).

To create an instance of ``tf_planck2018_lite``, we import ``CosmoPower`` and remember to input:

- a path to the ``tf_planck2018_lite`` likelihood. It will be used to access the _Planck_ data;

- parameters of the analysis, as well as their priors;

- the ``CosmoPower`` emulators.


```python
import cosmopower as cp

# CosmoPower emulators
tt_emu_model = cp.cosmopower_NN(restore=True,
                                restore_filename='cmb_TT_NN')

te_emu_model = cp.cosmopower_PCAplusNN(restore=True,
                                       restore_filename='cmb_TE_PCAplusNN')

ee_emu_model = cp.cosmopower_NN(restore=True,
                                restore_filename='cmb_EE_NN')

# path to the tf_planck2018_lite likelihood
tf_planck2018_lite_path = '/path/to/cosmopower/likelihoods/tf_planck2018_lite/'

# parameters of the analysis, and their priors
parameters_and_priors = {'omega_b':      [0.001, 0.04, 'uniform'],
                         'omega_cdm':    [0.005, 0.99,  'uniform'],
                         'h':            [0.2,   1.0,   'uniform'],
                         'tau_reio':     [0.01,  0.8,   'uniform'],
                         'n_s':          [0.9,   1.1,   'uniform'],
                         'ln10^{10}A_s': [1.61,  3.91,  'uniform'],
                         'A_planck':     [1.0,   0.01,  'gaussian'],
                          }

# instantiation
tf_planck = cp.tf_planck2018_lite(parameters=parameters_and_priors, 
                                  tf_planck2018_lite_path=tf_planck2018_lite_path,
                                  tt_emu_model=tt_emu_model,
                                  te_emu_model=te_emu_model,
                                  ee_emu_model=ee_emu_model
                                  )
```
