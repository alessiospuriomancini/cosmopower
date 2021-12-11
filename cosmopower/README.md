# cosmopower

This folder contains the ``CosmoPower`` package:

- [cosmopower_NN.py](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/cosmopower_NN.py) contains the class ``cosmopower_NN``. This class implements a direct neural network mapping between cosmological parameters and (log)-power spectra;

<div align="center">
<img src='https://github.com/alessiospuriomancini/cosmopower/blob/main/docs/static/nn_scheme-1.png' width="600" height="300">
</div>

- [cosmopower_PCA.py](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/cosmopower_PCA.py) contains the class ``cosmopower_PCA``, which performs (incremental) Principal Component Analysis of the training set (log-)power spectra. Note that an instance of this class is fed into an instance of the ``cosmopower_PCAplusNN`` class;

- [cosmopower_PCAplusNN.py](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/cosmopower_PCAplusNN.py) contains the class ``cosmopower_PCAplusNN``. This class implements a neural network mapping between cosmological parameters and PCA coefficients of the (log)-power spectra.

<div align="center">
<img src='https://github.com/alessiospuriomancini/cosmopower/blob/main/docs/static/pca_nn_scheme-1.png' width="600" height="300">
</div>


In the subfolder [likelihoods](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/likelihoods) the user can find examples of likelihoods that source power spectra from ``CosmoPower``.
