<div align="center"><img src="https://github.com/alessiospuriomancini/cosmopower/blob/main/docs/static/logo.png" width="500" height="400"> </div>

<div align="center">

![](https://img.shields.io/badge/Python-181717?style=plastic&logo=python)
![](https://img.shields.io/badge/Tensorflow-181717?style=plastic&logo=tensorflow)
![](https://img.shields.io/badge/License-GPLv3-181717?style=plastic)
![](https://img.shields.io/badge/Author-Alessio%20Spurio%20Mancini-181717?style=plastic)
![](https://img.shields.io/badge/Installation-pip%20install%20cosmopower-181717?style=plastic)

[Overview](#overview) •
[Documentation](#documentation) •
[Installation](#installation) •
[Getting Started](#getting-started) •
[Training](#training) •
[Trained Models](#trained-models) •
[Likelihoods](#likelihoods) •
[Support](#contributing-support-community) • 
[Citation](#citation)

</div>


# Overview

``CosmoPower`` is a library for Machine Learning - accelerated Bayesian inference. While the emphasis is on building algorithms to accelerate Bayesian inference in *cosmology*, the interdisciplinary nature of the methodologies implemented in the package allows for their application across a wide range of scientific fields. The ultimate goal of ``CosmoPower`` is to solve _inverse_ problems in science, by developing Bayesian inference pipelines that leverage the computational power of Machine Learning to accelerate the inference process. This approach represents a principled application of Machine Learning to scientific research, with the Machine Learning component embedded within a rigorous framework for uncertainty quantification.

In cosmology, ``CosmoPower`` aims to become a fully _differentiable_ library for cosmological analyses. Currently, ``CosmoPower`` provides neural network emulators of matter and Cosmic Microwave Background power spectra. These emulators can be used to replace Boltzmann codes such as [CAMB](https://github.com/cmbant/CAMB) or [CLASS](https://github.com/lesgourg/class_public) in cosmological inference pipelines, to source the power spectra needed for two-point statistics analyses. This provides orders-of-magnitude acceleration to the inference pipeline and integrates naturally with efficient techniques for sampling very high-dimensional parameter spaces. The power spectra emulators implemented in `CosmoPower`, and first presented in its [release paper](https://arxiv.org/abs/2106.03846), have been applied to the analysis of real cosmological data from  experiments, as well as having been tested against the accuracy requirements for the analysis of next-generation cosmological surveys.

``CosmoPower`` is written entirely in [Python](https://www.python.org/). Neural networks are implemented using the [TensorFlow](https://www.tensorflow.org/) library. Please check out [COSMOPOWER-JAX](https://github.com/dpiras/cosmopower-jax) for a [JAX](https://github.com/google/jax)-based version.


# Documentation

Comprehensive documentation is available [here](https://alessiospuriomancini.github.io/cosmopower).


# Installation

We recommend installing ``CosmoPower`` within a [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) virtual environment. 
For example, to create and activate an environment called ``cp_env``, use:

    conda create -n cp_env python=3.11 pip && conda activate cp_env

Once inside the environment, you can install ``CosmoPower``:

- **from PyPI or conda-forge**

        pip install cosmopower

    or

        conda install -c conda-forge cosmopower

    (kudos to [@joezuntz](https://github.com/joezuntz) for the conda-forge package).

    To test the installation, you can use

        python3 -c 'import cosmopower as cp'
    
    If you do not have a GPU on your machine, you will see a warning message about it which you can safely ignore.

- **from source**

        git clone https://github.com/alessiospuriomancini/cosmopower
        cd cosmopower
        pip install .

    To test the installation, you can use

        pytest


# Getting Started

CosmoPower currently provides two ways to emulate power spectra, implemented in the classes ``cosmopower_NN`` and ``cosmopower_PCAplusNN``:

<table border="0">
 <tr>
     <td><b style="font-size:30px"><div align="center">cosmopower_NN</div></b></td>
    <td><b style="font-size:30px"><div align="center">cosmopower_PCAplusNN</div></b></td>
 </tr>
 <tr>
    <td>a neural network mapping cosmological parameters directly to (log)-power spectra
<div align="center">
<img src='https://github.com/alessiospuriomancini/cosmopower/blob/main/docs/static/nn_scheme-1.png' width="800" height="300">
</div>
</td>
    <td>a neural network mapping cosmological parameters to coefficients of a Principal Component Analysis (PCA) of the (log)-power spectra<div align="center">
<img src='https://github.com/alessiospuriomancini/cosmopower/blob/main/docs/static/pca_nn_scheme-1.png' width="700" height="300">
</div>
</td>
 </tr>
</table>

Below you can find minimal working examples that use ``CosmoPower`` pre-trained models from the [code release paper](https://arxiv.org/abs/2106.03846), shared in the [trained_models](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/trained_models) folder (see the [Trained models](#trained-models) section for details) to predict power spectra for a given set of input parameters. You need to clone the repository and replace ``/path/to/cosmopower`` with the location of the cloned repository to make these examples work. Further examples are available as demo notebooks in the [getting_started_notebooks](https://github.com/alessiospuriomancini/cosmopower/blob/main/notebooks/getting_started_notebooks) folder, for both [cosmopower_NN](https://github.com/alessiospuriomancini/cosmopower/blob/main/notebooks/getting_started_notebooks/getting_started_with_cosmopower_NN.ipynb) ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fv70tJXCDnlTZYxMzr43q25PK5Kc3G7i?usp=sharing)) and [cosmopower_PCAplusNN](https://github.com/alessiospuriomancini/cosmopower/blob/main/notebooks/getting_started_notebooks/getting_started_with_cosmopower_PCAplusNN.ipynb) ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TZI15JEl0LvSyfoY812TkxueyU72nMGv?usp=sharing)).

Note that, whenever possible, we recommend working with models trained on _log_-power spectra, to reduce the dynamic range. Both ``cosmopower_NN`` and ``cosmopower_PCAplusNN`` have methods to provide predictions (cf. ``cp_pca_nn.predictions_np`` in the example below) as well as "10^predictions" (cf. ``cp_nn.ten_to_predictions_np`` in the example below).


<table>
<tr>
<th> Using <code>cosmopower_NN</code> </th>
<th> Using <code>cosmopower_PCAplusNN</code> </th>
</tr>
<tr>
<td>

```python
import cosmopower as cp

# load pre-trained NN model: maps cosmological parameters to CMB TT log-C_ell
cp_nn = cp.cosmopower_NN(restore=True, 
                         restore_filename='/path/to/cosmopower'\
                         +'/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')

# create a dict of cosmological parameters
params = {'omega_b': [0.0225],
          'omega_cdm': [0.113],
          'h': [0.7],
          'tau_reio': [0.055],
          'n_s': [0.96],
          'ln10^{10}A_s': [3.07],
          }

# predictions (= forward pass through the network) -> 10^predictions
spectra = cp_nn.ten_to_predictions_np(params)
```

</td>
<td>

```python
import cosmopower as cp

# load pre-trained PCA+NN model: maps cosmological parameters to CMB TE C_ell
cp_pca_nn = cp.cosmopower_PCAplusNN(restore=True, 
                                    restore_filename='/path/to/cosmopower'\
                                    +'/cosmopower/trained_models/CP_paper/CMB/cmb_TE_PCAplusNN')

# create a dict of cosmological parameters
params = {'omega_b': [0.0225],
          'omega_cdm': [0.113],
          'h': [0.7],
          'tau_reio': [0.055],
          'n_s': [0.96],
          'ln10^{10}A_s': [3.07],
          }

# predictions (= forward pass through the network)
spectra = cp_pca_nn.predictions_np(params)
```

</td>
</tr>
</table>

Note that the suffix ``_np`` of the ``predictions_np`` and ``ten_to_predictions_np`` functions refer to their implementation using [NumPy](https://numpy.org/). These functions are best suited to standard analysis pipelines fully implemented in normal Python, normally run on Central Processing Units. For pipelines built using the [TensorFlow](https://tensorflow.org/) library, highly optimised to run on Graphics Processing Units, we recommend the use of the corresponding ``_tf`` functions (i.e. ``predictions_tf`` and ``ten_to_predictions_tf``) in both ``cosmopower_NN`` and ``cosmopower_PCAplusNN`` (see [Likelihoods](#likelihoods) for further details and examples). 


# Training

The [training_notebooks](https://github.com/alessiospuriomancini/cosmopower/blob/main/notebooks/training_notebooks) folder contains examples of how to: 

- [train ``cosmopower_NN``](https://github.com/alessiospuriomancini/cosmopower/blob/main/notebooks/training_notebooks/cosmopower_NN_CMB_training.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eiDX_P0fxcuxv530xr2iceaPbY4CA5pD?usp=sharing)

- [train ``cosmopower_PCAplusNN``](https://github.com/alessiospuriomancini/cosmopower/blob/main/notebooks/training_notebooks/cosmopower_PCAplusNN_CMB_training.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1G8vABcUk9yztXYDx8bDFaNhJrtVIA5ei?usp=sharing)

These notebooks implement emulation of CMB temperature (TT) and lensing potential (<img src="https://render.githubusercontent.com/render/math?math=\phi \phi">)
 power spectra as practical examples - the procedure is completely analogous for the matter power spectrum.


# Trained Models

Trained models are available in the [trained_models](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/trained_models) folder. The folder contains all of the emulators used in the [CosmoPower release paper](https://arxiv.org/abs/2106.03846); as new models are trained, they will be shared in this folder, along with a description and BibTex entry of the relevant paper to be cited when using these models. Please consider sharing your own model in this folder with a pull request! 

Please refer to the [README](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/trained_models/README.md) file within the [trained_models](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/trained_models) folder for all of the details on the models contained there.


# Likelihoods

The [likelihoods](https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/likelihoods) folder contains examples of likelihood codes sourcing power spectra from ``CosmoPower``. Some of these likelihoods are written in pure [TensorFlow](https://github.com/tensorflow), hence they can be run with highly optimised [TensorFlow](https://github.com/tensorflow)-based samplers, such as the ones from [TensorFlow Probability](https://www.tensorflow.org/probability). Being written entirely in [TensorFlow](https://github.com/tensorflow), these codes can be massively accelerated by running on Graphics or Tensor Processing Units. We recommend the use of the ``predictions_tf`` and ``ten_to_predictions_tf`` functions within these pipelines, to compute (log)-power spectra predictions for input parameters. The [likelihoods_notebooks](https://github.com/alessiospuriomancini/cosmopower/blob/main/notebooks/likelihoods_notebooks) folder contains an [example](https://github.com/alessiospuriomancini/cosmopower/blob/main/notebooks/likelihoods_notebooks/tf_planck2018_lite.ipynb) of how to run a pure-Tensorflow likelihood, the Planck-lite 2018 TTTEEE likelihood [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1TUDp1MWe0nU79JJXlsHBuMszWVpOLg7S/view?usp=sharing).


# Contributing, Support, Community

For bugs and feature requests consider using the [issue tracker](https://github.com/alessiospuriomancini/cosmopower/issues). 

Contributions to the code via pull requests are most welcome!

For general support, please send an email to `a dot spuriomancini at ucl dot ac dot uk`, or post on [GitHub discussions](https://github.com/alessiospuriomancini/cosmopower/discussions).

Users of ``CosmoPower`` are strongly encouraged to join the [GitHub discussions](https://github.com/alessiospuriomancini/cosmopower/discussions) forum to follow the latest news on the code as well as to discuss all things Machine Learning / Bayesian Inference in cosmology!


# Citation

If you use ``CosmoPower`` at any point in your work please cite its [release paper](https://arxiv.org/abs/2106.03846):

    @article{SpurioMancini2022,
             title={CosmoPower: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys},
             volume={511},
             ISSN={1365-2966},
             url={http://dx.doi.org/10.1093/mnras/stac064},
             DOI={10.1093/mnras/stac064},
             number={2},
             journal={Monthly Notices of the Royal Astronomical Society},
             publisher={Oxford University Press (OUP)},
             author={Spurio Mancini, Alessio and Piras, Davide and Alsing, Justin and Joachimi, Benjamin and Hobson, Michael P},
             year={2022},
             month={Jan},
             pages={1771–1788}
             }

If you use a specific likelihood or trained model then in addition to the [release paper](https://arxiv.org/abs/2106.03846) please _also_ cite their relevant papers (always listed in the corresponding directory). If you use the custom activation function implemented in the code please also cite [Alsing et al.(2020)](https://doi.org/10.3847/1538-4365/ab917f).


# License

``CosmoPower`` is released under the GPL-3 license (see [LICENSE](https://github.com/alessiospuriomancini/cosmopower/blob/main/LICENSE)) subject to 
the non-commercial use condition (see [LICENSE_EXT](https://github.com/alessiospuriomancini/cosmopower/blob/main/LICENSE_EXT)).

    CosmoPower
    Copyright (C) 2021 A. Spurio Mancini & contributors
    
    This program is released under the GPL-3 license (see LICENSE), 
    subject to a non-commercial use condition (see LICENSE_EXT).
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
