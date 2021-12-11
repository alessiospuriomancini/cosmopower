# likelihoods 

This folder contains examples of likelihoods sourcing power spectra from ``CosmoPower``. The user who develops a new likelihood for use with ``CosmoPower`` is invted to share it within this folder.

Each likelihood is contained in a subfolder. In the subfolder, a README.md file explains details of the likelihood: importantly, these include the BibTex entry for the paper to be cited when using that likelihood.

Some of the likelihoods contained in this folder may have be highly optimised for runs on Graphics Processing Units, by virtue of being pure-[TensorFlow](https://www.tensorflow.org/) implementations (for example, [tf_planck2018_lite](https://github.com/alessiospuriomancini/cosmopower_public/blob/main/likelihoods/tf_planck2018_lite)). These likelihoods all have a ``tf_`` prefix in their name, to highlight their native TensorFlow implementation.
