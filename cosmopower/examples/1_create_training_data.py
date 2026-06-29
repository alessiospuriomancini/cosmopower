#!/usr/bin/env/python

# Author: A. Spurio Mancini, H. T. Jense

from tqdm import tqdm
import numpy as np
from cosmopower.parser import YAMLParser
from cosmopower.spectra import init_boltzmann_code, get_boltzmann_spectra
from cosmopower.dataset import Dataset
import matplotlib.pyplot as plt

"""
Compare the example.yaml file. It contains a basic overview of a linear P(k,z)
emulator we want to train from 4,000 example spectra in LCDM parameters. In
addition, we generate 200 validation spectra across the same parameter range.

The result of this file will be several training spectra. You can compare the
result of this tutorial with invoking the command

    python -m cosmopower generate example.yaml

which generates both the LHC and full set of training and validation spectra
similar to how this script works.
"""

"""
Let's have a look at Cosmopower's python interface. We have an `example.yaml`
file which verbosely explains what kind of emulator we want to train. In this
case, the yaml file prescribes:
- we want to emulate P(k,z) computed by camb;
- we want to use ombh2, omch2, H0, ns, log(As), and z as inputs for the
  emulator;
- we want to generate 4000 training and 200 validation samples across this
  parameter space;
- we want a 3 layer 512 neuron network to emulate the computation of
  log(P(k,z)), using some learning settings we suggest there.

All this is written in the `example.yaml` file, which we need to parse for
interpretation in Cosmopower. The `YAMLParser` object can load this file for
us. This parser contains a variety of functions to automatically interpret this
file and generate or load the training data and emulators for us.
"""
parser = YAMLParser("example.yaml")

# Let's create some directories. Note the force_clean flag will clear out any
# existing files, if they alreay exist.
parser.setup_path(force_clean=True)

print(parser.all_parameters)

samples, validation_samples, testing_samples = \
    parser.get_parameter_samples()

for k in samples:
    print(k, samples[k][:10])

parser.save_samples_to_file(samples, validation_samples)

"""
Cosmopower has a series of built-in interpreters for boltzmann codes. We can
directly load the code from the parser.
"""
boltzmann_code = init_boltzmann_code(parser)
extra_args = parser.boltzmann_extra_args

"""
First we'll simply generate the first sample and plot it directly.
"""
params = {k: samples[k][0] for k in parser.boltzmann_inputs}

print("Plot a spectrum with these parameters:")
for k in params:
    print(f"{k:10s} = {params[k]:.3e}")

"""
Note that this sample does not include the redshift z, we use that separately.
"""
get_boltzmann_spectra(parser, boltzmann_code, params, ["Pk/lin"], extra_args)

"""
And now we plot k vs P(k)
"""
k = parser.modes("Pk/lin")
Pk = boltzmann_code["Pk/lin"]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(k, Pk)
ax.semilogx()
ax.semilogy()

ax.set_xlabel(r"k")
ax.set_ylabel(r"P(k)")
ax.set_title(f"Linear matter power spectrum at $z = {samples['z'][0]:.3f}$")

plt.show()

"""
Now we save this to a dataset.
Datasets are simply large table files that contain the spectra. The Dataset
wrapper included in Cosmopower helps interface these datasets between the
hdf5 file format and tensorflow.
"""
dataset = Dataset(parser, "Pk/lin", "Pk_lin.0.hdf5")
network_params = np.array([
    samples[k][0] for k in parser.network_input_parameters("Pk/lin")
])

"""
If we want to save the spectra as log(Pk) instead of Pk, we need to transform
them. Because even individual values of P(k) will vary over many orders of
magnitude, it is better to train the emulator over log(Pk).
"""
log = np.log10 if parser.is_log("Pk/lin") else (lambda x: x)

with Dataset(parser, "Pk/lin", "Pk_lin.0.hdf5") as dataset:
    if 0 not in dataset:
        dataset.write_data(0, network_params, log(Pk))

    """
    Now we just do this many more times!
    """
    for n in tqdm(range(1, parser.ntraining)):
        if n in dataset:
            continue
        params = {k: samples[k][n] for k in parser.boltzmann_inputs}
        if get_boltzmann_spectra(parser, boltzmann_code, params, ["Pk/lin"],
                                 extra_args):
            # This function returns False if the sample is invalid, so it's a
            # simple check to discard invalid datapoints.
            network_params = np.array([
                samples[k][n]
                for k in parser.network_input_parameters("Pk/lin")
            ])
            dataset.write_data(n, network_params,
                               log(boltzmann_code["Pk/lin"]))

"""
Obviously, we have to do the same for the validation and training datasets.
"""
with Dataset(parser, "Pk/lin", "Pk_lin.validation.0.hdf5") as dataset:
    for n in tqdm(range(parser.nvalidation)):
        if n in dataset:
            continue
        params = {k: validation_samples[k][n] for k in parser.boltzmann_inputs}
        if get_boltzmann_spectra(parser, boltzmann_code, params, ["Pk/lin"],
                                 extra_args):
            # This function returns False if the sample is invalid, so it's a
            # simple check to discard invalid datapoints.
            network_params = np.array([
                samples[k][n]
                for k in parser.network_input_parameters("Pk/lin")
            ])
            dataset.write_data(n, network_params,
                               log(boltzmann_code["Pk/lin"]))

with Dataset(parser, "Pk/lin", "Pk_lin.test.0.hdf5") as dataset:
    for n in tqdm(range(parser.ntesting)):
        if n in dataset:
            continue
        params = {k: testing_samples[k][n] for k in parser.boltzmann_inputs}
        if get_boltzmann_spectra(parser, boltzmann_code, params, ["Pk/lin"],
                                 extra_args):
            # This function returns False if the sample is invalid, so it's a
            # simple check to discard invalid datapoints.
            network_params = np.array([
                samples[k][n]
                for k in parser.network_input_parameters("Pk/lin")
            ])
            dataset.write_data(n, network_params,
                               log(boltzmann_code["Pk/lin"]))
