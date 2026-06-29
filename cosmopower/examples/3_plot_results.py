#!/usr/bin/env/python

# Author: A. Spurio Mancini, H. T. Jense

import numpy as np
import matplotlib.pyplot as plt
from cosmopower.parser import YAMLParser
from cosmopower.dataset import Dataset

"""
In the previous examples, we generated training data and trained a cosmopower
network to emulate the LCDM computation of P(k,z). Here we will compare the
accuracy of the emulator with the validation dataset and compare the accuracy.
"""
parser = YAMLParser("example.yaml")

"""
First, let's plot some simple progress monitoring of the emulator training.
"""
step, learning_rate, batch_size, epoch, vloss, vloss_best = \
    np.loadtxt("example/networks/example_Pk_lin.progress", unpack=True)

step = step.astype(int)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(step, vloss, c="blue", lw=1, label="Validation Loss")
ax.step(step, vloss_best, c="red", lw=1, ls="--", label="Best Validation Loss")
ax.semilogy()

for i in step[1:]:
    if batch_size[i-1] != batch_size[i]:
        ax.axvline(i, c="grey", ls=":", lw=1)

ax.set_xlabel("Learning step")
ax.set_ylabel("Loss")
ax.set_xlim(0, None)

ax.legend(loc="best")

plt.show()

"""
Next up, let's load the validation dataset and plot the prediction versus a
random spectrum.
"""
samples = parser.get_parameter_samples()

emulators = parser.restore_networks()
k = parser.modes("Pk/lin")

pk_lin_emu = emulators["Pk/lin"]

"""
Load a random validation spectrum (we use the validation dataset, because the
emulator was not trained over these spectra).
"""
with Dataset(parser, "Pk/lin", "Pk_lin.testing.0.hdf5") as dataset:
    j = int(np.random.choice(dataset.indices))

    params = dataset.read_parameters(j)
    pk = dataset.read_spectra(j)

print(params)
print(pk.shape)
prediction = pk_lin_emu.ten_to_predictions_np(params)[0, :]
print(prediction.shape)
example = 10. ** pk


fig, axes = plt.subplots(2, 1, figsize=(8, 6),
                         gridspec_kw={"height_ratios": [3, 2]}, sharex=True)

ax = axes[0]
ax.plot(k, example, c="k", lw=1, ls="--", label="camb")
ax.plot(k, prediction, c="r", lw=1, label="Emulator")
ax.set_title(f"Linear Matter Power Spectrum at $z={params['z']:.3f}$")
ax.set_ylabel(r"$P(k)$")
ax.legend(loc="best")

ax.semilogx()
ax.semilogy()

ax = axes[1]
ax.axhline(0.0, c="k", lw=1, ls="--")
ax.plot(k, 100.0 * (prediction - example) / example, c="r", lw=1)

ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\Delta P / P(k)$ [%]")

plt.show()

"""
Let's compute the percentile regions of the absolute difference between the
validation dataset and the emulator.
"""
with Dataset(parser, "Pk/lin", "Pk_lin.testing.0.hdf5") as dataset:
    inputs, outputs = dataset.read_data(as_dict=True)

prediction = pk_lin_emu.ten_to_predictions_np(inputs)
true_values = 10. ** outputs

quantiles = [68.0, 95.0, 99.0]
abs_diff = 100.0 * np.abs((prediction - true_values) / true_values)
mean_abs_diff = np.mean(abs_diff, axis=0)
percentiles = np.percentile(abs_diff, quantiles, axis=0)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(k, mean_abs_diff, c="k", ls=":", lw=1, label="Mean")
ax.fill_between(k, percentiles[0, :], np.zeros_like(k), fc="red", alpha=0.7,
                label="68%")
ax.fill_between(k, percentiles[1, :], np.zeros_like(k), fc="red", alpha=0.5,
                label="95%")
ax.fill_between(k, percentiles[2, :], np.zeros_like(k), fc="red", alpha=0.3,
                label="99%")

ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$|\Delta P(k)|\, / \, P(k)$ [%]", fontsize=14)

ax.set_xlim(k.min(), k.max())
ax.set_ylim(0.0, None)
ax.semilogx()

ax.legend(loc="best")

plt.show()
