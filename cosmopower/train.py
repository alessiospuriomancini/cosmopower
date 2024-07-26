import os
import numpy as np
from typing import Optional
from .cosmopower_NN import cosmopower_NN
from .cosmopower_PCA import cosmopower_PCA
from .cosmopower_PCAplusNN import cosmopower_PCAplusNN
from .dataset import Dataset
from .parser import YAMLParser
from .util import get_noise_curves_SO, get_noise_curves_CVL


def find_files(parser: YAMLParser, quantity: str) -> np.ndarray:
    """
    This function locates the data that quantity {quantity} should be trained
    on in the parser, and returns a data array, which is a numpy array of
    dimension (N, K) with K being the dimension of a spectrum and N being the
    number of training samples.
    """
    import glob

    fn = os.path.join(parser.path, "spectra",
                      quantity.replace("/", "_") + ".*.hdf5")

    return glob.glob(fn)


def train_network_NN(parser: YAMLParser, quantity: str, device: str = "",
                     overwrite: bool = False) -> Optional[cosmopower_NN]:
    """
    This function should take in a network (of type cosmopower_NN)
    and then train it on the given data.

    parser: The YAML parser for this environment.
    quantity: The quantity that we are training for.
    device: The tensorflow device that we want to use.
    overwrite: Whether or not to overwrite if the file already exists.

    Return: The emulator if successful, or None if not.
    """
    filenames = find_files(parser, quantity)

    if len(filenames) == 0:
        raise IOError(f"No files found to train quantity {quantity} with.")

    saved_filename = os.path.join(parser.path, "networks",
                                  parser.network_filename(quantity))
    if os.path.exists(saved_filename) and not overwrite:
        print("\tFile already exists, skipping.")
        return None

    parameters = parser.network_input_parameters(quantity)

    settings = parser.get_traits(quantity)

    import tensorflow as tf

    network = cosmopower_NN(parameters=parameters,
                            modes=parser.modes(quantity), verbose=True,
                            **settings.get("n_traits", {}))

    datasets = [Dataset(parser, quantity, os.path.basename(filename))
                for filename in filenames]

    with tf.device(device):
        print("\tTraining NN.")
        network.train(training_data=datasets,
                      filename_saved_model=saved_filename,
                      **parser.network_training_parameters(quantity))

    return network


def train_network_PCAplusNN(parser: YAMLParser, quantity: str,
                            settings: dict = {}, device: str = "",
                            overwrite: bool = False
                            ) -> Optional[cosmopower_PCAplusNN]:
    """
    This function should take in a network (of type cosmopower_PCAplusNN)
    and then train it on the given data.

    Return: whether or not the training was successful.
    """
    filenames = find_files(parser, quantity)

    saved_filename = os.path.join(parser.path, "networks",
                                  parser.network_filename(quantity))
    if os.path.exists(saved_filename) and not overwrite:
        print("\tNetwork file already exists, skipping.")
        return None

    import tensorflow as tf

    datasets = [Dataset(parser, quantity, os.path.basename(filename))
                for filename in filenames]

    parameters = parser.network_input_parameters(quantity)

    settings = parser.get_traits(quantity)
    n_pcas = settings.get("p_traits").get("n_pcas")
    n_batches = settings.get("p_traits").get("n_batches", 2)
    modes = parser.modes(quantity)

    cp_pca = cosmopower_PCA(parameters=parameters, modes=modes, n_pcas=n_pcas,
                            n_batches=n_batches, verbose=True)
    network = cosmopower_PCAplusNN(cp_pca=cp_pca, verbose=True,
                                   **settings.get("n_traits", {}))

    with tf.device(device):
        print("\tTraining PCA+NN.")
        network.train(training_data=datasets,
                      filename_saved_model=saved_filename,
                      **parser.network_training_parameters(quantity))

    return network


def show_training(args: Optional[list] = None) -> None:
    """
    When you run
        [python -m] cosmopower show-training <yamlfile>
    this should make a plot of all training spectra and show it.
    """

    import argparse
    argp = argparse.ArgumentParser(prog="cosmopower show-training",
                                   description="Plot all generated spectra.")
    argp.add_argument("yamlfile", type=str, help="The .yaml file to parse.")
    argp.add_argument("-d", "--dir", dest="root_dir", type=str,
                      help="A root directory to work in.")
    argp.add_argument("-f", "--fraction", dest="plot_fraction", type=float,
                      default=0.3, help="The fraction of spectra to plot.")
    argp.add_argument("-s", "--show", dest="showfig", action="store_true",
                      help="Show the pyplot figure dialog.")
    argp.add_argument("-m", "--plot-mean", dest="plotmean",
                      action="store_true",
                      help="Overplot the mean in each figure.")

    args = argp.parse_args(args)

    import matplotlib.pyplot as plt

    parser = YAMLParser(args.yamlfile, root_dir=args.root_dir)

    plot_quantities = [q for q in parser.quantities if q != "derived"]

    nx = int(np.ceil(np.sqrt(len(plot_quantities))))
    ny = len(plot_quantities) // nx

    fig, axes = plt.subplots(ny, nx, figsize=(12, 8))

    for n, quantity in enumerate(plot_quantities):
        filenames = find_files(parser, quantity)

        if len(filenames) == 0:
            raise IOError(f"No files found for {quantity}.")

        i = n % nx
        j = n // nx

        data = None
        nplot = 0
        modes = parser.modes(quantity)

        for filename in filenames:
            with Dataset(parser, quantity,
                         os.path.basename(filename)) as dataset:
                idx = dataset.indices
                np.random.shuffle(idx)
                idx = idx[:int(args.plot_fraction * len(idx))]
                nplot += len(idx)
                spec = dataset.read_spectra(idx)

            if data is None:
                data = spec
            else:
                data = np.concatenate((data, spec), axis=0)

        print(f"Plotting {nplot} spectra for {quantity}.")

        if ny > 1:
            ax = axes[j, i]
        elif nx > 1:
            ax = axes[n]
        else:
            ax = axes

        if parser.is_log(quantity):
            data = np.power(10.0, data)
            ax.semilogy()

        ax.plot(modes, data.T, c=f"C{n}", lw=1, alpha=0.3)
        if args.plotmean:
            ax.plot(modes, np.nanmean(data, axis=0), lw=2, c="k")

        if parser.modes_log(quantity):
            ax.semilogx()
        ax.set_xlim(modes.min(), modes.max())
        ax.set_xlabel("$" + parser.modes_label(quantity) + "$")

        ax.set_title(quantity)

    if args.showfig:
        plt.show()
    else:
        filename = os.path.join(parser.path, "training_data.pdf")
        plt.savefig(filename, bbox_inches="tight")


def show_validation(args: Optional[list] = None) -> None:
    """
    When you run
        [python -m] cosmopower show-validation <yamlfile>
    this should make a plot of all validation spectra, compute the cosmopower
    prediction and then plot the differences.
    """

    import argparse
    argp = argparse.ArgumentParser(prog="cosmopower show-validation",
                                   description="Plot the prediction \
                                                residuals.")
    argp.add_argument("yamlfile", type=str, help="The .yaml file to parse.")
    argp.add_argument("-s", "--show", dest="showfig", action="store_true",
                      help="Show the pyplot figure dialog.")
    argp.add_argument("-d", "--dir", dest="root_dir", type=str,
                      help="A root directory to work in.")
    argp.add_argument("-N", "--noise-curve", dest="noise_curve", default="",
                      type=str, choices=["SO", "CVL"], help="Use a specific \
                      noise curve for comparison, instead of percentual \
                      error. Options: CVL (cosmic variance limit), SO (Simons \
                      Observatory).")
    argp.add_argument("-f", "--fraction", dest="plot_fraction", type=float,
                      default=0.1, help="The fraction of spectra to plot.")

    args = argp.parse_args(args)

    import matplotlib.pyplot as plt

    parser = YAMLParser(args.yamlfile, root_dir=args.root_dir)

    networks = parser.restore_networks()

    plot_quantities = [q for q in networks if q != "derived"]

    denominator = {}
    data = {}
    inputs = {}

    for quantity in networks:
        if quantity == "derived":
            continue

        filenames = find_files(parser, quantity)

        if len(filenames) == 0:
            raise IOError(f"No files found for {quantity}.")

        nplot = 0

        for filename in filenames:
            with Dataset(parser, quantity,
                         os.path.basename(filename)) as dataset:
                idx = dataset.indices
                np.random.shuffle(idx)
                n = max(1, int(args.plot_fraction * len(idx)))
                idx = idx[:n]
                nplot += len(idx)

                pars = dataset.read_parameters(idx)
                spec = dataset.read_spectra(idx)

                inp = inputs.get(quantity, None)
                dat = data.get(quantity, None)

                if dat is None:
                    inputs[quantity] = pars
                    data[quantity] = spec
                else:
                    for k in inp:
                        inputs[quantity][k] = np.concatenate([inp[k], pars[k]])
                    data[quantity] = np.concatenate([dat, spec])

        print(f"Averaging over {nplot} spectra for {quantity}.")

    nx = int(np.ceil(np.sqrt(len(plot_quantities))))
    ny = 2 * (len(plot_quantities) // nx + 1)

    fig, axes = plt.subplots(ny, nx, figsize=(12, 8))

    if args.noise_curve == "CVL":
        for quantity in networks:
            if quantity == "derived":
                continue

            denominator[quantity] = data[quantity]
            if parser.is_log(quantity):
                denominator[quantity] = np.power(10.0, denominator[quantity])

        denominator = get_noise_curves_CVL(parser, denominator)
    elif args.noise_curve == "SO":
        for quantity in networks:
            if quantity == "derived":
                continue

            denominator[quantity] = data[quantity]
            if parser.is_log(quantity):
                denominator[quantity] = np.power(10.0, denominator[quantity])

        denominator = get_noise_curves_SO(parser, denominator)
    else:
        for quantity in networks:
            if quantity == "derived":
                continue

            denominator[quantity] = data[quantity]
            if parser.is_log(quantity):
                denominator[quantity] = np.power(10.0, denominator[quantity])

    for n, quantity in enumerate(plot_quantities):
        print(f"Plotting {quantity}.")
        modes = parser.modes(quantity)

        ins = inputs[quantity].copy()

        try:
            if parser.is_log(quantity):
                data[quantity] = np.power(10.0, data[quantity])
                prediction = networks[quantity].ten_to_predictions_np(ins)
            else:
                prediction = networks[quantity].predictions_np(ins)
        except KeyError:
            continue

        p = [68.0, 95.0, 99.0]

        i = n % nx
        j = 2 * (n // nx)

        diff = (prediction - data[quantity]) / data[quantity]
        absdiff = np.abs(prediction - data[quantity]) / denominator[quantity]

        percentiles = np.zeros((6, diff.shape[1]))
        # 1, 2, and 3 sigma bands
        for k in range(3):
            percentiles[k, :] = np.percentile(diff, 100.0 - p[k], axis=0)
            percentiles[k + 3, :] = np.percentile(diff, p[k], axis=0)

        if nx > 1:
            ax = axes[j, i]
        else:
            ax = axes[j]
        ax.set_title(quantity)
        ax.set_xlim(modes.min(), modes.max())
        mdata = np.nanmean(data[quantity], axis=0)
        lo_err = mdata[np.newaxis, :] * (1.0 + percentiles[:3, :])
        hi_err = mdata[np.newaxis, :] * (1.0 + percentiles[3:, :])

        if parser.is_log(quantity):
            ax.semilogy()

        ax.fill_between(modes, lo_err[2, :], hi_err[2, :], color=f"C{n}",
                        alpha=0.3, lw=0)
        ax.fill_between(modes, lo_err[1, :], hi_err[1, :], color=f"C{n}",
                        alpha=0.6, lw=0)
        ax.fill_between(modes, lo_err[0, :], hi_err[0, :], color=f"C{n}",
                        alpha=0.9, lw=0)
        ax.plot(modes, mdata, color="k", lw=2)

        ax.set_xlim(modes.min(), modes.max())
        if parser.modes_log(quantity):
            ax.semilogx()
        ax.set_xticks([])

        if nx > 1:
            ax = axes[j + 1, i]
        else:
            ax = axes[j + 1]

        percentiles = np.zeros((3, diff.shape[1]))
        # 1, 2, and 3 sigma bands
        percentiles[0, :] = np.percentile(absdiff, 68.0, axis=0)
        percentiles[1, :] = np.percentile(absdiff, 95.0, axis=0)
        percentiles[2, :] = np.percentile(absdiff, 99.0, axis=0)

        ax.fill_between(modes, 0, percentiles[2, :], color=f"C{n}", alpha=0.3,
                        lw=0)
        ax.fill_between(modes, 0, percentiles[1, :], color=f"C{n}", alpha=0.6,
                        lw=0)
        ax.fill_between(modes, 0, percentiles[0, :], color=f"C{n}", alpha=0.9,
                        lw=0)
        ax.set_xlim(modes.min(), modes.max())

        if parser.modes_log(quantity):
            ax.semilogx()

        ax.set_ylim(0.0, None)
        _, ymax = ax.get_ylim()
        ax.set_xlabel("$" + parser.modes_label(quantity) + "$")

        if i == 0:
            if args.noise_curve == "CVL":
                ax.set_ylabel(r"$\Delta C_\ell \, / \, \
                                \sigma_{C_\ell,\mathrm{CVL}}$")
            elif args.noise_curve == "SO":
                ax.set_ylabel(r"$\Delta C_\ell \, / \, \
                                \sigma_{C_\ell,\mathrm{SO}}$")
            else:
                ax.set_ylabel(r"$\Delta C_\ell \, / \, \
                                C_\ell^\mathrm{theory}$")
                ax.set_yticklabels([f"{x:,.1%}" for x in ax.get_yticks()])

    # Delete unused plots
    for n in range(len(plot_quantities), nx * ny // 2):
        i = n % nx
        j = 2 * (n // nx)

        if nx > 1:
            ax = axes[j, i]
        else:
            ax = axes[j]

        fig.delaxes(ax)

        if nx > 1:
            ax = axes[j + 1, i]
        else:
            ax = axes[j + 1]

        fig.delaxes(ax)

    if args.showfig:
        plt.show()
    else:
        filename = os.path.join(parser.path, "validation.pdf")
        plt.savefig(filename, bbox_inches="tight")

    if "derived" in networks:
        # make derived parameter plot.
        nx = int(np.sqrt(len(parser.computed_parameters)))
        ny = (len(parser.computed_parameters) // nx)

        filenames = find_files(parser, "derived")

        if len(filenames) == 0:
            raise IOError("No files found for derived parameters.")

        nplot = 0

        inputs = None
        data = None

        for filename in filenames:
            with Dataset(parser, "derived",
                         os.path.basename(filename)) as dataset:
                idx = dataset.indices
                np.random.shuffle(idx)
                n = max(1, int(args.plot_fraction * len(idx)))
                idx = idx[:n]
                nplot += len(idx)

                pars = dataset.read_parameters(idx)
                spec = dataset.read_spectra(idx)

                if data is None:
                    inputs = pars
                    data = spec
                else:
                    for k in inputs:
                        inputs[k] = np.concatenate([inputs[k], pars[k]])
                    data = np.concatenate([data, spec])

        pred = networks["derived"].predictions_np(inputs)
        if parser.is_log("derived"):
            pred = 10.0 ** pred

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                 gridspec_kw={"hspace": 0.03})

        ax = axes[0]

        diff = [((pred - data) / data)[:, n]
                for n, _ in enumerate(parser.computed_parameters)]

        conf = np.tile(np.array([0.68, 0.95])[np.newaxis, :],
                       (len(parser.computed_parameters), 1))

        ax.boxplot(diff, conf_intervals=conf)
        ax.axhline(0.0, c="r", lw=1, ls="--")
        ax.set_ylabel(r"$( x_\mathrm{pred} - x_\mathrm{true} ) \, / \, \
                        x_\mathrm{true}$")
        ax.set_yticklabels([f"{x:,.1%}" for x in ax.get_yticks()])

        ax = axes[1]

        diff = [np.abs((pred - data) / data)[:, n]
                for n, _ in enumerate(parser.computed_parameters)]

        for i, _ in enumerate(parser.computed_parameters):
            diff = np.abs(pred[:, i] - data[:, i]) / data[:, i]

            ax.bar(i + 1, np.percentile(diff, 68.0), alpha=0.9, color=f"C{i}",
                   lw=2)
            ax.bar(i + 1, np.percentile(diff, 95.0), alpha=0.6, color=f"C{i}",
                   lw=1)
            ax.bar(i + 1, np.percentile(diff, 99.0), alpha=0.3, color=f"C{i}",
                   lw=1)

        ax.semilogy()
        ax.set_ylabel(r"$| x_\mathrm{pred} - x_\mathrm{true} | \, / \, \
                        x_\mathrm{true}$")

        ax.set_xticks(1 + np.arange(len(parser.computed_parameters)))
        ax.set_xticklabels(parser.computed_parameters)

        if args.showfig:
            plt.show()
        else:
            filename = os.path.join(parser.path, "derived_validation.pdf")
            plt.savefig(filename, bbox_inches="tight")


def train_networks(args: Optional[list] = None) -> None:
    """
    When you run
        [python -m] cosmopower train <yamlfile>
    this should train the networks based on the given yamlfile.
    """

    # Parse the command line arguments.
    import argparse
    argp = argparse.ArgumentParser(prog="cosmopower train",
                                   description="Train a cosmopower network \
                                                from generated spectra.")
    argp.add_argument("yamlfile", type=str, help="The .yaml file to parse.")
    argp.add_argument("-d", "--dir", dest="root_dir", type=str,
                      help="A root directory to work in.")
    argp.add_argument("-g", "--gpu", dest="use_gpu", action="store_true",
                      help="Force the usage of a GPU.")
    argp.add_argument("-c", "--cpu", dest="use_cpu", action="store_true",
                      help="Force the usage of a CPU.")
    argp.add_argument("-f", "--force", dest="force", action="store_true",
                      help="If set, overwrite existing networks (skip already \
                            existing networks if not).")
    argp.add_argument("-q", "--quantity", dest="quantities", type=str,
                      default=None, help="A list of comma-separated \
                                          quantities that you want to train. \
                                          If not set, all quantities will be \
                                          trained. Example: `-q Cl/tt,Cl/te` \
                                          will train only Cl/tt and Cl/te.")

    args = argp.parse_args(args)

    if args.use_gpu and args.use_cpu:
        raise Exception("The --gpu and --cpu flags are mutually exclusive.")

    device_name = ""
    if args.use_gpu:
        device_name = "/GPU:0"
    if args.use_cpu:
        device_name = "/device:CPU:0"

    parser = YAMLParser(args.yamlfile, root_dir=args.root_dir)

    os.makedirs(os.path.join(parser.path, "networks"), exist_ok=True)

    quantities = None
    if args.quantities is not None:
        quantities = args.quantities.split(",")
    else:
        quantities = parser.quantities

    for quantity in quantities:
        network_settings = parser.settings(quantity)
        network_type = network_settings.get("type")

        if network_type == "NN":
            train_network_NN(parser, quantity, device=device_name,
                             overwrite=args.force)
        elif network_type == "PCAplusNN":
            train_network_PCAplusNN(parser, quantity, device=device_name,
                                    overwrite=args.force)
        else:
            raise ValueError(f"Unknown network type '{network_type}' [should \
                               be of NN or PCAplusNN type].")
