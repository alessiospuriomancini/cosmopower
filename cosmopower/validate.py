import os
import numpy as np
from .dataset import Dataset
from .parser import YAMLParser
from .util import get_noise_curves_CVL, get_noise_curves_SO


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


def show_validation(args: list = None) -> None:
    """
    When you run
        [python -m] cosmopower show-validation <yamlfile>
    this should make a plot of all validation spectra, compute the cosmopower
    prediction and then plot the differences
    """

    import argparse
    argp = argparse.ArgumentParser(prog="cosmopower show-validation",
                                   description="Plot the prediction \
                                                residuals.")
    argp.add_argument("yamlfile", type=str, help="The .yaml file to parse.")
    argp.add_argument("-d", "--dir", dest="root_dir", type=str,
                      help="A root directory to work in.")
    argp.add_argument("-s", "--show", dest="showfig", action="store_true",
                      help="Show the pyplot figure dialog.")
    argp.add_argument("-N", "--noise-curve", dest="noise_curve", default="",
                      type=str, choices=["SO", "CVL"],
                      help="Use a specific \
                            noise curve for comparison, instead of percentual \
                            error. Options: CVL (cosmic variance limit), SO \
                            (Simons Observatory).")
    argp.add_argument("-f", "--fraction", dest="plot_fraction", type=float,
                      default=0.1, help="The fraction of spectra to plot.")
    argp.add_argument("-q", "--quantity", dest="quantities", type=str,
                      help="Which quantities (not `derived`) to plot. \
                            Multiple quantities can be separated by commas. \
                            For example, `Cl/tt,Cl/te` will plot both Cl/tt \
                            and Cl/te only.")
    argp.add_argument("--logx", dest="logx", action="store_true", help="Use \
                      log scaling for x axis.")
    argp.add_argument("--plot-size", dest="plot_size", nargs=2, type=int,
                      default=(8, 6), help="Plot size (width x height, in \
                                            inches).")

    args = argp.parse_args(args)

    import matplotlib.pyplot as plt

    parser = YAMLParser(args.yamlfile, root_dir=args.root_dir)

    networks = parser.restore_networks()

    quantities = (networks.keys() if args.quantities is None else
                  args.quantities.split(","))
    plot_quantities = [q for q in quantities if q != "derived"]

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

    nx = int(np.floor(np.sqrt(len(plot_quantities))))
    ny = -(len(plot_quantities) // -nx)

    fig, axes = plt.subplots(ny, nx, figsize=args.plot_size,
                             gridspec_kw={"hspace": 0.1})

    if args.noise_curve == "CVL":
        print("LOADING COSMIC VARIANCE LIMIT NOISE CURVES.")
        for quantity in networks:
            if quantity == "derived":
                continue

            denominator[quantity] = data[quantity]
            if parser.is_log(quantity):
                denominator[quantity] = np.power(10.0, denominator[quantity])

        denominator = get_noise_curves_CVL(parser, denominator)
    elif args.noise_curve == "SO":
        print("LOADING SO NOISE CURVES.")
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
        print(f"PLOTTING {quantity}.")
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
        j = n // nx

        diff = (prediction - data[quantity]) / data[quantity]
        absdiff = np.abs(prediction - data[quantity]) / denominator[quantity]

        percentiles = np.zeros((6, diff.shape[1]))
        # 1, 2, and 3 sigma bands
        for k in range(3):
            percentiles[k, :] = np.percentile(diff, 100.0 - p[k], axis=0)
            percentiles[k + 3, :] = np.percentile(diff, p[k], axis=0)

        if nx > 1:
            ax = axes[j, i]
        elif ny > 1:
            ax = axes[j]
        else:
            ax = axes

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
        if parser.modes_log(quantity) or args.logx:
            ax.semilogx()
        ax.set_ylim(0.0, None)
        _, ymax = ax.get_ylim()
        if parser.modes_label(quantity) == "l":
            ax.set_xlabel(r"$\ell$")
        else:
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

        if args.noise_curve == "":
            ax.set_yticklabels(["{:,.2%}".format(x) for x in ax.get_yticks()])

    # Delete unused plots
    for n in range(len(plot_quantities), nx * ny):
        i = n % nx
        j = n // nx

        if nx > 1:
            ax = axes[j, i]
        elif ny > 1:
            ax = axes[j]
        else:
            ax = axes

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
        ax.set_yticklabels(["{:,.1%}".format(x) for x in ax.get_yticks()])

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
