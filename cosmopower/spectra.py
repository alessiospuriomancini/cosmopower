from .dataset import Dataset
from .parser import YAMLParser

import os
import sys
import tqdm
import numpy as np
import h5py
from importlib import import_module
from typing import Optional


def setup_path(parser: YAMLParser, args: object) -> bool:
    """
    Checks the given path {path} and creates the directory structure needed to
    generate training spectra there.

    Parameters:
      path -- the path where the data will be generated.
      args -- the command line arguments passed on to the generation script.
    """
    os.makedirs(parser.path, exist_ok=True)

    if not args.force_overwrite:
        with os.scandir(parser.path) as sd:
            for entry in sd:
                if entry.is_file() and not args.resume:
                    raise IOError(f"Directory {parser.path} not empty: check \
                                    directory or run with flag -f to force \
                                    overwrite or -r to resume a previous run!")

    network_path = os.path.join(parser.path, "networks")
    os.makedirs(network_path, exist_ok=True)

    spectra_path = os.path.join(parser.path, "spectra")
    os.makedirs(spectra_path, exist_ok=True)

    with os.scandir(spectra_path) as sd:
        for entry in sd:
            if entry.is_file():
                if not args.force_overwrite and not args.resume:
                    raise IOError(f"Directory {spectra_path} not empty: check \
                                    directory or run with flag -f to force \
                                    overwrite or -r to resume a previous run!")
                elif not args.resume:
                    os.remove(entry)
                    print(f"Overwrote file {entry}.")

    return True


def init_boltzmann_code(parser: YAMLParser) -> dict:
    """
    Given a code {code}, this function will try to import that code and
    initialize it with the given arguments {extra_args}. It returns a
    dictionary containing the code under "code", with any other important
    parameters that are to be handled elsewhere.
    """
    res = {}

    code = parser.boltzmann_code
    extra_args = parser.boltzmann_extra_args

    # Try to import the correct python module for spectrum generation
    try:
        code_module = import_module("cosmopower.spectra_" + code)
        code_init_hook = getattr(code_module, "initialize")
        res = code_init_hook(parser, extra_args)
    except ImportError as e:
        raise Exception(f"Failed to initialize boltzmann code {code}:\n{e}")

    return res


def get_boltzmann_spectra(parser: YAMLParser, state: dict, args: dict = {},
                          quantities: list = [], extra_args: dict = {}
                          ) -> bool:
    """
    Given a state {state} (which can be initialized with
    `init_boltzmann_code`), apply the arguments {args} and create a new state.
    We want to compute the quantities {quantities}.
    """
    try:
        code_module = import_module("cosmopower.spectra_" + state["code"])
        code_spectra_hook = getattr(code_module, "get_spectra")

        if not code_spectra_hook(parser, state, args, quantities, extra_args):
            return False
    except ImportError as e:
        raise Exception(f"Failed to call boltzmann code {state['code']}:\n{e}")

    return True


def new_parameter_lhc(parser: YAMLParser) -> dict:
    """
    Locate the parameter ranges in the parser and get the latin hypercube
    samples.
    """
    import pyDOE

    ranges = {}

    for param in parser.input_parameters:
        xmin, xmax = parser.parameter_value(param)
        ranges[param] = np.linspace(xmin, xmax, parser.nsamples)

    lhc = pyDOE.lhs(len(parser.input_parameters), parser.nsamples,
                    criterion=None)
    idx = (lhc * parser.nsamples).astype(int)

    lhc = {par: ranges[par][idx[:, i]]
           for i, par in enumerate(parser.input_parameters)}

    # Add in the derived parameters.
    lhc = add_derived_params(parser, lhc)

    return lhc


def add_derived_params(parser: YAMLParser, parameters: dict) -> dict:
    for param in parser.derived_parameters:
        if param in parameters:
            continue

        val = parser.parameter_value(param)

        if type(val) is str:
            lambda_function = eval(val)
            inputs = [parameters[i]
                      for i in lambda_function.__code__.co_varnames]
            parameters[param] = lambda_function(*inputs)
        elif type(val) is float:
            parameters[param] = np.tile(val, (parser.nsamples,))

    return parameters


def augment_parameter_lhc(lhc: dict, parser: YAMLParser) -> dict:
    """
    Augment an existing LHC so that it matches the prescription of the parser
    parameters. In the current implementation, the number of samples is
    recursively doubled by adding a second LHC on top of the given LHC, that
    interlaces the existing sample points.
    """
    import pyDOE

    p = list(lhc.keys())[0]
    n = lhc[p].shape[0]

    while n < parser.nsamples:
        lhc_new = {}
        reparse = {}

        for param in (parser.sampled_parameters + parser.input_parameters):
            val = parser.parameter_value(param)

            if type(val) is tuple:
                oldrange = np.sort(lhc[param])
                lhc_new[param] = (oldrange[1:] + oldrange[:-1]) / 2.0
            elif type(val) is float:
                lhc_new[param] = np.tile(val, (n - 1,))
            else:
                reparse[param] = val

        lhc_idx = pyDOE.lhs(len(parser.sampled_parameters), n - 1,
                            criterion=None)
        lhc_idx = (lhc_idx * (n - 1)).astype(int)

        lhc_new = {par: lhc_new[par][lhc_idx[:, i]]
                   for i, par in enumerate(parser.input_parameters)}

        for param in reparse:
            lambda_function = eval(reparse[param])

            inputs = [lhc_new[i] for i in lambda_function.__code__.co_varnames]
            lhc_new[param] = lambda_function(*inputs)

        for param in lhc:
            lhc[param] = np.concatenate((lhc[param], lhc_new[param]))

        n += (n - 1)

    return lhc


def cycle_spectrum_file(parser: YAMLParser, quantity: str,
                        fp: Optional[Dataset], n: int = 0) -> Dataset:
    """
    Cycle the given spectrum file, i.e. open the next file we expect will
    contain spectrum data.
    If fp is None, search the spectrum directory for the first file that can
      contain spectra for quantity.
    If fp is not None, search the spectrum directory for the NEXT file that
      can contain spectra for quantity.
    If the resulting file does not exist, it will automatically create one.
    """
    if fp is None:
        dataset = Dataset(parser, quantity,
                          quantity.replace("/", "_") + f".{n}.hdf5")
    else:
        i = int(fp.filename.split(".")[1]) + 1
        dataset = Dataset(parser, quantity,
                          quantity.replace("/", "_") + f".{i}.hdf5")
        fp.close()
    dataset.open()
    return dataset


def generate_spectra(args: list = None) -> None:
    """
    Hook for the "generate spectra" command.
    Should be called as
        cosmopower generate [flags] <yamlfile>
    Will then open the given yamlfile and generate the spectra based on the
    given description.

    Flags:
        -f --force      Force overwrite of existing files. If this flag is not
                        set, and directories already exist and are not empty,
                        the program will terminate prematurely.
        -r --resume     Resume generation from an existing dataset. If this
                        flag is set, CosmoPower will seek to augment the
                        existing LHC if needed, and add any new sample that
                        does not yet exist for all given quantities.
    """
    import argparse

    argp = argparse.ArgumentParser(prog="cosmopower generate",
                                   description="Generate the spectra needed \
                                                to train the cosmopower \
                                                networks.")
    argp.add_argument("yamlfile", type=str, help="The .yaml file to parse.")
    argp.add_argument("-d", "--dir", dest="root_dir", type=str,
                      help="A root directory to work in.")
    argp.add_argument("-f", "--force", dest="force_overwrite",
                      action="store_true",
                      help="Force spectra generation even if files already \
                            exist.")
    argp.add_argument("-r", "--resume", dest="resume", action="store_true",
                      help="Continue generation if there are existing \
                            spectra.")

    args = argp.parse_args(args)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        n_tot = comm.Get_size()
        Barrier = comm.Barrier
    except:  # noqa E722
        MPI = None
        comm = None
        rank = 0
        n_tot = 1
        Barrier = lambda: -1  # noqa E731

    Barrier()

    parser = YAMLParser(args.yamlfile, root_dir=args.root_dir)

    failed = False
    if rank == 0 and not setup_path(parser, args):
        failed = True

    if comm is not None:
        failed = comm.bcast(failed, root=0)
    if failed:
        if rank == 0:
            raise RuntimeError("Failed to setup path.")

        sys.exit()

    files = {q: None for q in parser.quantities}

    if rank == 0:
        # Find the parameter file.
        param_file = h5py.File(os.path.join(parser.path, "spectra",
                                            "parameters.hdf5"), "a")
        header = param_file.require_group("header")
        Nexisting = int(header.require_dataset("nsamples", (1,), dtype=int,
                                               data=0)[()])
        sampled_parameters = \
            header.require_dataset("sampled_parameters",
                                   (len(parser.sampled_parameters),),
                                   dtype=h5py.special_dtype(vlen=str))

        # First, we want to make a dictionary containing the
        # { param_name : sample_vector } pairs of the LHC. We first check if
        # there are already samples in our dataset, and if so, check if we
        # need to augment the LHC.
        if Nexisting > 0:
            print("Resuming from an existing dataset, validating...")
            # We are resuming from an existing dataset.
            # Check that the number of parameters didn't change.
            assert len(sampled_parameters) == len(parser.sampled_parameters)
            for i, p in enumerate(parser.sampled_parameters):
                # Check that we are using the same parameters still.
                assert sampled_parameters[i].decode() == str(p)

            print("\tHeader looks good to me.")

            data = param_file["data"]

            assert data.shape[0] == Nexisting, \
                   "Header quantity does not match number of samples."
            assert data.shape[1] == len(parser.sampled_parameters), \
                   "Number of parameters per sample does not match header."

            samples = {sampled_parameters[i].decode(): data[:, i]
                       for i, _ in enumerate(parser.sampled_parameters)}

            add_derived_params(parser, samples)

            Nsamples = parser.nsamples
            if Nsamples > Nexisting:
                samples = augment_parameter_lhc(samples, parser)
                p = list(samples.keys())[0]
                data.resize(len(samples[p]), axis=0)
                print(f"\tAugmented LHC from {Nexisting} ==> \
                        {len(samples[p])} samples.")
            else:
                print("\tNot touching the LHC.")
        else:
            print("Generating new dataset.")
            for i, p in enumerate(parser.sampled_parameters):
                sampled_parameters[i] = str(p)

            data = \
                param_file.require_dataset("data",
                                           (parser.nsamples,
                                            len(parser.sampled_parameters)),
                                           dtype=float,
                                           maxshape=  # noqa E251
                                           (None,
                                            len(parser.sampled_parameters)))

            samples = new_parameter_lhc(parser)

        for i, par in enumerate(sampled_parameters):
            data[:, i] = samples[par.decode()]

        for p in parser.computed_parameters:
            if p not in samples:
                samples[p] = np.tile(np.nan, (parser.nsamples,))

        header["nsamples"][()] = data.shape[0]

        param_file.close()
    else:
        samples = None

    first = 0
    last = parser.nsamples
    first_file = 0
    last_file = (last // parser.max_filesize)

    # If mpi-ing, share the data here.
    if comm is not None:
        samples = comm.bcast(samples, root=0)
        # Amount of spectra/files handled by each runner
        cut = (last - first) // n_tot
        fcut = int(np.ceil(float(cut) / parser.max_filesize))

        first, last = first + (rank) * cut, first + (rank + 1) * cut - 1
        first_file, last_file = (first_file + (rank) * fcut,
                                 first_file + (rank + 1) * fcut - 1)

    print(f"[{rank}]: Iterating over samples {first}--{last} in files \
            {first_file}--{last_file}.")

    state = init_boltzmann_code(parser)
    extra_args = parser.boltzmann_extra_args

    accepted = 0
    tbar = tqdm.tqdm(np.arange(first, last + 1))

    Barrier()

    for n in tbar:
        tbar.set_description(("" if MPI is None else f"[{rank}] ")
                             + f"{accepted/n:.1%} success rate")

        boltzmann_params = {k: samples[k][n] for k in parser.boltzmann_inputs}
        quantities_to_be_computed = []

        for q in parser.quantities:
            if files[q] is None:
                # Open first file to read from.
                files[q] = cycle_spectrum_file(parser, q, files[q],
                                               n=first_file)
            while files[q].empty_size == 0 and files[q].indices.max() < n:
                # Current file is full, so we have to cycle to the next one.
                files[q] = cycle_spectrum_file(parser, q, files[q],
                                               n=first_file)

            if n not in files[q].indices:
                quantities_to_be_computed.append(q)

        if len(quantities_to_be_computed) == 0:
            accepted += 1
            continue

        if get_boltzmann_spectra(parser, state, boltzmann_params,
                                 quantities_to_be_computed, extra_args):
            accepted += 1

            for k in state["derived"]:
                samples[k][n] = state["derived"][k]

            for q in quantities_to_be_computed:
                if q == "derived":
                    spec = np.asarray([state["derived"].get(p)
                                       for p in parser.computed_parameters])
                else:
                    spec = state.get(q, None)

                network_params = np.array([
                    samples[k][n] for k in parser.network_input_parameters(q)
                ])

                if spec is None:
                    continue

                if parser.is_log(q):
                    spec = np.log10(spec)

                if np.any(np.isnan(spec)):
                    continue

                if files[q] is None:
                    files[q] = cycle_spectrum_file(parser, q, files[q],
                                                   n=first_file)

                files[q].write_data(n, network_params, spec)

    for q in files:
        if files[q] is not None and files[q].is_open:
            files[q].close()

    if rank == 0:
        print(f"Finished generating {accepted} spectra.")
        print(f"You can now run\n\tcosmopower train {args.yamlfile}\n\
                to train the networks on this dataset.")
