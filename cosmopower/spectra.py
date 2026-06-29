from .dataset import Dataset
from .parser import YAMLParser

import os
import sys
import tqdm
import numpy as np
from importlib import import_module
from typing import Optional, Tuple
from types import ModuleType


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
    extra_args = parser.boltzmann_extra_args or {}

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


def cycle_spectrum_file(parser: YAMLParser, quantity: str,
                        fp: Optional[Dataset], n: int = 0,
                        suffix: str = "") -> Dataset:
    """
    Cycle the given spectrum file, i.e. open the next file we expect will
    contain spectrum data.
    If fp is None, search the spectrum directory for the first file that can
      contain spectra for quantity.
    If fp is not None, search the spectrum directory for the NEXT file that
      can contain spectra for quantity.
    If the resulting file does not exist, it will automatically create one.
    """
    if suffix != "":
        suffix = "." + suffix

    if fp is None:
        dataset = Dataset(parser, quantity,
                          quantity.replace("/", "_") + f"{suffix}.{n}.hdf5")
    else:
        i = int(fp.filename.split(".")[1]) + 1
        dataset = Dataset(parser, quantity,
                          quantity.replace("/", "_") + f"{suffix}.{i}.hdf5")
        fp.close()
    dataset.open()
    return dataset


def split_samples(MPI: Optional[ModuleType], parser: YAMLParser, samples: dict,
                  nsamples: int) -> Tuple:
    """
    Given a set of samples, check how to split them evenly between the MPI
    processes, and get the range over which this process needs to operate.
    """
    if MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        n_tot = comm.Get_size()
        Barrier = comm.Barrier
    else:
        MPI = None
        comm = None
        rank = 0
        n_tot = 1
        Barrier = lambda: -1  # noqa E731

    first = 0
    last = nsamples-1
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

    return samples, first, last, first_file, last_file


def check_open_files(parser: YAMLParser, files: dict, n: int,
                     first_file: int, suffix: str = "") -> Tuple[dict, list]:
    quantities_to_be_computed = []

    for q in parser.quantities:
        if files[q] is None:
            # Open first file to read from.
            files[q] = cycle_spectrum_file(parser, q, files[q],
                                           n=first_file, suffix=suffix)
        while files[q].empty_size == 0 and files[q].indices.max() < n:
            # Current file is full, so we have to cycle to the next one.
            files[q] = cycle_spectrum_file(parser, q, files[q],
                                           n=first_file, suffix=suffix)

        if n not in files[q].indices:
            quantities_to_be_computed.append(q)

    return files, quantities_to_be_computed


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
        Barrier = comm.Barrier
    except:  # noqa E722
        MPI = None
        comm = None
        rank = 0
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
        training, validation, testing = \
            parser.get_parameter_samples(force_new=args.force_overwrite)
    else:
        training, validation, testing = None, None, None

    # TODO: This is the same code three times and should be made a lot
    #       cleaner!
    training, first, last, first_file, last_file = \
        split_samples(MPI, parser, training, parser.ntraining)

    print(f"[{rank}]: Iterating over training samples {first}--{last} in "
          f"files {first_file}--{last_file}.")

    state = init_boltzmann_code(parser)
    extra_args = parser.boltzmann_extra_args or {}

    accepted = 0
    tbar = tqdm.tqdm(np.arange(first, last + 1))

    Barrier()

    for n in tbar:
        tbar.set_description(("" if MPI is None else f"[{rank}] ")
                             + f"{accepted/max(n, 1):.1%} success rate")

        boltzmann_params = {k: training[k][n] for k in parser.boltzmann_inputs}

        files, quantities_to_be_computed = \
            check_open_files(parser, files, n, first_file)

        if len(quantities_to_be_computed) == 0:
            accepted += 1
            continue

        if get_boltzmann_spectra(parser, state, boltzmann_params,
                                 quantities_to_be_computed, extra_args):
            accepted += 1

            for k in state["derived"]:
                training[k][n] = state["derived"][k]

            for q in quantities_to_be_computed:
                if q == "derived":
                    spec = np.asarray([state["derived"].get(p)
                                       for p in parser.computed_parameters])
                else:
                    spec = state.get(q, None)

                if spec is None:
                    continue

                if parser.is_log(q):
                    spec = np.log10(spec)

                if np.any(np.isnan(spec)):
                    continue

                network_params = np.array([
                    training[k][n] for k in parser.network_input_parameters(q)
                ])

                if files[q] is None:
                    files[q] = cycle_spectrum_file(parser, q, files[q],
                                                   n=first_file)

                files[q].write_data(n, network_params, spec)

    for q in files:
        if files[q] is not None and files[q].is_open:
            files[q].close()
            files[q] = None

    if validation == {} and testing == {}:
        if rank == 0:
            print(f"Finished generating {accepted} spectra.")
            print(f"You can now run\n\tcosmopower train {args.yamlfile}\n"
                  f"to train the networks on this dataset.")
            return

    # Do the exact same thing all over again, but for validation samples.
    validation, first, last, first_file, last_file = \
        split_samples(MPI, parser, validation, parser.nvalidation)

    print(f"[{rank}]: Iterating over validation samples {first}--{last} in "
          f"files {first_file}--{last_file}.")

    accepted = 0
    tbar = tqdm.tqdm(np.arange(first, last + 1))

    Barrier()

    for n in tbar:
        tbar.set_description(("" if MPI is None else f"[{rank}] ")
                             + f"{accepted/n:.1%} success rate")

        boltzmann_params = {
            k: validation[k][n] for k in parser.boltzmann_inputs
        }

        files, quantities_to_be_computed = \
            check_open_files(parser, files, n, first_file, suffix="validation")

        if len(quantities_to_be_computed) == 0:
            accepted += 1
            continue

        if get_boltzmann_spectra(parser, state, boltzmann_params,
                                 quantities_to_be_computed, extra_args):
            accepted += 1

            for k in state["derived"]:
                validation[k][n] = state["derived"][k]

            for q in quantities_to_be_computed:
                if q == "derived":
                    spec = np.asarray([state["derived"].get(p)
                                       for p in parser.computed_parameters])
                else:
                    spec = state.get(q, None)

                if spec is None:
                    continue

                if parser.is_log(q):
                    spec = np.log10(spec)

                if np.any(np.isnan(spec)):
                    continue

                network_params = np.array([
                    validation[k][n]
                    for k in parser.network_input_parameters(q)
                ])

                if files[q] is None:
                    files[q] = cycle_spectrum_file(parser, q, files[q],
                                                   n=first_file,
                                                   suffix="validation")

                files[q].write_data(n, network_params, spec)

    for q in files:
        if files[q] is not None and files[q].is_open:
            files[q].close()
            files[q] = None

    # Do the exact same thing all over again, but for testing samples.
    testing, first, last, first_file, last_file = \
        split_samples(MPI, parser, testing, parser.ntesting)

    print(f"[{rank}]: Iterating over testing samples {first}--{last} in "
          f"files {first_file}--{last_file}.")

    accepted = 0
    tbar = tqdm.tqdm(np.arange(first, last + 1))

    Barrier()

    for n in tbar:
        tbar.set_description(("" if MPI is None else f"[{rank}] ")
                             + f"{accepted/n:.1%} success rate")

        boltzmann_params = {
            k: testing[k][n] for k in parser.boltzmann_inputs
        }

        files, quantities_to_be_computed = \
            check_open_files(parser, files, n, first_file, suffix="testing")

        if len(quantities_to_be_computed) == 0:
            accepted += 1
            continue

        if get_boltzmann_spectra(parser, state, boltzmann_params,
                                 quantities_to_be_computed, extra_args):
            accepted += 1

            for k in state["derived"]:
                testing[k][n] = state["derived"][k]

            for q in quantities_to_be_computed:
                if q == "derived":
                    spec = np.asarray([state["derived"].get(p)
                                       for p in parser.computed_parameters])
                else:
                    spec = state.get(q, None)

                if spec is None:
                    continue

                if parser.is_log(q):
                    spec = np.log10(spec)

                if np.any(np.isnan(spec)):
                    continue

                network_params = np.array([
                    testing[k][n]
                    for k in parser.network_input_parameters(q)
                ])

                if files[q] is None:
                    files[q] = cycle_spectrum_file(parser, q, files[q],
                                                   n=first_file,
                                                   suffix="testing")

                files[q].write_data(n, network_params, spec)

    for q in files:
        if files[q] is not None and files[q].is_open:
            files[q].close()
