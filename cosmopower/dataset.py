from __future__ import annotations
from .parser import YAMLParser
import h5py
import os
import numpy as np
from typing import Optional, Tuple, Union


class Dataset:
    """
    A Dataset is a utility wrapper for accessing CosmoPower training data.
    Using a dataset allows easier access to the data compared to opening the
    raw HDF5 file yourself, while also providing some quick functions to
    ensure proper formatting of the file for the user.

    If you want to read/write your own CosmoPower training data using a
    Dataset class, the best way to do so is as::

        dataset = Dataset(parser, quantity, filename)
        with dataset:
            for spectrum in generator:
                dataset.write_data(index, parameters, spectrum)

    Where the ``with``-statement ensures the dataset is properly opened,
    resized, closed, and cleaned up. You can also manually open and close
    a Dataset::

        dataset = Dataset(parser, quantity, filename)
        dataset.open()
        for spectrum in generator:
            dataset.write_data(index, parameters, spectrum)
        dataset.close()

    Which has an equivalent behaviour. Note however that the
    ``dataset.close()`` function can have some significant overhead. It is
    recommended not to open and close your dataset in a loop, but instead
    loop within your with-statement.
    """
    def __init__(self, parser: YAMLParser,
                 quantity: str, filename: str) -> None:
        """Create a new dataset wrapper.

        :param parser: The YAMLParser associated with this dataset.
        :param quantity: The quantity that we are saving in this dataset.
        :param filename: The filename (excluding path) for this dataset."""
        self._fp = None
        self._parser = parser
        self._quantity = quantity
        self._filename = filename
        self._first_empty = 0

    def open(self) -> None:
        """Open the associated HDF5 file."""
        if os.path.exists(self.filepath):
            self._fp = h5py.File(self.filepath, "a")
            self.restore_from_file()
        else:
            self._fp = h5py.File(self.filepath, "w")
            self.initialize_file()

    def close(self) -> None:
        """Close the associated HDF5 file."""
        self.check_open()

        # Shrink to fit
        size = self.size
        self.file["data"]["indices"].resize(size, axis=0)
        self.file["data"]["parameters"].resize(size, axis=0)
        self.file["data"]["spectra"].resize(size, axis=0)

        self.file.close()
        self._fp = None

    def initialize_file(self) -> None:
        """Initialize the current file pointer as an empty file."""
        self.check_open()

        header = self.file.require_group("header")
        header["modes"] = self.parser.modes(self.quantity)
        header["parameters"] = self.parser.network_input_parameters(
            self.quantity)
        header["quantity"] = self.quantity

        nparams = len(header["parameters"])

        data = self.file.require_group("data")
        data.create_dataset("indices", (self.max_size,),
                            maxshape=(self.max_size,), dtype=int)
        data["indices"][:] = -1
        data.create_dataset("parameters", (self.max_size, nparams),
                            maxshape=(self.max_size, nparams))
        data.create_dataset("spectra", (self.max_size,
                                        len(self.parser.modes(self.quantity))),
                            maxshape=(self.max_size,
                                      len(self.parser.modes(self.quantity))))

    def restore_from_file(self) -> None:
        """Restore the current dataset from an existing file pointer."""
        self.check_open()

        if self.file["data"]["indices"].shape[0] < self.max_size:
            oldsize = self.file["data"]["indices"].shape[0]
            self.file["data"]["indices"].resize(self.max_size, axis=0)
            self.file["data"]["indices"][oldsize:] = -1
        if self.file["data"]["parameters"].shape[0] < self.max_size:
            self.file["data"]["parameters"].resize(self.max_size, axis=0)
        if self.file["data"]["spectra"].shape[0] < self.max_size:
            self.file["data"]["spectra"].resize(self.max_size, axis=0)

    def write_data(self, indices: Union[int, np.ndarray],
                   parameters: np.ndarray, spectra: np.ndarray,
                   overwrite: bool = False) -> int:
        """Write parameters & spectra to the dataset.

           Keyword arguments:
           :param indices: the index/-ices to write to.
           :param parameters: the input parameters for the network.
           :param spectra: the output spectra for the network.
           :param overwrite: allow overwriting existing data. If false, raise
                             an exception when conflicting indices are found.

           :return: the number of indices written to the file."""
        self.check_open()

        indices = np.atleast_1d([indices,])
        parameters = np.atleast_2d(parameters)
        spectra = np.atleast_2d(spectra)

        if indices.shape[0] != parameters.shape[0]:
            raise ValueError(f"Number of indices {indices.shape[0]} does not \
            match number of parameter entries {parameters.shape[0]}.")
        if indices.shape[0] != spectra.shape[0]:
            raise ValueError(f"Number of indices {indices.shape[0]} does not \
            match number of spectra entries {spectra.shape[0]}.")

        i = self._first_empty

        for n, idx in enumerate(indices):
            # Find empty slot.
            while self.file["data"]["indices"][i] != -1:
                i += 1
                if i == self.file["data"]["indices"].shape[0]:
                    return n

            self.file["data"]["indices"][i] = idx
            self.file["data"]["parameters"][i, :] = parameters[n, :]
            self.file["data"]["spectra"][i, :] = spectra[n, :]
            i += 1

        self._first_empty = i

        return len(indices)

    @property
    def indices(self) -> np.ndarray:
        """All valid (non-negative) indices in this dataset.

        :return: All valid (non-negative) indices in this dataset."""
        self.check_open()
        indices = self.file["data"]["indices"][:]
        return indices[indices != -1]

    def read_parameters(self, indices: Union[int, np.ndarray]) -> dict:
        """Read the parameters with given indices.

        :param indices: The index/indices you want read from.
        :return: The parameters associated with the requested indices."""
        self.check_open()

        cast_1d = (type(indices) == int)
        indices = np.atleast_1d(indices)

        entries = np.where(np.in1d(self.file["data"]["indices"][:], indices))[0]

        parameters = {
            k.decode(): self.file["data"]["parameters"][entries, i]
            for i, k in enumerate(self.file["header"]["parameters"])
        }

        if cast_1d:
            parameters = {k: float(v) for k, v in parameters.items()}

        return parameters

    def read_spectra(self, indices: Union[int, np.ndarray]) -> np.ndarray:
        """Read the spectra with given indices.

        :param indices: The index/indices you want read from.
        :return: The spectra associated with the requested indices."""
        self.check_open()

        cast_1d = (type(indices) == int)
        indices = np.atleast_1d(indices)

        entries = np.where(np.in1d(self.file["data"]["indices"][:], indices))[0]
        spec = self.file["data"]["spectra"][entries, :]
        if cast_1d:
            spec = spec[0, :]

        return spec

    def read_data(self) -> Tuple[np.ndarray,np.ndarray]:
        """Read the ENTIRE file, returning a (parameters, spectra) tuple of
           numpy arrays of generated spectra."""
        self.check_open()
        entries = np.where(self.file["data"]["indices"][:] != -1)[0]

        parameters = self.file["data"]["parameters"][entries,:]
        spectra = self.file["data"]["spectra"][entries,:]

        return parameters, spectra

    def __enter__(self) -> Dataset:
        """Allows for with-statements with datasets."""
        self.open()
        return self

    def __exit__(self, ex_type, ex_val, ex_tb) -> None:
        """Allows for with-statements with datasets."""
        self.close()

    @property
    def filename(self) -> str:
        """The file name for this dataset."""
        return self._filename

    @property
    def filepath(self) -> str:
        """The full path to this dataset."""
        return os.path.join(self.parser.path, "spectra", self.filename)

    @property
    def file(self) -> Optional[h5py.File]:
        """The h5py File wrapper for this dataset."""
        return self._fp

    @property
    def parser(self) -> YAMLParser:
        """The YAMLParser for this dataset."""
        return self._parser

    @property
    def quantity(self) -> str:
        """The quantity associated with this dataset."""
        return self._quantity

    @property
    def max_size(self) -> int:
        """The maximum file size for this dataset (retrieved from parser)."""
        return self.parser.max_filesize

    @property
    def modes(self) -> np.ndarray:
        """The modes saved to this dataset."""
        self.check_open()
        return self.file["header"]["modes"][:]

    @property
    def is_open(self) -> bool:
        """Check whether this file is open."""
        return self.file is not None

    def check_open(self) -> None:
        """Check whether the file is open, excepting when not."""
        if self.file is None:
            raise RuntimeError("Not open.")

    @property
    def empty_size(self) -> int:
        """Number of empty indices in the file."""
        self.check_open()
        return self.max_size - self.size

    @property
    def size(self) -> int:
        """Number of non-empty indices in the file."""
        self.check_open()
        indices = self.file["data"]["indices"][:]
        return sum(indices != -1)

    @property
    def is_empty(self) -> bool:
        """Whether there are zero entries in this dataset."""
        self.check_open()
        return self.size == 0

    @property
    def is_full(self) -> bool:
        """Whether this dataset is full or not."""
        self.check_open()
        return self.size == self.max_size
