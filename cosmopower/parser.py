import numpy as np
import os
import yaml
from typing import Any, Optional, Sequence

from .cosmopower_PCA import cosmopower_PCA
from .cosmopower_PCAplusNN import cosmopower_PCAplusNN
from .cosmopower_NN import cosmopower_NN

NETWORK_CLASSES = {
  "NN" : cosmopower_NN,
  "PCA" : cosmopower_PCA,
  "PCAplusNN" : cosmopower_PCAplusNN
}

class YAMLParser:
    def __init__(self, filename: str = "", root_dir: Optional[str] = None) -> None:
        """
        Open a YAML package file and do some basic parsing.
        """
        data = { }
        self._filename = filename
        self._root_dir = root_dir or "."

        with open(self.yaml_filename, "r") as fp:
            data = yaml.safe_load(fp)

        self._boltzmann_code = data.get("emulated_code")

        self._samples = data.get("samples", { })
        self._parameters = self._samples.get("parameters", { })
        self._modes = self._samples.get("modes", { })
        self._networks = data.get("networks", { })

        self._network_name = data.get("network_name", "network")
        self._path = data.get("path", ".")

        self._allow_pickle = data.get("allow_pickle", False)
        self._max_filesize = data.get("max_filesize", 10000)

        # TODO: ensure minimal information exists within the data set.
        # e.g. every network should have a number of training samples, ensure that that value exists.
        # determine which values can be safely set to a default value and which ones HAVE to be given.

        # Sort the parameters into three sets:
        # - those that are sampled in the LHC.
        # - those that the parser can derive from an exact equation.
        # - those that we need to compute in the theory code.
        self._lhc_input_parameters = []
        self._derived_parameters = []
        self._theory_computed_parameters = []

        for param in self._parameters:
            if type(self._parameters[param]) in [list,tuple]:
                self._lhc_input_parameters.append(param)
            elif type(self._parameters[param]) in [int,float,str]:
                self._derived_parameters.append(param)
            else:
                self._theory_computed_parameters.append(param)

    def initialize_networks(self) -> dict:
        """
        This function should setup the (untrained) networks and return a dictionary
        with them.
        """
        networks = { }

        for quantity in self.quantities:
            network_type = self.network_type(quantity)

            # TODO: implement PCA+NN instantization.
            if network_type == "PCAplusNN":
                continue

            modes = self.modes(quantity)

            network_class = NETWORK_CLASSES[network_type](
                parameters = self.network_input_parameters(quantity),
                modes = modes
            )

            networks[quantity] = network_class

        return networks

    def restore_networks(self) -> dict:
        """
        This function should look up the pickle (or other) files to restore the
        functional networks from and then return those as a dictionary.
        """
        networks = { }

        for quantity in self.quantities:
            try:
                network_type = self.network_type(quantity)

                network_class = NETWORK_CLASSES[network_type](
                    restore_filename = self.network_path(quantity),
                    allow_pickle = self.allow_pickle
                )

                networks[quantity] = network_class
            except Exception as e:
                print(f"Failed to restore network {quantity}: {str(e)}.")

        return networks

    def settings(self, quantity: str) -> dict:
        result = None

        for network_data in self._networks:
            if network_data["quantity"] == quantity:
                result = network_data.copy()

        if result is None:
            raise KeyError(f"Quantity {quantity} is not defined in {self.yaml_filename}.")

        return result

    def is_log(self, quantity: str) -> bool:
        return self.settings(quantity).get("log", False)

    @property
    def yaml_filename(self) -> str:
        return self._filename

    @property
    def quantities(self) -> Sequence[str]:
        quantities = []

        for network_data in self._networks:
            quantities.append(network_data["quantity"])

        return quantities

    @property
    def base_name(self) -> str:
        return self._network_name

    def network_data(self, quantity) -> dict:
        for network in self._networks:
            if network["quantity"] == quantity:
                return network

        raise KeyError(f"Unknown quantity {quantity}.")

    def network_type(self, quantity) -> str:
        for network in self._networks:
            if network["quantity"] == quantity:
                return network.get("type", "NN")

        raise KeyError(f"Unknown quantity {quantity}.")

    def network_filename(self, quantity) -> str:
        """
        Return the filename for the network. If this is not set, the default is the
        network_name and the network quantity, concatenated by an underscore.
        """
        for network in self._networks:
            if network["quantity"] == quantity:
                return network.get("filename", "_".join([self.base_name, quantity.replace("/", "_")]))

        raise KeyError(f"Network computing {quantity} not found.")

    def network_path(self, quantity) -> str:
        """
        Return the full path to the network for a given quantity.
        """
        return os.path.join(self.path, "networks", self.network_filename(quantity))

    def network_training_parameters(self, quantity) -> dict:
        for network in self._networks:
            if network["quantity"] == quantity:
                return network.get("training", { })

        raise KeyError(f"Unknown quantity {quantity}.")

    @property
    def path(self) -> str:
        return os.path.join(self._root_dir, self._path)

    @property
    def boltzmann_code(self) -> str:
        """
        The emulated boltzmann code.
        """
        return self._boltzmann_code["name"]

    @property
    def boltzmann_path(self) -> Optional[str]:
        """
        The path from which to import the boltzmann code.
        """
        return self._boltzmann_code.get("path", None)

    @property
    def boltzmann_version(self) -> str:
        """
        The version of the used boltzmann code.
        """
        return self._boltzmann_code.get("version", None)

    @property
    def boltzmann_inputs(self) -> Sequence[str]:
        """
        The input parameters given to the boltzmann code.
        """
        return self._boltzmann_code.get("inputs")

    @property
    def boltzmann_extra_args(self) -> dict:
        """
        The extra_args parameter of the boltzmann code.
        """
        return self._boltzmann_code.get("extra_args", {})

    def settings(self, quantity: str) -> dict:
        result = None

        for network_data in self._networks:
            if network_data["quantity"] == quantity:
                result = network_data.copy()

        if result is None:
            raise KeyError(f"Quantity {quantity} is not defined in {self.yaml_filename}.")

        return result

    def is_log(self, quantity: str) -> bool:
        return self.settings(quantity).get("log", False)

    @property
    def yaml_filename(self) -> str:
        return self._filename

    @property
    def quantities(self) -> Sequence[str]:
        quantities = []

        for network_data in self._networks:
            quantities.append(network_data["quantity"])

        return quantities

    def modes(self, quantity) -> np.ndarray:
        """
        The different modes (i.e. the x-axis of each quantity).
        """
        if quantity == "derived":
            return self.computed_parameters

        for network_data in self._networks:
            if network_data["quantity"] == quantity:
                mode_data = network_data["modes"]
                xmin, xmax = mode_data.get("range")
                steps = mode_data.get("steps", None)
                spacing = mode_data.get("spacing", "lin")

                if spacing == "lin":
                    if steps is None:
                        modes = np.arange(xmin, xmax + 1)
                    else:
                        modes = np.linspace(float(xmin), float(xmax), steps)
                elif spacing == "log":
                    modes = np.logspace(np.log10(float(xmin)), np.log10(float(xmax)), steps)
                else:
                    raise ValueError(f"Unknown spacing option {spacing}.")

                return modes

        raise KeyError(f"Unknown quantity {quantity}.")

    def modes_label(self, quantity) -> str:
        """
        The different modes (i.e. the x-axis of each quantity).
        """
        for network_data in self._networks:
            if network_data["quantity"] == quantity:
                mode_data = network_data["modes"]
                return mode_data.get("label")

        raise KeyError(f"Unknown quantity {quantity}.")

    def modes_log(self, quantity) -> bool:
        """
        Whether or not the modes (e.g. the x-axis of the quantity) is log-spaced.
        """
        for network_data in self._networks:
           if network_data["quantity"] == quantity:
               mode_data = network_data["modes"]
               return mode_data.get("spacing", "lin") == "log"

        raise KeyError(f"Unknown quantity {quantity}.")

    @property
    def nsamples(self) -> int:
        """
        The number of (desired) training samples in the network.
        """
        return self._samples.get("Ntraining")

    def parameter_value(self, param) -> Any:
        """
        Obtain the parameter value in a formatted way. Returns a:
        - tuple of (min,max) values if this parameter is an input for the LHC.
        - float if the parameter is a fixed value.
        - str if the parameter is a directly derived value.
        """
        if not param in self._parameters:
            raise KeyError(f"Unknown parameter {param}.")

        par = self._parameters.get(param, None)

        if par is None:
            return None

        if type(par) in [list,tuple]:
            return tuple(par)

        if type(par) in [float,int]:
            return float(par)

        return str(par)

    @property
    def input_parameters(self) -> Sequence[str]:
        """
        The parameters we sampled the LHC over.
        """
        return self._lhc_input_parameters.copy()

    @property
    def derived_parameters(self) -> Sequence[str]:
        """
        A list of all parameters that we derive from the LHC directly.
        """
        return self._derived_parameters.copy()

    @property
    def computed_parameters(self) -> Sequence[str]:
        """
        A list of all parameters that we want to be computed by the theory code.
        """
        return self._theory_computed_parameters.copy()

    @property
    def sampled_parameters(self) -> Sequence[str]:
        """
        All parameters that are included in the parameters file.
        (these are all the parameters that are NOT computed by the theory code).
        """
        return self.input_parameters + self.derived_parameters

    @property
    def all_parameters(self) -> Sequence[str]:
        """
        All parameters (input, derived, and computed).
        """
        return self.input_parameters + self.derived_parameters + self.computed_parameters

    @property
    def allow_pickle(self) -> bool:
        """
        Whether networks are allowed to be loaded as pickle (.pkl) files.
        """
        return self._allow_pickle

    def network_input_parameters(self, quantity) -> Sequence[str]:
        """
        All the INPUT parameters for this particular network.
        If not specified, then the LHC input parameters are used for this
        """

        for network_data in self._networks:
            if network_data["quantity"] == quantity:
                return network_data.get("inputs", self.input_parameters)

        raise KeyError(f"Unknown quantity {quantity}.")

    def get_traits(self, network: str) -> dict:
        """
        Retrieve all the network traits for a certain network.
        """
        traits = { }
        for net in self._networks:
            if net.get("quantity", None) == network:
                for q in net:
                    if q.endswith("_traits"):
                        traits[q] = net[q]

                return traits

        raise KeyError(f"Network {network} does not exist.")

    def get_training_parameters(self, network: str) -> dict:
        """
        Retrieve all the training parameters for a certain network.
        """
        for net in self._networks:
            if net.get("quantity", None) == network:
                return net.get("training", {})

        raise KeyError(f"Network {network} does not exist.")

    @property
    def max_filesize(self) -> int:
        return self._max_filesize
