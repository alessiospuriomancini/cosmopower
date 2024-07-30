"""
.. module:: CosmoPower

:Synopsis: A Cobaya wrapper for CosmoPower.
:Author: Hidde T. Jense
"""

import numpy as np
import scipy.interpolate as itp
import cosmopower as cp
from typing import Any, Iterable, Union
from cobaya.log import LoggedError
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.conventions import Const

H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": Const.c_km_s}


class CosmoPower(BoltzmannBase):
    """A CosmoPower network wrapper for Cobaya."""

    def initialize(self) -> None:
        super().initialize()

        self.log.info(f"Importing CosmoPower from {cp.__file__}.")

        try:
            self._parser = cp.YAMLParser(self.package_file, self.root_dir)
        except Exception as e:
            raise LoggedError(self.log, f"Failed to parse cosmopower package \
                                          file:\n{str(e)}")

        self._networks = self.parser.restore_networks()
        self._networks_to_eval = []

    def must_provide(self, **requirements: dict) -> dict:
        super().must_provide(**requirements)

        self.requires = set([])
        self.provides = set([])
        self._networks_to_eval = set([])
        required_quantities = []
        must_provide = {}

        for k, v in self._must_provide.items():
            if k == "Cl":
                for a in v:
                    required_quantities.append(f"Cl/{a.lower()}")
            else:
                required_quantities.append(f"{k}" if v is not None else k)

        for q in required_quantities:
            if q in ["Hubble", "H", "h"]:
                found_Hubble = False

                for n in self.networks:
                    if n in ["Hubble", "H", "h"]:
                        found_Hubble = True
                        self._networks_to_eval.add(n)
                        self.provides.add(q)
                        for param in self.networks[n].parameters:
                            p = self.find_translation(param)
                            must_provide[p] = None
                            self.requires.add(p)

                if not found_Hubble:
                    raise LoggedError(self.log, f"Cannot provide {q}.")

                continue

            if q in self.networks:
                self._networks_to_eval.add(q)
                self.provides.add(q)

                for param in self.networks[q].parameters:
                    p = self.find_translation(param)
                    must_provide[p] = None
                    self.requires.add(p)

                continue
            elif q == "fsigma8":
                # If we do not have an fsigma8 network, we *could* derive
                # it from sigma8z instead.
                if "sigma8z" in self.networks:
                    self.log.debug("fsigma8 is not provided, but sigma8(z) \
                                    is.")
                    self._networks_to_eval.add("sigma8z")
                    self.provides.add(q)
                    for param in self.networks["derived"].parameters:
                        p = self.find_translation(param)
                        must_provide[p] = None
                        self.requires.add(p)
            elif "derived" in self.networks:
                if q in self.parser.computed_parameters or \
                   self.translate_param(q) in self.parser.computed_parameters:
                    self.log.debug(f"Can compute {q} as derived parameter.")
                    self._networks_to_eval.add("derived")
                    self.provides.add(q)
                    for param in self.networks["derived"].parameters:
                        p = self.find_translation(param)
                        must_provide[p] = None
                        self.requires.add(p)

                    continue

        if len(self._networks_to_eval) == 0:
            raise RuntimeError("No networks are to be evaluated (maybe none \
                                were loaded?).")

        self.log.debug(f"Will evaluate networks {self._networks_to_eval}")

        return must_provide

    def calculate(self, state: dict, want_derived: bool = True,
                  **params: dict) -> bool:
        input_params = {
            **{p: [params[p]] for p in params},
            **{self.translate_param(p): [params[p]] for p in params}
        }

        if "derived" not in state:
            state["derived"] = {}

        # We need to explicitly compute derived parameters beforehand, in case
        # we need a derived parameter as a network input.
        if "derived" in self._networks_to_eval:
            network = self.networks["derived"]

            used_params = {p: input_params[p] for p in network.parameters}
            outputs = {}

            if self.parser.is_log("derived"):
                data = network.ten_to_predictions_np(used_params)[0, :]
            else:
                data = network.predictions_np(used_params)[0, :]

            for i, param in enumerate(self.parser.computed_parameters):
                input_params[param] = data[i]
                p = (self.find_translation(param)
                     if self.find_translation(param) in self.provides
                     else param)
                outputs[p] = data[i]
                state["derived"][p] = data[i]

            self.log.debug(f"Computed parameters {outputs}.")

        for quantity in self._networks_to_eval:
            if quantity == "derived":
                continue

            network = self.networks[quantity]

            used_params = {p: input_params[p] for p in network.parameters}

            if self.parser.is_log(quantity):
                data = network.ten_to_predictions_np(used_params)[0, :]
            else:
                data = network.predictions_np(used_params)[0, :]

            self.set_in_dict(state, quantity, data)
            if self.parser.modes_label(quantity) == "l":
                state["ell"] = self.parser.modes(quantity)
            elif self.parser.modes_label(quantity) == "z":
                state["z"] = self.parser.modes(quantity)
            elif self.parser.modes_label(quantity) == "k":
                state["k"] = self.parser.modes(quantity)

        return True

    def get_param(self, p: str) -> float:
        if p in self.current_state["derived"]:
            return self.current_state["derived"][p]

        return self.current_state["derived"][self.translate_param(p)]

    def get_Hubble(self, z: Union[float, np.ndarray],
                   units: str = "km/s/Mpc") -> np.ndarray:
        # First we need to find the H(z) function we can provide.
        hz = None
        prefac = H_units_conv_factor[units]
        for quantity in self.current_state:
            # H(z) in units of km/s/Mpc
            if quantity == "Hubble":
                prefac /= H_units_conv_factor["km/s/Mpc"]
                hz = {"h": self.current_state[quantity],
                      "z": self.current_state["z"]}
                break
            # H(z) in units of 100*km/s/Mpc
            if quantity == "h":
                prefac /= H_units_conv_factor["km/s/Mpc"] / 100.0
                hz = {"h": self.current_state[quantity],
                      "z": self.current_state["z"]}
                break
            # H(z) in units of 1/Mpc
            if quantity == "H":
                prefac /= H_units_conv_factor["1/Mpc"]
                hz = {"h": self.current_state[quantity],
                      "z": self.current_state["z"]}
                break
        if hz is None:
            raise LoggedError(self.log, "Hubble parameter not computed.")

        hz = itp.interp1d(hz["z"], hz["h"], fill_value="extrapolate",
                          bounds_error=None)
        return prefac * np.atleast_1d(hz(z))

    def get_angular_diameter_distance(self, z: Union[float, np.ndarray]
                                      ) -> np.ndarray:
        # First we need to find the DA(z) function we can provide.
        daz = None
        if "angular_diameter_distance" in self.current_state:
            daz = itp.interp1d(self.current_state["z"],
                               self.current_state["angular_diameter_distance"],
                               fill_value="extrapolate", bounds_error=None)
            return np.atleast_1d(daz(z))

        raise LoggedError(self.log, "Angular diameter distance not computed.")

    def get_sigma8(self, z: Union[float, np.ndarray]) -> np.ndarray:
        s8z = None

        if "sigma8z" in self.current_state:
            s8z = itp.interp1d(self.current_state["z"],
                               self.current_state["sigma8"],
                               fill_value="extrapolate", bounds_error=None)
            return np.atleast_1d(s8z(z))

        raise LoggedError(self.log, "sigma8(z) not computed.")

    def get_fsigma8(self, z: Union[float, np.ndarray]) -> np.ndarray:
        fs8z = None

        if "fsigma8" in self.current_state:
            fs8z = itp.interp1d(self.current_state["z"],
                                self.current_state["fsigma8"],
                                fill_value="extrapolate",
                                bounds_error=None)
            return np.atleast_1d(fs8z(z))
        elif "sigma8z" in self.current_state:
            fs8z = -((1.0 + self.current_state["z"])
                     * np.gradient(self.current_state["sigma8z"],
                                   self.current_state["z"]))
            fs8z = itp.interp1d(self.current_state["z"], fs8z,
                                fill_value="extrapolate",
                                bounds_error=None)
            return np.atleast_1d(fs8z(z))

        raise LoggedError(self.log, "Need either fsigma8(z) or sigma8(z) to \
                                     compute fsigma8, but neither are \
                                     computed.")

    def get_Cl(self, ell_factor: bool = False,
               units: str = "FIRASmuK2") -> dict:
        cls_old = self.current_state.copy()

        if "ell" not in cls_old:
            raise Exception(f"Parser file {self.parser.yaml_filename} does \
                              not have l modes defined.")

        lmax = self.extra_args["lmax"] or cls_old["ell"].max()
        cls = {"ell": np.arange(lmax + 1).astype(int)}
        ls = cls_old["ell"]

        for k in cls_old["Cl"]:
            cls[k] = np.tile(np.nan, cls["ell"].shape)

        for k in cls_old["Cl"]:
            # TODO: check for ell factor.
            prefac = np.ones_like(ls).astype(float)
            if self.parser.settings(f"Cl/{k}").get("ell_factor", True):
                prefac /= cp.util.ell_factor(ls, k)
            if ell_factor:
                prefac *= cp.util.ell_factor(ls, k)

            cls[k][ls] = (cls_old["Cl"][k] * prefac
                          * cp.util.cmb_unit_factor(k, units, 2.7255))
            cls[k][:2] = 0.0

        return cls

    def get_in_dict(self, dct: dict, path: str) -> Any:
        """
        Networks are labeled with paths like "Cl/tt". This function parses
        such a path and finds the correct value in a dictionary.

        E.g.
        In the case for
            get_in_dict(state, "Cl/tt")
        it would return state["Cl"]["tt"], provided that exists.
        """
        path = path.split("/")
        tgt = dct

        while len(path) > 1:
            if path[0] not in tgt:
                return None
            tgt = tgt.get(path.pop(0))

        return tgt.get(path[0], None)

    def set_in_dict(self, dct: dict, path: str, value: Any) -> dict:
        """
        Networks are labeled with paths like "Cl/tt". This function parses
        such a path and finds the correct value in a dictionary.

        E.g.
        In the case for
            set_in_dict(state, "Cl/tt", cltt)
        it would set state["Cl"]["tt"] = cltt
        and ensure that the correct entries (sub-dictionaries) are created.
        """
        path = path.split("/")
        tgt = dct

        while len(path) > 1:
            new = tgt.get(path[0], {})
            tgt[path.pop(0)] = new
            tgt = new

        tgt[path[0]] = value

        return dct

    def get_can_provide(self) -> Iterable[str]:
        can_provide = {}

        for network in self._networks:
            if network == "derived":
                continue
            self.set_in_dict(can_provide, network,
                             self.networks[network].modes.max())

        return can_provide

    def get_can_provide_params(self) -> Iterable[str]:
        if "derived" in self.networks:
            params = set(self.parser.computed_parameters.copy())
            for p in self.parser.computed_parameters:
                if p != self.find_translation(p):
                    params.add(self.find_translation(p))

            return params

        return []

    def find_translation(self, p: str) -> str:
        """Checks if a parameter can be renamed, and returns the source param
        if so. In other words, if find_translation(p) == q, then
        translate_param(q) = p."""
        for k, v in self.renames.items():
            if v == p:
                return k
        return p

    def translate_param(self, p: str) -> str:
        """Checks if a parameter needs to be renamed, and returns the renamed
        param if so."""
        return self.renames.get(p, p)

    @property
    def networks(self) -> dict:
        return self._networks

    @property
    def parser(self) -> cp.YAMLParser:
        """The yaml parser used by this wrapper."""
        return self._parser
