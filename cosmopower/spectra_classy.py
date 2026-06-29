from __future__ import annotations
from .parser import YAMLParser
import numpy as np
import classy
from typing import Sequence


def initialize(parser: YAMLParser, extra_args: dict = {}) -> dict:
    state = {}

    for quantity in parser.quantities:
        state[quantity + ".modes"] = parser.modes(quantity)

    if extra_args.get("auto", False):
        extra_args.pop("auto")
        extra_args = classy_args_interpret(parser, extra_args)
        print(f"Auto-generated class settings: {extra_args}")

    state["code"] = "classy"
    state["module"] = classy
    state["params"] = extra_args
    state["derived"] = {k: np.nan for k in parser.computed_parameters}

    return state


def classy_args_interpret(parser: YAMLParser, extra_args: dict = {}) -> dict:
    extra_args["l_max_scalars"] = extra_args.get("l_max_scalars", 0)
    extra_args["l_max_tensors"] = extra_args.get("l_max_tensors", 0)

    extra_args["lensing"] = "no"
    cl_spec = set(extra_args.get("output", "").split(" "))
    cl_modes = set(extra_args.get("modes", "s").split(" "))
    want_cl = False

    for quantity in parser.quantities:
        if quantity == "derived":
            if "sigma8" in parser.computed_parameters:
                cl_spec.add("mPk")

        qpath = quantity.split("/")
        if qpath[0] == "Cl" or qpath[0] == "unlensed_Cl":
            want_cl = True
            if qpath[1] == "tt" or qpath[1] == "te":
                cl_spec.add("tCl")
            if qpath[1] == "te" or qpath[1] == "ee":
                cl_spec.add("pCl")
            if qpath[0] == "Cl" or qpath[1] == "pp":
                extra_args["lensing"] = "yes"
                cl_spec.add("lCl")

            if "s" in cl_modes:
                extra_args["l_max_scalars"] = max(extra_args["l_max_scalars"],
                                                  parser.modes(quantity).max())
            if "t" in cl_modes:
                extra_args["l_max_tensors"] = max(extra_args["l_max_tensors"],
                                                  parser.modes(quantity).max())
        elif qpath[0] == "Pk":
            cl_spec.add("mPk")
            zval = parser.parameter_value("z")
            if type(zval) is float:
                z_max_pk = zval
            elif type(zval) is tuple:
                z_max_pk = max(zval)
            extra_args["z_max_pk"] = max(extra_args.get("z_max_pk", 0.0),
                                         z_max_pk)
            k_max_pk = parser.modes(quantity).max()
            extra_args["P_k_max_1/Mpc"] = max(extra_args.get("P_k_max_1/Mpc",
                                                             1.0),
                                              k_max_pk)

            if qpath[1] == "nonlin":
                extra_args["non_linear"] = "hmcode"
        elif qpath[0] == "sigma8":
            cl_spec.add("mPk")
            z = parser.modes(quantity)
            extra_args["z_max_pk"] = max(extra_args.get("z_max_pk", 0.0),
                                         z.max())
            k_max_pk = 1e-1
            extra_args["P_k_max_1/Mpc"] = max(extra_args.get("P_k_max_1/Mpc",
                                                             1.0),
                                              k_max_pk)

    if "s" not in cl_modes or not want_cl:
        extra_args.pop("l_max_scalars")
    if "t" not in cl_modes or not want_cl:
        extra_args.pop("l_max_tensors")
    extra_args["output"] = " ".join(list(cl_spec))
    extra_args["modes"] = " ".join(list(cl_modes))

    print(extra_args)

    return extra_args


def get_classy_derived(params: dict, cosmo: classy.Class,
                       parameters: Sequence[str]) -> dict:
    res = {}

    for p in parameters:
        if p == "rs_drag":
            res[p] = cosmo.rs_drag()
        elif p == "Omega_nu":
            res[p] = cosmo.Omega_nu
        elif p == "T_cmb":
            res[p] = cosmo.T_cmb()
        else:
            res[p] = cosmo.get_current_derived_parameters([p])[p]

    return res


def get_spectra(parser: YAMLParser, state: dict, args: dict = {},
                quantities: list = [], extra_args: dict = {}) -> bool:

    classy = state["module"]

    if "z" in args:
        z_pk = args.pop("z")

    try:
        cosmo = classy.Class()
        cosmo.set(state["params"] | args)
        cosmo.compute()
    except (classy.CosmoComputationError, classy.CosmoSevereError) as e:
        print(str(e))
        cosmo.struct_cleanup()
        cosmo.empty()
        return False

    state["derived"] = get_classy_derived(state["params"], cosmo,
                                          list(state["derived"].keys()))

    for quantity in quantities:
        qpath = quantity.split("/")

        if qpath[0] == "derived":
            continue
        elif qpath[0] == "Cl":
            spec = qpath[1]

            cls = cosmo.lensed_cl()
            ell = cls["ell"]
            Cl = cls[spec]
            Dl = Cl * (ell * (ell + 1) / (2 * np.pi))

            state[quantity] = Dl[state[quantity + ".modes"]]
        elif qpath[0] == "unlensed_Cl":
            spec = qpath[1]

            cls = cosmo.raw_cl()
            ell = cls["ell"]
            Cl = cls[spec]
            Dl = Cl * (ell * (ell + 1) / (2 * np.pi))

            state[quantity] = Dl[state[quantity + ".modes"]]
        elif qpath[0] == "Pk":
            k = state[quantity + ".modes"]
            spec = qpath[1]

            if spec == "lin":
                Pk = np.zeros_like(k)
                for i in range(len(k)):
                    Pk[i] = cosmo.pk_lin(k[i], z_pk)
            if spec == "nonlin":
                Pk = np.zeros_like(k)
                for i in range(len(k)):
                    Pk[i] = cosmo.pk(k[i], z_pk)

            state[quantity] = Pk
        elif qpath[0] == "DA" or qpath[0] == "angular_diameter_distance":
            z = state[quantity + ".modes"]
            state[quantity] = cosmo.angular_distance(z)
        elif qpath[0] == "Hubble":
            z = state[quantity + ".modes"]
            state[quantity] = cosmo.Hubble(z)
        elif qpath[0] == "sigma8":
            z = state[quantity + ".modes"]
            state[quantity] = cosmo.sigma(8, z)

    cosmo.struct_cleanup()
    cosmo.empty()

    return True
