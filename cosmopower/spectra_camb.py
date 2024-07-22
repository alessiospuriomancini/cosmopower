from __future__ import annotations
from .parser import YAMLParser
import numpy as np
import scipy.interpolate as itp
import sys
from types import ModuleType
from typing import Sequence
from packaging.version import parse as vparse
import warnings

def initialize(parser: YAMLParser, extra_args: dict = {}) -> dict:
    if parser.boltzmann_path is not None:
        sys.path.insert(0, parser.boltzmann_path)
    
    import camb
    
    version_desired = vparse(parser.boltzmann_version)
    version_received = vparse(camb.__version__)
    
    if version_desired != version_received:
        if version_desired.major == version_received.major and version_desired.minor == version_received.minor:
            warnings.warn(f"Different camb version: camb version {version_desired} was requested, but you imported version {version_received}.")
        else:
            raise ImportError(f"Incompatible camb version: camb version {version_desired} was requested, but you imported version {version_received}.")
    
    res = {}
    
    extra_args["lmax"] = extra_args.get("lmax", 2)
    extra_args["kmax"] = extra_args.get("kmax", 1.0)
    extra_args["redshifts"] = np.array([ 0.0, ])
    res["requested_quantities"] = parser.computed_parameters
    
    for quantity in parser.quantities:
        if quantity == "derived":
            continue
        
        res["requested_quantities"].append(quantity)
        
        # Check that the maximum values for mode ranges matches the requested outputs.
        if parser.modes_label(quantity) == "l" and extra_args["lmax"] < parser.modes(quantity).max():
            extra_args["lmax"] = parser.modes(quantity).max()
        
        if parser.modes_label(quantity) == "k" and extra_args["kmax"] < parser.modes(quantity).max():
            extra_args["kmax"] = parser.modes(quantity).max()
        
        if parser.modes_label(quantity) == "z":
            z1 = extra_args["redshifts"]
            z2 = parser.modes(quantity)
            extra_args["redshifts"] = np.sort(np.unique(np.concatenate((z1, z2), 0)))
        
        res[quantity + ".modes"] = parser.modes(quantity)
    
    res["code"] = "camb"
    res["module"] = camb
    res["version"] = camb.__version__
    res["params"] = camb.set_params(**extra_args)
    res["derived"] = { k : np.nan for k in parser.computed_parameters }
    
    return res

def get_camb_derived(pars: camb.CAMBparams, results: camb.CAMBdata, parameters: Sequence[str]) -> dict:
    res = {}
    
    cambDerived = results.get_derived_params()
    
    for p in parameters:
        if p in cambDerived:
            res[p] = cambDerived[p]
        elif p == "sigma8":
            res[p] = results.get_sigma8_0()
        elif p == "As":
            res[p] = results.Params.InitPower.As
        else:
            try:
                res[p] = getattr(pars, p)
            except AttributeError:
                try:
                    res[p] = getattr(results, p)
                except AttributeError:
                    res[p] = getattr(pars, "get_" + p, lambda: None)()
    
    return res

def get_camb_Hubble(pars: camb.CAMBparams, results: camb.CAMBdata, redshifts: Sequence[float]) -> np.ndarray:
    return results.hubble_parameter(redshifts) # km/s/Mpc units

def get_camb_H(pars: camb.CAMBparams, results: camb.CAMBdata, redshifts: Sequence[float]) -> np.ndarray:
    return results.h_of_z(redshifts) # /Mpc units

def get_camb_sigma8(pars: camb.CAMBparams, results: camb.CAMBdata, redshifts: Sequence[float]) -> np.ndarray:
    s8 = results.get_sigma8()
    z = results.Params.Transfer.PK_redshifts
    return itp.interp1d(z, s8)(redshifts)

def get_camb_angular_diameter_distance(pars: camb.CAMBparams, results: camb.CAMBdata, redshifts: Sequence[float]) -> np.ndarray:
    return results.get_angular_diameter_distance(redshifts)

def get_camb_Omega(pars: camb.CAMBparams, results: camb.CAMBdata, redshifts: Sequence[float], var: str) -> np.ndarray:
    if var == "m":
        return results.get_Omega(var = "baryon", z = redshifts) + results.get_Omega(var = "cdm", z = redshifts)

    return results.get_Omega(var = var, z = redshifts)

def get_camb_cmb_power_spectra(pars: camb.CAMBparams, results: camb.CAMBdata, ls: np.ndarray, spectra: str) -> np.ndarray:
    spec_indices = { "tt" : 0, "te" : 3, "ee" : 1, "bb" : 2 }
    lens_indices = { "pp" : 0, "pt" : 1, "pe" : 2 }
    idx = spec_indices.get(spectra, None)
    
    if idx is None:
        # check for lensing
        if spectra in lens_indices:
            idx = lens_indices.get(spectra, None)
            
            lens_spectra = results.get_lens_potential_cls(CMB_unit = None)
            
            return lens_spectra[ls,idx]
        
        return np.zeros((params.max_l))
    
    cmb_spectra = results.get_cmb_power_spectra(CMB_unit = None)["total"]
    
    return cmb_spectra[ls,idx]

def get_camb_linear_matter_power(camb: ModuleType, pars: camb.CAMBparams, results: camb.CAMBdata, k: np.ndarray, z: np.ndarray, kmax: float) -> np.ndarray:
    if pars.NonLinear != camb.model.NonLinear_none:
        pars.NonLinear = camb.model.NonLinear_none
        pars.set_matter_power(redshifts = z, kmax = kmax)
        results = camb.get_results(pars)
    
    _, _, pk = results.get_matter_power_spectrum(minkh = k.min() / (pars.H0 / 100.0), maxkh = k.max() / (pars.H0 / 100.0), npoints = len(k))
    
    return pk[0,:]

def get_camb_nonlinear_matter_power(camb: ModuleType, pars: camb.CAMBparams, results: camb.CAMBdata, k: np.ndarray, z: np.ndarray, kmax: float) -> np.ndarray:
    
    if pars.NonLinear != camb.model.NonLinear_both:
        pars.NonLinear = camb.model.NonLinear_both
        pars.set_matter_power(redshifts = z, kmax = kmax)
        results = camb.get_results(pars)

    _, _, pk = results.get_matter_power_spectrum(minkh = k.min() / (pars.H0 / 100.0), maxkh = k.max() / (pars.H0 / 100.0), npoints = len(k))

    return pk[0,:]

def get_camb_nonlinear_matter_power_boost(camb: ModuleType, pars: camb.CAMBparams, results: camb.CAMBdata, k: np.ndarray, z: np.ndarray, kmax: float) -> np.ndarray:
    pk_lin = get_camb_linear_matter_power(pars, results, k, z, kmax)
    
    if pars.NonLinear != camb.model.NonLinear_both:
        pars.NonLinear = camb.model.NonLinear_both
        pars.set_matter_power(redshifts = z, kmax = kmax)
        results = camb.get_results(pars)
    
    _, _, pk = results.get_matter_power_spectrum(minkh = k.min() / (pars.H0 / 100.0), maxkh = k.max() / (pars.H0 / 100.0), npoints = len(k))

    return pk[0,:] / pk_lin - 1.0

def get_spectra(parser: YAMLParser, state: dict, args: dict = {}, quantities: list = [], extra_args: dict = {}) -> bool:
    camb = state["module"]
    
    try:
        state["params"] = camb.set_params(cp = state["params"], **translate_camb_params(**args))
        state["results"] = camb.get_results(state["params"])
    except (camb.CAMBError, camb.CAMBFortranError) as e:
        return False
    
    # TODO We can probably do something like
    # - initialize function -> store in the state dict what results we want to obtain
    # - calculate function -> just save everything we requested in sequential order
    for quantity in quantities:
        qpath = quantity.split("/")
        
        state["derived"] = get_camb_derived(state["params"], state["results"], list(state["derived"].keys()))
        
        if qpath[0] == "derived":
            continue
        elif qpath[0] == "Cl":
            # C_ells
            state[quantity] = get_camb_cmb_power_spectra(state["params"], state["results"], state[quantity + ".modes"], qpath[1])
        elif qpath[0] == "Pk" and qpath[1] == "lin":
            # Linear matter power
            state[quantity] = get_camb_linear_matter_power(camb, state["params"], state["results"], state[quantity + ".modes"], np.asarray([args.get("z", 0.0)]), extra_args.get("kmax", 1.0))
        elif qpath[0] == "Pk" and qpath[1] == "nonlin":
            # Non-linear matter power
            state[quantity] = get_camb_nonlinear_matter_power(camb, state["params"], state["results"], state[quantity + ".modes"], np.asarray([args.get("z", 0.0)]), extra_args.get("kmax", 1.0))
        elif qpath[0] == "Pk" and qpath[1] == "nlboost":
            # Non-linear matter power
            state[quantity] = get_camb_nonlinear_matter_power_boost(camb, state["params"], state["results"], state[quantity + ".modes"], np.asarray([args.get("z", 0.0)]), extra_args.get("kmax", 1.0))
        elif qpath[0] in ["Hubble","Hubble_z"]:
            # Hubble parameter [in km/s/Mpc]
            state[quantity] = get_camb_Hubble(state["params"], state["results"], parser.modes(qpath[0]))
        elif qpath[0] in ["h","h_z"]:
            # Hubble parameter [in 1/100/Mpc]
            state[quantity] = get_camb_Hubble(state["params"], state["results"], parser.modes(qpath[0])) / 100.0
        elif qpath[0] in ["sigma8","sigma8_z"]:
            # sigma8(z)
            state[quantity] = get_camb_sigma8(state["params"], state["results"], parser.modes(qpath[0]))
        elif qpath[0] == "DA" or qpath[0] == "angular_diameter_distance":
            # angular diameter distance
            state[quantity] = get_camb_sigma8(state["params"], state["results"], parser.modes(qpath[0]))
        elif qpath[0] in [ "Omega_b", "Omega_cdm" ]:
            # background evolution of densities
            var = {
              "Omega_b" : "baryon",
              "Omega_cdm" : "cdm"
            }.get(qpath[0])
            state[quantity] = get_camb_sigma8(state["params"], state["results"], parser.modes(qpath[0]), var)
        else:
            raise ValueError(f"Unknown quantity {quantity}.")
    
    return True

def translate_camb_params(**kwargs):
    args = {}
    
    for arg in kwargs:
        if arg == "z":
            args["redshifts"] = [kwargs[arg],]
        else:
            args[arg] = kwargs[arg]
    
    return args
