import numpy as np

def _cmb_unit_factor(units, T_cmb):
    units_factors = {"1": 1,
                     "muK2": T_cmb * 1.e6,
                     "K2": T_cmb,
                     "FIRASmuK2": 2.7255e6,
                     "FIRASK2": 2.7255
                     }
    try:
        return units_factors[units]
    except KeyError:
        raise NotImplemented("Units '%s' not recognized. Use one of %s.",
                              units, list(units_factors))

def cmb_unit_factor(spectra: str, units: str = "FIRASmuK2", Tcmb: float = 2.7255) -> float:
    """
    Calculate the CMB prefactor for going from dimensionless power spectra to CMB units.
    :param spectra: a length 2 string specifying the spectrum for which to calculate the units.
    :param units: a string specifying which units to use.
    :param Tcmb: the used CMB temperature [units of K].
    :return: The CMB unit conversion factor.
    """
    res = 1.0
    x, y = spectra.lower()

    if x == "t" or x == "e" or x == "b":
        res *= _cmb_unit_factor(units, Tcmb)
    elif x == "p":
        res *= 1. / np.sqrt(2.0 * np.pi)

    if y == "t" or y == "e" or y == "b":
        res *= _cmb_unit_factor(units, Tcmb)
    elif y == "p":
        res *= 1. / np.sqrt(2.0 * np.pi)

    return res

def ell_factor(ls: np.ndarray, spectra: str) -> np.ndarray:
    """
    Calculate the ell factor for a specific spectrum.
    These prefactors are used to convert from Cell to Dell and vice-versa.

    The cosmosis-standard-library clik interface expects ell(ell+1)/2 pi Cl
    for all angular power spectra, including the lensing potential.
    For compatability reasons, we provide that scaling here as well.

    Example:
    ell_factor(l, "tt") -> l(l+1)/(2 pi).
    ell_factor(l, "pp") -> l(l+1)/(2 pi).

    :param ls: the range of ells.
    :param spectra: a two-character string with each character being one of [tebp].

    :return: an array filled with ell factors for the given spectrum.
    """
    ellfac = np.ones_like(ls).astype(float)

    if spectra in ["tt", "te", "tb", "ee", "et", "eb", "bb", "bt", "be"]:
        ellfac = ls * (ls + 1.0) / (2.0 * np.pi)
    elif spectra in ["pt", "pe", "pb", "tp", "ep", "bp"]:
        ellfac = ls * (ls + 1.0) / (2.0 * np.pi)
    elif spectra in ["pp"]:
        ellfac = ls * (ls + 1.0) / (2.0 * np.pi)

    return ellfac

def get_noise_curves_CVL(parser, spectra, fsky = 1.0):
    results = {}
    
    # TT spectra.
    if "Cl/tt" in spectra:
        ls = parser.modes("Cl/tt")
        prefac = 1.0 / (fsky * (2.0 * ls + 1.0))
        results["Cl/tt"] = np.sqrt(2.0 * prefac * spectra["Cl/tt"] ** 2.0)
    
    # TE spectra.
    if "Cl/te" in spectra:
        ls = parser.modes("Cl/te")
        prefac = 1.0 / (fsky * (2.0 * ls + 1.0))
        cltt = spectra["Cl/tt"]
        clte = spectra["Cl/te"]
        clee = spectra["Cl/ee"]
        
        results["Cl/te"] = np.sqrt(prefac * (clte ** 2.0 + cltt * clee))
    
    # EE spectra.
    if "Cl/ee" in spectra:
        ls = parser.modes("Cl/ee")
        prefac = 1.0 / (fsky * (2.0 * ls + 1.0))
        results["Cl/ee"] = np.sqrt(2.0 * prefac * spectra["Cl/ee"] ** 2.0)
    
    # BB spectra.
    if "Cl/bb" in spectra:
        ls = parser.modes("Cl/bb")
        prefac = 1.0 / (fsky * (2.0 * ls + 1.0))
        results["Cl/bb"] = np.sqrt(2.0 * prefac * spectra["Cl/bb"] ** 2.0)
    
    # pp spectra.
    if "Cl/pp" in spectra:
        ls = parser.modes("Cl/pp")
        prefac = 1.0 / (fsky * (2.0 * ls + 1.0))
        results["Cl/pp"] = np.sqrt(2.0 * prefac * spectra["Cl/pp"] ** 2.0)
    
    return results

def get_noise_curves_SO(parser, spectra, T_cmb = 2.7255):
    results = {}
    
    TT_ells, TT_Nell = np.loadtxt("test/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_CMB.txt", usecols = (0, 1), unpack = True)
    EE_ells, EE_Nell = np.loadtxt("test/SO_LAT_Nell_P_goal_fsky0p4_ILC_CMB_E.txt", usecols = (0, 1), unpack = True)
    BB_ells, BB_Nell = np.loadtxt("test/SO_LAT_Nell_P_goal_fsky0p4_ILC_CMB_B.txt", usecols = (0, 1), unpack = True)
    PP_ells, PP_Nell = np.loadtxt("test/nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat", usecols = (0, 7), unpack = True)
    
    TT_Nell /= (T_cmb * 1e6) ** 2.0
    EE_Nell /= (T_cmb * 1e6) ** 2.0
    BB_Nell /= (T_cmb * 1e6) ** 2.0
    PP_Nell /= (PP_ells * (PP_ells + 1) / 2) ** 2.0
    f_sky = 0.4
    
    # TT spectra.
    if "Cl/tt" in spectra:
        TT_modes = parser.modes("Cl/tt")
        
        # re-cut TT_Nell to the correct indices
        Nell = np.tile(np.nan, (spectra["Cl/tt"].shape[1],))
        idx = np.where(np.in1d(TT_modes, TT_ells))[0]
        jdx = np.where(np.in1d(TT_ells, TT_modes))[0]
        Nell[idx] = TT_Nell[jdx]
        
        prefac = np.sqrt(2.0 / (f_sky * (2.0 * TT_modes + 1.0)))
        tot_spec = (spectra["Cl/tt"] + Nell)
        
        results["Cl/tt"] = prefac * tot_spec
    if "Cl/te" in spectra:
        TE_modes = parser.modes("Cl/te")
        
        NellTT = np.tile(np.nan, (spectra["Cl/te"].shape[1],))
        NellEE = np.tile(np.nan, (spectra["Cl/te"].shape[1],))
        
        idx = np.where(np.in1d(TE_modes, TT_ells))[0]
        jdx = np.where(np.in1d(TT_ells, TE_modes))[0]
        NellTT[idx] = TT_Nell[jdx]
        
        idx = np.where(np.in1d(TE_modes, EE_ells))[0]
        jdx = np.where(np.in1d(EE_ells, TE_modes))[0]
        NellEE[idx] = EE_Nell[jdx]
        
        prefac = np.sqrt(1.0 / (f_sky * (2.0 * TE_modes + 1.0)))
        tot_spec = np.sqrt((spectra["Cl/te"] * spectra["Cl/te"]) + (spectra["Cl/tt"] + NellTT) * (spectra["Cl/ee"] + NellEE))
        
        results["Cl/te"] = prefac * tot_spec
    if "Cl/ee" in spectra:
        EE_modes = parser.modes("Cl/ee")
        
        Nell = np.tile(np.nan, (spectra["Cl/ee"].shape[1],))
        idx = np.where(np.in1d(EE_modes, EE_ells))[0]
        jdx = np.where(np.in1d(EE_ells, EE_modes))[0]
        Nell[idx] = EE_Nell[jdx]
        
        prefac = np.sqrt(2.0 / (f_sky * (2.0 * EE_modes + 1.0)))
        tot_spec = (spectra["Cl/ee"] + Nell)
        
        results["Cl/ee"] = prefac * tot_spec
    if "Cl/bb" in spectra:
        BB_modes = parser.modes("Cl/bb")
        
        Nell = np.tile(np.nan, (spectra["Cl/bb"].shape[1],))
        idx = np.where(np.in1d(BB_modes, BB_ells))[0]
        jdx = np.where(np.in1d(BB_ells, BB_modes))[0]
        Nell[idx] = BB_Nell[jdx]
        
        prefac = np.sqrt(2.0 / (f_sky * (2.0 * BB_modes + 1.0)))
        tot_spec = (spectra["Cl/bb"] + Nell)
        
        results["Cl/bb"] = prefac * tot_spec
    if "Cl/pp" in spectra:
        PP_modes = parser.modes("Cl/pp")
        
        Nell = np.zeros(spectra["Cl/pp"].shape[1])
        idx = np.where(np.in1d(PP_modes, PP_ells))[0]
        jdx = np.where(np.in1d(PP_ells, PP_modes))[0]
        Nell[idx] = PP_Nell[jdx]
        
        prefac = np.sqrt(2.0 / (f_sky * (2.0 * PP_modes + 1.0)))
        tot_spec = (spectra["Cl/pp"] + Nell)
        
        results["Cl/pp"] = prefac * tot_spec
    
    return results
