import pytest

try:
    import cobaya
    from cobaya.model import get_model
    
    HAS_COBAYA = True
except ImportError:
    HAS_COBAYA = False

"""
Ideas for tests:
 - Check that the wrapper can be loaded
 - Check that the wrapper gives proper chi square values for CMB/Lensing/P(k)/H(z) etc. values
 - Check that the wrapper provides "close enough" values when compared to the original boltzmann code.
"""

fiducial_cosmology = {
    "ombh2" : 0.022,
    "omch2" : 0.117,
    "h" : 0.67,
    "logA" : 3.05,
    "ns" : 0.96,
    "tau" : 0.0544,
    "A_planck" : 1.0,
    "As" : { "value" : "lambda logA: 1e-10 * np.exp(logA)" },
    "H0" : { "value" : "lambda h: 100.0 * h" },
}

@pytest.mark.skipif(not HAS_COBAYA, reason = "Requires cobaya")
def test_wrapper_cobaya(request):
    """
    Test that the wrapper can be loaded and evaluated.
    """
    test_model = {
        "sampler" : { "evaluate" : None },
        "theory" : { "wrappers.cobaya.CosmoPower" : { "package_file" : "example.yaml" } },
        "likelihood" : { "planck_2018_highl_plik.TTTEEE_lite_native" : { "stop_at_error" : True } },
        "params" : fiducial_cosmology
    }
    
    model = get_model(test_model)
    posterior = model.logposterior({}, as_dict = True)
