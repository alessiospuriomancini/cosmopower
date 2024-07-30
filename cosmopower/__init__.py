from .cosmopower_PCA import cosmopower_PCA  # noqa: F401
from .cosmopower_PCAplusNN import cosmopower_PCAplusNN  # noqa: F401
from .cosmopower_NN import cosmopower_NN  # noqa: F401
from .parser import YAMLParser  # noqa: F401
from .dataset import Dataset  # noqa: F401
from .likelihoods import *  # noqa: F401, F403
from .util import *  # noqa: F401, F403

__version__ = "0.2.0"
__author__ = "Alessio Spurio Mancini, Hidde Jense, and Ian Harrison"


def get_cobaya_class():
    """Utility function for Cobaya to find the CosmoPower wrapper class."""
    from wrappers.cobaya.cosmopower import CosmoPower
    return CosmoPower
