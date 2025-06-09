"""Top level package for the Anubis toolkit.

The library collects common utilities used when planning and analysing
controlled experiments. Import submodules to access specific functionality:

- :mod:`anubis.preprocessing` – data cleaning helpers.
- :mod:`anubis.power` – power and sample size calculations.
- :mod:`anubis.tests` – wrappers for statistical tests.
- :mod:`anubis.stratification` – building balanced cohorts.
- :mod:`anubis.cuped` – CUPED variance reduction implementation.
- :mod:`anubis.simulation` – tools for running synthetic experiments.
"""

from . import preprocessing
from . import power
from . import tests
from . import stratification
from . import cuped
from . import simulation

__all__ = [
    "preprocessing",
    "power",
    "tests",
    "stratification",
    "cuped",
    "simulation",
]
