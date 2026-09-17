'''
evolute: Energy- and VOLUme-preserving Time integration Engine.

A Python package implementing one-step integrators for separable
Hamiltonian systems, centred on EnergyVolumeSplit, which conserves
energy and preserves phase-space volume exactly. Symplectic and
discrete gradient methods are included for comparison.
EnergyVolumeSplit and Symplectic each provide a first-order method and
its second-order symmetric composition.

Classes
-------
HamiltonianSystem
    Separable Hamiltonian system built from a potential and its
    gradient.
ExplicitMethod
    Base class for methods that step without a nonlinear solve.
ImplicitMethod
    Base class for methods defined by a residual equation.

Functions
---------
canonical_J
    Canonical structure matrix on R^{2n}.

Energy- and volume-preserving methods
-------------------------------------
EnergyVolumeSplit
    Energy- and volume-preserving splitting, first or second order.

Comparison methods
------------------
Symplectic
    Symplectic Euler, or its symmetric composition, Stormer-Verlet.
ItohAbe
    Itoh-Abe discrete gradient method, first order.
Gonzalez
    Gonzalez discrete gradient method, second order.
'''

from .system import HamiltonianSystem, canonical_J
from .integrator import ExplicitMethod, ImplicitMethod
from .energy_volume_split import EnergyVolumeSplit
from .discrete_gradient import Gonzalez, ItohAbe
from .symplectic import Symplectic

__all__ = [
    'HamiltonianSystem', 'canonical_J',
    'ExplicitMethod', 'ImplicitMethod',
    'EnergyVolumeSplit', 'Gonzalez', 'ItohAbe', 'Symplectic',
]
