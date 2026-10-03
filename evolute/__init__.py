'''
evolute: Energy- and VOLUme-preserving Time integration Engine.

A Python package implementing one-step integrators for Hamiltonian
systems, centered on EnergyVolumeSplit, which conserves energy and
preserves phase-space volume exactly. Symplectic and discrete gradient
methods are included for comparison. EnergyVolumeSplit and Symplectic
each provide a first-order method, its second-order symmetric
composition, and the fourth-order triple jump of that.

Classes
-------
HamiltonianSystem
    Hamiltonian system built from a potential, an optional kinetic
    energy, and their gradients.

Functions
---------
integrate
    Run a method over a number of steps and return the final state.
evolve
    The same, yielding the state after every step.

Energy- and volume-preserving methods
-------------------------------------
EnergyVolumeSplit
    Energy- and volume-preserving splitting, first, second or fourth
    order.

Comparison methods
------------------
Symplectic
    Symplectic Euler, its symmetric composition Stormer-Verlet, or the
    fourth-order Forest-Ruth.
ItohAbe
    Itoh-Abe discrete gradient method, its symmetrized form, or the
    fourth-order triple jump of that.
'''

from .system import HamiltonianSystem
from .integrator import integrate, evolve
from .energy_volume_split import EnergyVolumeSplit
from .symplectic import Symplectic
from .discrete_gradient import ItohAbe

__all__ = [
    'HamiltonianSystem',
    'integrate',
    'evolve',
    'EnergyVolumeSplit',
    'Symplectic',
    'ItohAbe'
]
