'''
Symplectic Euler and Stormer-Verlet (comparison baselines).
'''

import numpy as np

from evolute.integrator import ExplicitMethod


class Symplectic(ExplicitMethod):
    '''
    Symplectic Euler M and its symmetric composition.

    M_h = drift_h o kick_h applies the kick first, and its adjoint
    M*_h = kick_h o drift_h applies the drift first. The symmetric
    method M*_{h/2} o M_{h/2} is kick-drift-kick, that is,
    Stormer-Verlet (velocity Verlet), mirroring the construction of
    EnergyVolumeSplit.

    Parameters
    ----------
    symmetric : bool, optional
        True (default) for Stormer-Verlet, second order; False for
        symplectic Euler, first order.
    '''

    def __init__(self, symmetric=True):
        self.symmetric = symmetric
        self.order = 2 if symmetric else 1
        self.name = "Störmer-Verlet" if symmetric else "Symplectic Euler"

    def step(self, system, z0, h):
        '''
        Advance the state by one step.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        z0 : np.ndarray
            Current state, shape (2n,).
        h : float
            Step size; may be negative.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''
        q, p = system.split(z0)
        if self.symmetric:
            q, p = self._forward(system, q, p, 0.5 * h)
            q, p = self._adjoint(system, q, p, 0.5 * h)
        else:
            q, p = self._forward(system, q, p, h)
        return np.concatenate([q, p])

    def _forward(self, system, q, p, h):
        '''
        Apply M: kick, then drift.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        q, p : np.ndarray
            Positions and momenta, each shape (n,).
        h : float
            Step length for this map.

        Returns
        -------
        q, p : np.ndarray
            Updated positions and momenta.
        '''
        p = self._kick(system, q, p, h)
        q = self._drift(system, q, p, h)
        return q, p

    def _adjoint(self, system, q, p, h):
        '''
        Apply M*: drift, then kick.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        q, p : np.ndarray
            Positions and momenta, each shape (n,).
        h : float
            Step length for this map.

        Returns
        -------
        q, p : np.ndarray
            Updated positions and momenta.
        '''
        q = self._drift(system, q, p, h)
        p = self._kick(system, q, p, h)
        return q, p
