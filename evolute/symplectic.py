'''
Symplectic Euler and Stormer-Verlet (comparison baselines).

Both are explicit for separable H = K(p) + V(q). For a general H they
are the implicit forms of Hairer, Lubich and Wanner (2006), with the
implicit equations solved by fixed-point iteration; they remain
symplectic, and Stormer-Verlet remains symmetric and second order.

References
----------
Hairer, E., Lubich, C. and Wanner, G., 2006. Geometric numerical
integration: Structure-preserving algorithms for ordinary
differential equations. 2nd ed., Springer.
'''

import numpy as np

from .integrator import PartitionedMethod


class Symplectic(PartitionedMethod):
    '''
    Symplectic Euler M and its symmetric composition.

    M_h applies the kick first, and its adjoint M*_h applies the drift
    first. Each map evaluates both derivatives of H at a single point,
    (q0, p1) for M and (q1, p0) for M*:

        M  : p1 = p0 - h dH/dq(q0, p1),  q1 = q0 + h dH/dp(q0, p1),
        M* : q1 = q0 + h dH/dp(q1, p0),  p1 = p0 - h dH/dq(q1, p0),

    so M is implicit in p and M* in q. The symmetric method
    M*_{h/2} o M_{h/2} is Stormer-Verlet, mirroring the construction of
    EnergyVolumeSplit. For a separable system both maps are explicit,
    and Stormer-Verlet is kick-drift-kick (velocity Verlet).

    Parameters
    ----------
    symmetric : bool, optional
        True (default) for Stormer-Verlet, second order; False for
        symplectic Euler, first order.
    xtol : float, optional
        Tolerance for the fixed-point iteration, used only for a
        non-separable system. Default 1e-14.
    maxiter : int, optional
        Iteration limit for the fixed-point iteration. Default 100.

    Notes
    -----
    - First-order (symplectic Euler) or second-order (Stormer-Verlet)
      accurate in h
    - Symplectic, and so volume-preserving in phase space
    - Energy not conserved, though its error typically stays bounded
    - Conserves linear invariants, such as the momentum conjugate to a
      cyclic coordinate, and bilinear invariants q . C p, such as
      angular momentum
    - Explicit for separable H, implicit otherwise
    '''

    def __init__(self, symmetric=True, xtol=1e-14, maxiter=100):
        self.symmetric = symmetric
        self.order = 2 if symmetric else 1
        self.name = 'Störmer-Verlet' if symmetric else 'Symplectic Euler'
        self.xtol = xtol
        self.maxiter = maxiter

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

        Raises
        ------
        RuntimeError
            If the fixed-point iteration does not converge, for a
            non-separable system.
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
        Apply M: kick implicit in p, then explicit drift.

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
        p = self._kick(system, q, p, h, implicit=True)
        q = self._drift(system, q, p, h)
        return q, p

    def _adjoint(self, system, q, p, h):
        '''
        Apply M*: drift implicit in q, then explicit kick.

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
        q = self._drift(system, q, p, h, implicit=True)
        p = self._kick(system, q, p, h)
        return q, p
