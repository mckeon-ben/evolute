'''
Symplectic Euler and Stormer-Verlet (comparison baselines).

Both are explicit for separable H = K(p) + V(q). For a general H they
are the implicit forms of Hairer, Lubich and Wanner (2006), with the
implicit equations solved by fixed-point iteration; they remain
symplectic, and Stormer-Verlet remains symmetric and second order.
Forest and Ruth's fourth-order method is the triple jump of
Stormer-Verlet.

References
----------
Forest, E. and Ruth, R.D., 1990. Fourth-order symplectic
integration. Physica D, 43(1), pp.105-117.

Hairer, E., Lubich, C. and Wanner, G., 2006. Geometric numerical
integration: Structure-preserving algorithms for ordinary
differential equations. 2nd ed., Springer.

Yoshida, H., 1990. Construction of higher order symplectic
integrators. Physics Letters A, 150(5-7), pp.262-268.
'''

import numpy as np

from .integrator import PartitionedMethod


class Symplectic(PartitionedMethod):
    '''
    Symplectic Euler M and its symmetric compositions.

    M_h applies the kick first, and its adjoint M*_h applies the drift
    first. Each map evaluates both derivatives of H at a single point,
    (q0, p1) for M and (q1, p0) for M*::

        M  : p1 = p0 - h dH/dq(q0, p1),  q1 = q0 + h dH/dp(q0, p1),
        M* : q1 = q0 + h dH/dp(q1, p0),  p1 = p0 - h dH/dq(q1, p0),

    so M is implicit in p and M* in q. The symmetric method
    M*_{h/2} o M_{h/2} is Stormer-Verlet, mirroring the construction of
    EnergyVolumeSplit. For a separable system both maps are explicit,
    and Stormer-Verlet is kick-drift-kick (velocity Verlet). Applying
    Stormer-Verlet at the three triple-jump step lengths gives
    Forest-Ruth, fourth order and symplectic, since each factor is.

    Parameters
    ----------
    order : int, optional
        1 for symplectic Euler, 2 (default) for Stormer-Verlet, and 4
        for Forest-Ruth, the triple jump of Stormer-Verlet.
    xtol : float, optional
        Tolerance for the fixed-point iteration, used only for a
        non-separable system. Default 1e-14.
    maxiter : int, optional
        Iteration limit for the fixed-point iteration. Default 100.

    Notes
    -----
    - First-, second- or fourth-order accurate in h
    - Symplectic, and so volume-preserving in phase space
    - Energy not conserved, though its error typically stays bounded
    - Conserves linear and bilinear invariants
    - Explicit for separable H, implicit otherwise
    '''

    def __init__(self, order=2, xtol=1e-14, maxiter=100):
        if order not in (1, 2, 4):
            raise ValueError(f'order must be 1, 2 or 4, got {order}')
        self.order = order
        self.name = {1: 'Symplectic Euler', 2: 'Störmer-Verlet',
                     4: 'Forest-Ruth'}[order]
        self.xtol = xtol
        self.maxiter = maxiter

    def step(self, system, z0, h, E=None):
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
        E : float, optional
            Ignored; a symplectic method does not conserve energy.

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
        return self._by_order(system, z0, h, E)

    def _first(self, system, z0, h, E=None):
        '''
        Apply symplectic Euler, M.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        z0 : np.ndarray
            Current state, shape (2n,).
        h : float
            Step size; may be negative.
        E : float, optional
            Ignored; present so the map can be composed.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''
        q, p = system.split(z0)
        return np.concatenate(self._forward(system, q, p, h))

    def _symmetric(self, system, z0, h, E=None):
        '''
        Apply Stormer-Verlet, M*_{h/2} o M_{h/2}.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        z0 : np.ndarray
            Current state, shape (2n,).
        h : float
            Step size; may be negative.
        E : float, optional
            Ignored; present so the map can be composed.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''
        q, p = system.split(z0)
        q, p = self._forward(system, q, p, 0.5 * h)
        q, p = self._adjoint(system, q, p, 0.5 * h)
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
        q : np.ndarray
            Updated positions, shape (n,).
        p : np.ndarray
            Updated momenta, shape (n,).
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
        q : np.ndarray
            Updated positions, shape (n,).
        p : np.ndarray
            Updated momenta, shape (n,).
        '''
        q = self._drift(system, q, p, h, implicit=True)
        p = self._kick(system, q, p, h)
        return q, p
