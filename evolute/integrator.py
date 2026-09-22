'''
Base classes for one-step methods.

Every method maps a state z0 to z1 = Phi_h(z0) through step(system,
z0, h). PartitionedMethod supplies the kick and drift substeps used by
the symplectic methods; ImplicitMethod solves a residual equation for
z1.
'''

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import fixed_point, root


class OneStepMethod(ABC):
    '''
    One-step map z0 -> Phi_h(z0).

    Subclasses set the attributes below and implement step.

    Attributes
    ----------
    name : str
        Display name, used in labels and titles.
    order : int
        Classical order of accuracy.

    Examples
    --------
    Concrete methods are instantiated directly rather than through the
    base class:

    >>> from evolute import EnergyVolumeSplit, HamiltonianSystem
    >>> system = HamiltonianSystem(2, V=lambda q: 0.5 * (q @ q),
    ...                            grad_V=lambda q: q)
    >>> method = EnergyVolumeSplit()
    >>> z = method.step(system, np.array([1., 0., 0., 1.]), 0.1)
    >>> method.name, method.order
    ('Energy-volume split', 2)
    '''

    @abstractmethod
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

    def _fixed_point(self, g, x0):
        '''
        Solve x = g(x) by SciPy's fixed-point iteration from x0.

        SciPy accelerates the iteration by Steffensen's method (its
        default, del2). For subclasses that set the attributes xtol and
        maxiter.

        Parameters
        ----------
        g : callable
            Map on arrays, a contraction for small enough h.
        x0 : np.ndarray
            Initial guess.

        Returns
        -------
        np.ndarray
            Fixed point, to relative tolerance xtol.

        Raises
        ------
        RuntimeError
            If the iteration does not converge within maxiter steps.
        '''
        try:
            return fixed_point(func=g, x0=x0, xtol=self.xtol,
                               maxiter=self.maxiter)
        except RuntimeError as e:
            raise RuntimeError(
                f'{self.name} solver failed to converge: {e}'
            ) from e


class PartitionedMethod(OneStepMethod):
    '''
    Method that updates positions and momenta in separate substeps.

    Provides kick and drift substeps. Each substep evaluates the
    derivatives of H either with the variable it updates taken at its
    new value (implicit) or at its input (explicit):

        kick   p* = p - c dH/dq(q, p*)   or   p* = p - c dH/dq(q, p),
        drift  q* = q + c dH/dp(q*, p)   or   q* = q + c dH/dp(q, p).

    For a separable system, H = K(p) + V(q), the two forms coincide and
    every substep is explicit. Otherwise the implicit forms are solved by
    fixed-point iteration, so subclasses set the attributes xtol and
    maxiter.
    '''

    @staticmethod
    def _dH_dq(system, q, p):
        '''
        Evaluate dH/dq = grad V + dK/dq.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        q, p : np.ndarray
            Positions and momenta, each shape (n,).

        Returns
        -------
        np.ndarray
            Gradient with respect to q, shape (n,).
        '''
        return system.grad_V(q) + system.grad_kinetic(q, p)[0]

    @staticmethod
    def _dH_dp(system, q, p):
        '''
        Evaluate dH/dp = dK/dp.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        q, p : np.ndarray
            Positions and momenta, each shape (n,).

        Returns
        -------
        np.ndarray
            Gradient with respect to p, shape (n,).
        '''
        return system.grad_kinetic(q, p)[1]

    def _kick(self, system, q, p, c, implicit=False):
        '''
        Kick the momenta: p -> p - c dH/dq.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        q, p : np.ndarray
            Positions and momenta, each shape (n,).
        c : float
            Substep length.
        implicit : bool, optional
            Evaluate dH/dq at the new momenta if True, at p if False.
            Irrelevant for a separable system.

        Returns
        -------
        np.ndarray
            Updated momenta, shape (n,).
        '''
        if system.separable:
            return p - c * system.grad_V(q)

        def kick(x):
            return p - c * self._dH_dq(system, q, x)

        return self._fixed_point(kick, p) if implicit else kick(p)

    def _drift(self, system, q, p, c, implicit=False):
        '''
        Drift the positions: q -> q + c dH/dp.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        q, p : np.ndarray
            Positions and momenta, each shape (n,).
        c : float
            Substep length.
        implicit : bool, optional
            Evaluate dH/dp at the new positions if True, at q if False.
            Irrelevant for a separable system.

        Returns
        -------
        np.ndarray
            Updated positions, shape (n,).
        '''
        if system.separable:
            return q + c * self._dH_dp(system, q, p)

        def drift(x):
            return q + c * self._dH_dp(system, x, p)

        return self._fixed_point(drift, q) if implicit else drift(q)


class ImplicitMethod(OneStepMethod):
    '''
    Method defined by a residual F(z0, z1) = 0, solved with SciPy.

    The solve uses MINPACK's hybrid method from an explicit Euler
    predictor.

    Parameters
    ----------
    xtol : float, optional
        Solver tolerance. Default 1e-14: SciPy's own default (about
        1.5e-8) is far too loose here, since for an energy-conserving
        method the solver tolerance sets the observed energy error.
    '''

    def __init__(self, xtol=1e-14):
        self.xtol = xtol

    @abstractmethod
    def residual(self, system, z0, z1, h):
        '''
        Evaluate the residual whose root defines the step.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        z0, z1 : np.ndarray
            Current state and candidate new state, each shape (2n,).
        h : float
            Step size.

        Returns
        -------
        np.ndarray
            F(z0, z1), shape (2n,).
        '''

    def step(self, system, z0, h):
        '''
        Advance the state by one step by solving F(z0, z1) = 0.

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
            If the solver fails with a residual above 1e-10.
        '''
        guess = z0 + h * system.vector_field(z0)   # explicit Euler
        sol = root(lambda z1: self.residual(system, z0, z1, h), guess,
                   method='hybr', options={'xtol': self.xtol})
        # MINPACK reports "no further improvement" once it hits round-off
        # below xtol; that is a converged solve, so judge by the residual.
        if not sol.success and np.max(np.abs(sol.fun)) > 1e-10:
            raise RuntimeError(
                f'{self.name} solver failed to converge: {sol.message}')
        return sol.x
