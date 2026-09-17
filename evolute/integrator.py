'''
Base classes for one-step methods.

Every method maps a state z0 to z1 = Phi_h(z0) through step(system,
z0, h). ExplicitMethod supplies the kick and drift substeps used by the
symplectic methods; ImplicitMethod solves a residual equation for z1.
'''

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import root


class OneStepMethod(ABC):
    '''
    One-step map z0 -> Phi_h(z0).

    Subclasses set `name`, a display label, and `order`, the classical
    order of accuracy, and implement step.
    '''

    name: str
    order: int

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


class ExplicitMethod(OneStepMethod):
    '''
    Method that steps without a nonlinear solve.

    Provides kick and drift substeps built on the separable structure
    H = K(p) + V(q) of HamiltonianSystem: kicks use grad V, drifts use
    grad K.
    '''

    @staticmethod
    def _kick(system, q, p, c):
        '''
        Kick the momenta: p -> p - c grad V(q).

        Parameters
        ----------
        system : HamiltonianSystem
            System supplying grad V.
        q, p : np.ndarray
            Positions and momenta, each shape (n,).
        c : float
            Substep length.

        Returns
        -------
        np.ndarray
            Updated momenta, shape (n,).
        '''
        return p - c * system.grad_V(q)

    @staticmethod
    def _drift(system, q, p, c):
        '''
        Drift the positions: q -> q + c grad K(p).

        Parameters
        ----------
        system : HamiltonianSystem
            System supplying grad K, the full kinetic gradient.
        q, p : np.ndarray
            Positions and momenta, each shape (n,).
        c : float
            Substep length.

        Returns
        -------
        np.ndarray
            Updated positions, shape (n,).
        '''
        return q + c * system.grad_kinetic(p)


class ImplicitMethod(OneStepMethod):
    '''
    Method defined by a residual F(z0, z1) = 0, solved with SciPy.

    The solve uses MINPACK's hybrid method from an explicit Euler
    predictor.

    Parameters
    ----------
    xtol : float, optional
        Solver tolerance. SciPy's default (about 1.5e-8) is far too
        loose here: for an energy-conserving method the solver
        tolerance sets the observed energy error.
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
            raise RuntimeError(f'{self.name}: {sol.message}')
        return sol.x
