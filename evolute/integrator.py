'''
Base classes for one-step methods, and the drivers that run them.

Every method maps a state z0 to z1 = Phi_h(z0) through ``step(system,
z0, h, E)``. PartitionedMethod supplies the kick and drift substeps
used by the symplectic methods, each solving for one group of
variables; ResidualMethod solves F(z0, z1) = 0 for the whole state at
once. The Yoshida composition that lifts a symmetric second-order map
to fourth order lives here too, so that every method reaches fourth
order the same way.

The target energy E is an optional argument of every step, used by an
energy-conserving method and ignored by the others, so that one driver
runs them all. The drivers are ``integrate()`` and ``evolve()``.

References
----------
Yoshida, H., 1990. Construction of higher order symplectic
integrators. Physics Letters A, 150(5-7), pp.262-268.
'''

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import fixed_point, root


_CUBE_ROOT_2 = 2.0 ** (1 / 3)

_TRIPLE_JUMP = (1.0 / (2.0 - _CUBE_ROOT_2),
                -_CUBE_ROOT_2 / (2.0 - _CUBE_ROOT_2),
                1.0 / (2.0 - _CUBE_ROOT_2))
'''
Yoshida's coefficients for the triple jump.

They sum to one for consistency, and 2 g1^3 + g2^3 = 0 cancels the
leading error of a symmetric map.
'''

_RESIDUAL_MAX = 1e-10
'''
Largest residual a step may be left with.

Comparable to the local error of the coarsest step worth taking, and
far above the round-off floor of the residual itself, which is the
round-off of z1 - z0 and so of order eps |z0|.
'''

_XTOL_FLOOR = 1e-12
'''
Solver tolerance to fall back on when the requested one stalls.

The tightest tolerance MINPACK reaches without reporting a stall at
the finest steps these methods are run at. It returns the same
residual as a tighter request does, having converged rather than
given up.
'''


class OneStepMethod(ABC):
    '''
    One-step map z0 -> Phi_h(z0).

    Subclasses set the attributes below and implement ``step()``. One
    that comes in several orders implements the first-order map
    ``_first()`` and the symmetric map ``_symmetric()``, and routes
    ``step()`` through ``_by_order()``.

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
    ('Symmetric energy-volume split', 2)
    '''

    @abstractmethod
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
            Target energy for an energy-conserving method. By default
            it is taken from z0; ignored by methods that do not use it.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''

    def _by_order(self, system, z0, h, E):
        '''
        Apply the map of this method's order.

        Order 1 is _first, order 2 the symmetric composition _symmetric,
        and order 4 the triple jump of that.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        z0 : np.ndarray
            Current state, shape (2n,).
        h : float
            Step size; may be negative.
        E : float or None
            Target energy, passed on unchanged.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''
        if self.order == 4:
            return self._compose(self._symmetric, system, z0, h, E)
        if self.order == 2:
            return self._symmetric(system, z0, h, E)
        return self._first(system, z0, h, E)

    @staticmethod
    def _compose(symmetric, system, z0, h, E, coefficients=_TRIPLE_JUMP):
        '''
        Apply a symmetric map at a sequence of scaled step lengths.

        Structure is inherited: a composition of maps that each conserve
        energy, preserve volume or are symplectic does the same, since
        every factor does, at any step length and either sign. The same
        energy is passed to every factor, so an energy-conserving method
        holds one target across the composition.

        Parameters
        ----------
        symmetric : callable
            Symmetric map, called as symmetric(system, z, h, E).
        system : HamiltonianSystem
            System to integrate.
        z0 : np.ndarray
            Current state, shape (2n,).
        h : float
            Step size of the composition.
        E : float or None
            Target energy, passed to every factor.
        coefficients : sequence of float, optional
            Step lengths as fractions of h. The default is the
            triple jump, Yoshida's three lengths, which lift a
            symmetric second-order map to fourth order.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''
        z = z0
        for c in coefficients:
            z = symmetric(system, z, c * h, E)
        return z

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
    new value (implicit) or at its input (explicit)::

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


class ResidualMethod(OneStepMethod):
    '''
    Method defined by a residual F(z0, z1) = 0, solved with SciPy.

    The residual is over the whole state, so one solve advances every
    variable at once, where a PartitionedMethod solves for one group
    at a time.

    The solve uses MINPACK's hybrid method from an explicit Euler
    predictor.

    Parameters
    ----------
    xtol : float, optional
        Solver tolerance. Default 1e-14: SciPy's own default, about
        1.5e-8, is far too loose here, since for an energy-conserving
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

    def step(self, system, z0, h, E=None):
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
        E : float, optional
            Ignored. A discrete gradient conserves energy through the
            identity g . (z1 - z0) = H(z1) - H(z0), which refers to the
            previous state and has no slot for a supplied energy.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).

        Raises
        ------
        RuntimeError
            If the solver fails with a residual above _RESIDUAL_MAX,
            after a second attempt at _XTOL_FLOOR.
        '''
        def solve(xtol, start):
            return root(lambda z1: self.residual(system, z0, z1, h),
                        start, method='hybr', options={'xtol': xtol})

        # Explicit Euler predictor.
        sol = solve(self.xtol, z0 + h * system.vector_field(z0))
        # MINPACK reports no progress once it hits round-off below
        # xtol, on a quarter of the steps at a fine h, so judge by the
        # residual and retry at a reachable tolerance if it is large.
        if not sol.success and np.max(np.abs(sol.fun)) > _RESIDUAL_MAX:
            sol = solve(_XTOL_FLOOR, sol.x)
        residual = np.max(np.abs(sol.fun))
        if residual > _RESIDUAL_MAX:
            raise RuntimeError(
                f'{self.name} solver failed to converge at h = {h:.3e}, '
                f'residual {residual:.2e} against {_RESIDUAL_MAX:.0e}: '
                f'{sol.message}')
        return sol.x


def evolve(method, system, z0, h, steps):
    '''
    Advance a state, yielding it after every step.

    The target energy is read once from z0 and passed to every step, as
    integrate describes.

    Parameters
    ----------
    method : OneStepMethod
        Method to run.
    system : HamiltonianSystem
        System to integrate.
    z0 : np.ndarray
        Initial state, shape (2n,).
    h : float
        Step size; may be negative.
    steps : int
        Number of steps.

    Yields
    ------
    np.ndarray
        The state after each step, shape (2n,).

    Raises
    ------
    TypeError
        If steps is not an integer.
    ValueError
        If steps is not positive.

    Notes
    -----
    The checks run when evolve is called rather than when the generator
    it returns is first advanced, so a bad step count is reported at the
    call site. The stepping itself lives in _stepped.
    '''
    if not isinstance(steps, (int, np.integer)):
        raise TypeError(
            f'steps must be an integer, got {type(steps).__name__}')
    if steps <= 0:
        raise ValueError(f'steps must be a positive integer, got {steps}')
    return _stepped(method, system, z0, h, steps)


def _stepped(method, system, z0, h, steps):
    '''
    Yield the state after each step, the arguments already checked.

    Parameters
    ----------
    method : OneStepMethod
        Method to run.
    system : HamiltonianSystem
        System to integrate.
    z0 : np.ndarray
        Initial state, shape (2n,).
    h : float
        Step size; may be negative.
    steps : int
        Number of steps, positive.

    Yields
    ------
    np.ndarray
        The state after each step, shape (2n,).
    '''
    z = np.array(z0, dtype=float)
    E = system.H(z)
    for _ in range(steps):
        z = method.step(system, z, h, E)
        yield z


def integrate(method, system, z0, h, steps):
    '''
    Run a method over a number of steps and return the final state.

    The energy of z0 is the target of every step, rather than that of
    the state each step starts from. The two agree in exact arithmetic.
    In floating point, recomputing the target would make the rounding of
    one step the target of the next, and the energy error of an
    energy-conserving method would grow like the square root of the
    number of steps.

    Parameters
    ----------
    method : OneStepMethod
        Method to run.
    system : HamiltonianSystem
        System to integrate.
    z0 : np.ndarray
        Initial state, shape (2n,).
    h : float
        Step size; may be negative.
    steps : int
        Number of steps.

    Returns
    -------
    np.ndarray
        Final state, shape (2n,).

    Raises
    ------
    TypeError
        If steps is not an integer.
    ValueError
        If steps is not positive.
    '''
    z = np.array(z0, dtype=float)
    for state in evolve(method, system, z0, h, steps):
        z = state
    return z
