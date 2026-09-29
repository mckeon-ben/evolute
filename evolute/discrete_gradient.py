'''
Discrete gradient method (comparison baseline).

A discrete gradient gbar(z0, z1) satisfies

    gbar . (z1 - z0) = H(z1) - H(z0),

so the method z1 = z0 + h J gbar(z0, z1) gives
H(z1) - H(z0) = h gbar . J gbar = 0, since J is skew. Energy is
therefore conserved exactly, up to solver tolerance, for any smooth H,
and by any composition of such maps.

References
----------
Eidnes, S., 2022. Order theory for discrete gradient methods.
BIT Numerical Mathematics, 62(4), pp.1207-1255.

Itoh, T. and Abe, K., 1988. Hamiltonian-conserving discrete
canonical equations based on variational difference quotients.
Journal of Computational Physics, 76(1), pp.85-102.
'''

from abc import abstractmethod

import numpy as np

from .integrator import ImplicitMethod


class DiscreteGradientMethod(ImplicitMethod):
    '''
    Implicit method z1 = z0 + h J gbar(z0, z1) for a discrete gradient.

    Subclasses implement discrete_gradient.

    Parameters
    ----------
    dz_min : float, optional
        Increments at or below this size are treated as zero: the discrete
        gradient falls back to the exact gradient there, because the
        difference quotient is dominated by round-off.
    **kwargs
        Passed to ImplicitMethod (xtol).
    '''

    def __init__(self, dz_min=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.dz_min = dz_min

    @abstractmethod
    def discrete_gradient(self, system, z0, z1):
        '''
        Evaluate the discrete gradient.

        Parameters
        ----------
        system : HamiltonianSystem
            System supplying H and grad H.
        z0, z1 : np.ndarray
            Endpoints of the step, each shape (2n,).

        Returns
        -------
        np.ndarray
            gbar(z0, z1), shape (2n,).
        '''

    def residual(self, system, z0, z1, h):
        '''
        Evaluate the residual z1 - z0 - h J gbar(z0, z1).

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
            Residual, shape (2n,).
        '''
        gbar = self.discrete_gradient(system, z0, z1)
        return z1 - z0 - h * system.J(z0) @ gbar


class ItohAbe(DiscreteGradientMethod):
    '''
    Itoh-Abe (coordinate increment) discrete gradient method.

    The sweep from a to b is

        g_i(a, b) = [H(w_i) - H(w_{i-1})] / (b_i - a_i),

    where w_i = (b_1, ..., b_i, a_{i+1}, ..., a_d), so w_0 = a and
    w_d = b, and the sum g . (b - a) telescopes to H(b) - H(a). Sweeping
    the coordinates in the opposite order builds the same intermediate
    states in reverse, so it is the sweep with the endpoints swapped,
    g(z1, z0); as a map it is the adjoint Phi*_h = (Phi_{-h})^{-1}.

    The first-order map takes gbar = g(z0, z1), which is not symmetric.
    Order 2 takes the mean of the two sweeps,

        gbar = [g(z0, z1) + g(z1, z0)] / 2,

    the symmetrized Itoh-Abe discrete gradient of Eidnes (2022). A mean
    of discrete gradients is a discrete gradient, since the telescoping
    identity is linear in gbar, and swapping the endpoints swaps the two
    terms, so the mean is symmetric and the method second order. It
    costs one solve for a residual of two sweeps, rather than the two
    solves of a composition. Order 4 is the triple jump of that
    symmetric map. Energy is conserved exactly throughout, since every
    discrete gradient conserves it at any step length and either sign.

    Parameters
    ----------
    order : int, optional
        1 (default) for the coordinate-increment map, 2 for the
        symmetrized gradient, and 4 for the triple jump of that.
    reverse : bool, optional
        Sweep the coordinates in reverse order, which gives the adjoint
        map. Default False. Only for order 1.
    dz_min : float, optional
        Coordinate increments at or below this size use the i-th
        partial derivative at the midpoint of that coordinate step
        instead of the quotient. Default eps^(1/3), about 6e-6.
    **kwargs
        Passed to ImplicitMethod (xtol).

    Notes
    -----
    - First-, second- or fourth-order accurate in h
    - Conserves energy exactly, to solver tolerance
    - Not volume-preserving in general
    '''

    def __init__(self, order=1, reverse=False,
                 dz_min=np.finfo(float).eps ** (1 / 3), **kwargs):
        if order not in (1, 2, 4):
            raise ValueError(f'order must be 1, 2 or 4, got {order!r}')
        if reverse and order != 1:
            raise ValueError('reverse is only for the first-order map')
        self.order = order
        self.name = {1: 'Itoh-Abe', 2: 'Symmetrized Itoh-Abe',
                     4: 'Itoh-Abe triple jump'}[order]
        self.reverse = reverse
        # Coordinate increments are often small (e.g. near turning
        # points), where the quotient carries round-off ~ eps / |dz_i|,
        # which reaches the energy through the solver residual. The
        # fallback replaces the quotient by the midpoint derivative, so
        # the telescoping identity instead picks up an error ~ |dz_i|^3.
        # The two balance near (h eps |H|)^(1/4), which is a few times
        # 1e-6 for the examples, so eps^(1/3) ~ 6e-6 is about right. The
        # best threshold is problem-dependent: over 2^18 steps it beats
        # sqrt(eps) by 6x on the Kepler system and 2x on Henon-Heiles,
        # and loses to it by 12x on the logarithmic potential, whose
        # third derivative is large where the increments are small.
        super().__init__(dz_min=dz_min, **kwargs)

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
            Ignored, as for any discrete gradient method.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).

        Raises
        ------
        RuntimeError
            If the solver fails with a residual above 1e-10.
        '''
        return self._by_order(system, z0, h, E)

    def _first(self, system, z0, h, E=None):
        '''
        Solve one step with the single-sweep gradient.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        z0 : np.ndarray
            Current state, shape (2n,).
        h : float
            Step size; may be negative.
        E : float, optional
            Ignored, as for any discrete gradient method.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''
        return super().step(system, z0, h)

    def _symmetric(self, system, z0, h, E=None):
        '''
        Solve one step with the symmetrized gradient.

        The same solve as _first; discrete_gradient supplies the mean of
        the two sweeps above order 1, which is symmetric, so the map is
        symmetric and second order.

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
        return super().step(system, z0, h)

    def _sweep(self, system, a, b):
        '''
        Evaluate one coordinate sweep g(a, b), in increasing index order.

        Costs d + 1 evaluations of H, plus one of grad H for each
        coordinate at or below dz_min.

        Parameters
        ----------
        system : HamiltonianSystem
            System supplying H and grad H.
        a, b : np.ndarray
            Endpoints of the sweep, each shape (d,), d = 2n.

        Returns
        -------
        np.ndarray
            g(a, b), shape (d,).
        '''
        d = a.size
        g = np.empty(d)
        w = a.copy()
        H_prev = system.H(w)
        for i in range(d):
            dzi = b[i] - a[i]
            if abs(dzi) <= self.dz_min:
                w[i] = 0.5 * (a[i] + b[i])
                g[i] = system.grad_H(w)[i]
                w[i] = b[i]
                H_prev = system.H(w)
            else:
                w[i] = b[i]
                H_next = system.H(w)
                g[i] = (H_next - H_prev) / dzi
                H_prev = H_next
        return g

    def discrete_gradient(self, system, z0, z1):
        '''
        Evaluate the discrete gradient of this method's order.

        Order 1 takes one sweep, reversed if this instance is the
        adjoint; the higher orders take the mean of the two sweeps, the
        symmetrized Itoh-Abe gradient, at twice the cost.

        Parameters
        ----------
        system : HamiltonianSystem
            System supplying H and grad H.
        z0, z1 : np.ndarray
            Endpoints of the step, each shape (d,), d = 2n.

        Returns
        -------
        np.ndarray
            gbar(z0, z1), shape (d,).
        '''
        if self.order == 1:
            return (self._sweep(system, z1, z0) if self.reverse
                    else self._sweep(system, z0, z1))
        return 0.5 * (self._sweep(system, z0, z1)
                      + self._sweep(system, z1, z0))
