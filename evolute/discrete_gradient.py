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

    The discrete gradient is

        gbar_i = [H(w_i) - H(w_{i-1})] / (z1_i - z0_i),

    where w_i = (z1_1, ..., z1_i, z0_{i+1}, ..., z0_d), so w_0 = z0 and
    w_d = z1, and the sum gbar . dz telescopes to H(z1) - H(z0). Taking
    the coordinates in the opposite order builds the same intermediate
    states in reverse, which is the adjoint map Phi*_h = (Phi_{-h})^{-1}.
    The base map is first order and not symmetric; Phi*_{h/2} o Phi_{h/2}
    is symmetric and second order, and its triple jump is fourth order.
    Energy is conserved exactly by every such composition, since each
    factor conserves it.

    Parameters
    ----------
    order : int, optional
        1 (default) for the coordinate-increment map, 2 for the
        symmetric composition with its adjoint, and 4 for the triple
        jump of that.
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

    name = 'Itoh-Abe'

    def __init__(self, order=1, reverse=False,
                 dz_min=np.finfo(float).eps ** (1 / 3), **kwargs):
        if order not in (1, 2, 4):
            raise ValueError(f'order must be 1, 2 or 4, got {order!r}')
        if reverse and order != 1:
            raise ValueError('reverse is only for the first-order map')
        self.order = order
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
        if order > 1:
            # The two sweeps of the symmetric composition, each a
            # first-order map in its own right.
            self._sweep = ItohAbe(dz_min=dz_min, **kwargs)
            self._adjoint = ItohAbe(reverse=True, dz_min=dz_min, **kwargs)

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
        Apply the coordinate-increment map, sweeping in this instance's
        direction.

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
        Apply the symmetric composition Phi*_{h/2} o Phi_{h/2}.

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
        z = self._sweep.step(system, z0, 0.5 * h)
        return self._adjoint.step(system, z, 0.5 * h)

    def discrete_gradient(self, system, z0, z1):
        '''
        Evaluate the Itoh-Abe discrete gradient.

        Costs d + 1 evaluations of H, plus one of grad H for each
        coordinate at or below dz_min.

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
        d = z0.size
        gbar = np.empty(d)
        w = z0.copy()
        H_prev = system.H(w)
        for i in reversed(range(d)) if self.reverse else range(d):
            dzi = z1[i] - z0[i]
            if abs(dzi) <= self.dz_min:
                w[i] = 0.5 * (z0[i] + z1[i])
                gbar[i] = system.grad_H(w)[i]
                w[i] = z1[i]
                H_prev = system.H(w)
            else:
                w[i] = z1[i]
                H_next = system.H(w)
                gbar[i] = (H_next - H_prev) / dzi
                H_prev = H_next
        return gbar
