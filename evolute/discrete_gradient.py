'''
Discrete gradient methods (comparison baselines).

A discrete gradient gbar(z0, z1) satisfies

    gbar . (z1 - z0) = H(z1) - H(z0),

so the method z1 = z0 + h J gbar(z0, z1) gives
H(z1) - H(z0) = h gbar . J gbar = 0, since J is skew. Energy is
therefore conserved exactly, up to solver tolerance, for any smooth H.
'''

from abc import abstractmethod

import numpy as np

from evolute.integrator import ImplicitMethod


class DiscreteGradientMethod(ImplicitMethod):
    '''
    Implicit method z1 = z0 + h J gbar(z0, z1) for a discrete gradient.

    Subclasses implement discrete_gradient.

    Parameters
    ----------
    dz_min : float, optional
        Increments below this size are treated as zero: the discrete
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
    w_d = z1, and the sum gbar . dz telescopes to H(z1) - H(z0). Not
    symmetric, first order.

    Parameters
    ----------
    dz_min : float, optional
        Coordinate increments at or below this size use the i-th
        partial derivative at the midpoint of that coordinate step
        instead of the quotient. Default eps^(1/3), about 6e-6.
    **kwargs
        Passed to ImplicitMethod (xtol).
    '''

    name = "Itoh-Abe"
    order = 1

    def __init__(self, dz_min=np.finfo(float).eps ** (1 / 3), **kwargs):
        # Coordinate increments are often small (e.g. near turning
        # points), where the quotient carries round-off ~ eps / |dz_i|
        # and limits the solver residual, hence the energy error. The
        # fallback's error grows like |dz_i|^3; eps^(1/3) ~ 6e-6 sits
        # between the two in tests on the Kepler system.
        super().__init__(dz_min=dz_min, **kwargs)

    def discrete_gradient(self, system, z0, z1):
        '''
        Evaluate the Itoh-Abe discrete gradient.

        Costs d + 1 evaluations of H, plus one of grad H for each
        coordinate below dz_min.

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
        for i in range(d):
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


class Gonzalez(DiscreteGradientMethod):
    '''
    Gonzalez (midpoint) discrete gradient method.

    The discrete gradient is

        gbar = grad H(zm) + [H(z1) - H(z0) - grad H(zm) . dz] dz / |dz|^2,

    with zm = (z0 + z1) / 2 and dz = z1 - z0. It is unchanged when z0
    and z1 are swapped, so the method is symmetric; second order.

    Parameters
    ----------
    dz_min : float, optional
        Steps with |dz| at or below this size use grad H(zm) alone.
        Default 1e-12.
    **kwargs
        Passed to ImplicitMethod (xtol).
    '''

    name = "Gonzalez"
    order = 2

    def discrete_gradient(self, system, z0, z1):
        '''
        Evaluate the Gonzalez discrete gradient.

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
        dz = z1 - z0
        g = system.grad_H(0.5 * (z0 + z1))
        dz2 = dz @ dz
        if dz2 <= self.dz_min ** 2:
            # The correction is O(|dz|^2); dropping it is harmless.
            return g
        defect = system.H(z1) - system.H(z0) - g @ dz
        return g + (defect / dz2) * dz
