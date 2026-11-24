'''
McLachlan and Quispel's energy- and volume-preserving method
(comparison baseline, Henon-Heiles only).

The method splits the structure matrix J rather than the Hamiltonian.
Writing J = sum_i J_i, where J_i holds the 2 x 2 canonical block of the
pair (q_i, p_i) and is zero elsewhere, each piece

    dz/dt = J_i grad H(z)

moves that one pair and freezes the rest. Every piece conserves H,
because J_i is skew, and preserves phase-space volume, because J_i is
constant:

    div(J_i grad H) = trace(J_i grad^2 H) = 0.

Neither property depends on the piece being solved exactly, only on the
discrete map inheriting them. The pieces are not symplectic, since J_i
is singular, so no composition of them is either.

References
----------
McLachlan, R.I. and Quispel, G.R.W., 2003. Geometric integration of
conservative polynomial ODEs. Applied Numerical Mathematics, 45(4),
pp.411-418.
'''

import numpy as np

from .integrator import OneStepMethod


class McLachlanQuispel(OneStepMethod):
    '''
    Energy- and volume-preserving splitting of J, order 1, 2 or 4.

    The system must have H quadratic in each pair (q_i, p_i)
    separately, the other variables held fixed, in whichever frame
    its to_working reaches. Nothing checks that hypothesis, so it is
    the caller's to keep: the method is silently wrong without it,
    since the affine fit below would no longer be exact.

    Each substep moves one pair by the implicit midpoint rule, which
    for an affine piece is the 2 x 2 solve

        (2 - h g_q) u - h g_p       v = 2 q_i + h g(0, 0),
        h f_q       u + (2 + h f_p) v = 2 p_i - h f(0, 0),

    in the midpoint values u = (q_i + q_i') / 2 and
    v = (p_i + p_i') / 2, where f = dH/dq_i and g = dH/dp_i. The four
    slopes are read off from three evaluations of grad H, at
    (q_i, p_i) = (0, 0), (1, 0) and (0, 1), which is exact because f
    and g are affine in the pair. The pair moves under the matrix
    B = [[g_q, g_p], [-f_q, -f_p]], whose trace vanishes because g_q
    and f_p are both d^2H/dq_i dp_i, so

        det(I + h B / 2) = det(I - h B / 2)

    and each substep has Jacobian determinant exactly 1.

    First order sweeps the pairs in index order; the symmetric method is
    the Strang composition, a half step up the pairs and a half step
    back down; fourth order is the triple jump of that, inherited from
    OneStepMethod.

    Parameters
    ----------
    order : int, optional
        1, 2 (default) or 4.
    name : str, optional
        Display name; by default one naming the order.

    Raises
    ------
    ValueError
        If order is not 1, 2 or 4.

    Notes
    -----
    - First-, second- or fourth-order accurate in h
    - Conserves H exactly, to round-off
    - Preserves phase-space volume exactly
    - Not symplectic: the pieces have singular structure matrices
    - Effectively explicit: one 2 x 2 linear solve per pair per substep
    - Requires H biquadratic, quadratic in each pair (q_i, p_i)
      separately, which is not checked

    The state last returned is remembered with its working form, so a
    run crosses the change of variables once at each end rather than
    twice per step. That is not for speed: S^-1 S z differs from z in
    the last few bits, and those bits accumulate into an energy error
    growing like the square root of the number of steps. A state the
    cache does not recognize is transformed as usual, so the method
    remains a pure map from z0 to z1.
    '''

    def __init__(self, order=2, name=None):
        if order not in (1, 2, 4):
            raise ValueError(f'order must be 1, 2 or 4, got {order}')
        self.order = order
        self.name = name or {
            1: 'McLachlan-Quispel',
            2: 'Symmetric McLachlan-Quispel',
            4: 'McLachlan-Quispel triple jump',
        }[order]
        self._cached_z = None
        self._cached_x = None

    def step(self, system, z0, h, E=None):
        '''
        Advance the state by one step.

        Parameters
        ----------
        system : System
            System to integrate, whose H must be biquadratic in its
            working frame.
        z0 : np.ndarray
            Current state, shape (2n,).
        h : float
            Step size; may be negative.
        E : float, optional
            Ignored; the method conserves whatever energy z0 has.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''
        return self._by_order(system, z0, h, E)

    def _first(self, system, z0, h, E=None):
        '''
        Sweep the pairs once in index order.

        Parameters
        ----------
        system : System
            System to integrate.
        z0 : np.ndarray
            Current state, shape (2n,).
        h : float
            Step size; may be negative.
        E : float, optional
            Ignored.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''
        x = self._to_working(system, z0)
        for i in range(system.n):
            x = self._pair(system, x, i, h)
        return self._from_working(system, x)

    def _symmetric(self, system, z0, h, E=None):
        '''
        Apply the Strang composition, up the pairs and back down.

        Parameters
        ----------
        system : System
            System to integrate.
        z0 : np.ndarray
            Current state, shape (2n,).
        h : float
            Step size; may be negative.
        E : float, optional
            Ignored.

        Returns
        -------
        np.ndarray
            New state z1, shape (2n,).
        '''
        x = self._to_working(system, z0)
        for i in range(system.n):
            x = self._pair(system, x, i, 0.5 * h)
        for i in reversed(range(system.n)):
            x = self._pair(system, x, i, 0.5 * h)
        return self._from_working(system, x)

    def _pair(self, system, x, i, h):
        '''
        Advance the pair (q_i, p_i) by the implicit midpoint rule.

        The other components are held fixed, so the piece is affine and
        the rule reduces to one 2 x 2 linear solve. Three evaluations of
        grad H recover its coefficients exactly.

        Parameters
        ----------
        system : System
            System to integrate.
        x : np.ndarray
            State in the working frame, shape (2n,).
        i : int
            Index of the pair to move.
        h : float
            Step length for this substep.

        Returns
        -------
        np.ndarray
            State with the pair advanced, shape (2n,).
        '''
        m = system.n
        qi, pi = x[i], x[m + i]
        probe = x.copy()

        def at(u, v):
            probe[i], probe[m + i] = u, v
            g = system.working_grad_H(probe)
            return g[i], g[m + i]

        f00, g00 = at(0.0, 0.0)
        f10, g10 = at(1.0, 0.0)
        f01, g01 = at(0.0, 1.0)
        f_q, f_p = f10 - f00, f01 - f00
        g_q, g_p = g10 - g00, g01 - g00

        A = np.array([[2.0 - h * g_q, -h * g_p],
                      [h * f_q, 2.0 + h * f_p]])
        b = np.array([2.0 * qi + h * g00, 2.0 * pi - h * f00])
        u, v = np.linalg.solve(A, b)

        # The pair is returned as q_i + h g(u, v), not the algebraically
        # equal 2u - q_i, which subtracts nearly equal numbers once the
        # step is small and loses the leading bits of the increment.
        # Writing the increment directly lowers the energy error of a
        # long run by nearly an order of magnitude.
        out = x.copy()
        out[i] = qi + h * (g00 + g_q * u + g_p * v)
        out[m + i] = pi - h * (f00 + f_q * u + f_p * v)
        return out

    def _to_working(self, system, z):
        '''
        Map a state into the working frame, reusing the last one.

        Parameters
        ----------
        system : System
            System to integrate.
        z : np.ndarray
            State, shape (2n,).

        Returns
        -------
        np.ndarray
            The state in the working frame, shape (2n,).
        '''
        if (self._cached_z is not None
                and np.array_equal(z, self._cached_z)):
            return self._cached_x.copy()
        return system.to_working(z)

    def _from_working(self, system, x):
        '''
        Map a state back, remembering it for the next step.

        Parameters
        ----------
        system : System
            System to integrate.
        x : np.ndarray
            State in the working frame, shape (2n,).

        Returns
        -------
        np.ndarray
            The state in the caller's frame, shape (2n,).
        '''
        z = system.from_working(x)
        self._cached_z, self._cached_x = z, x
        return z
