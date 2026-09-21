'''
Energy- and volume-preserving splitting method.

For H = (p1^2 + p2^2)/2 + F(q, p3, ..., pn), F = T + V, n >= 2, one
step is the conjugation  Phi = Psi^{-1} o M o Psi, where:

Psi : (q, p) -> (E, phi, q, p3, ..., pn),  E = H(q, p),
      phi = atan2(p2, p1).
      Inverse: (p1, p2) = rho (cos phi, sin phi), rho = sqrt(2(E - F)).
      Since rho drho = dE - dF,
      dq ^ dp = dE ^ dphi ^ dq ^ dp3 ^ ... ^ dpn, so Psi has unit
      Jacobian.

M   : holds E fixed and composes symplectic Euler substeps, each acting
      in one plane of the new variables with all else frozen:
        (q1, phi)   planar Hamiltonian K1 =  rho sin(phi) = p2,
        (q2, phi)   planar Hamiltonian K2 = -rho cos(phi) = -p1,
        (q', p')    the remaining pairs k >= 3, Hamiltonian F.
      Each substep has unit Jacobian and leaves E unchanged. The (q', p')
      substep is explicit when F is separable, and otherwise solved by
      fixed-point iteration.

Hence Phi conserves H exactly (to round-off) and preserves phase-space
volume exactly; it is not symplectic. The first-order method is M; the
symmetric second-order method is  M*_{h/2} o M_{h/2}, with M* the
adjoint (substeps reversed, each replaced by its adjoint Euler variant).

Momentum. If F does not depend on some qk, k >= 3 (a cyclic coordinate,
as the azimuth is for an axisymmetric potential in cylindrical
coordinates), the (q', p') substep leaves pk unchanged, and no other
substep touches it. The conjugate momentum is then conserved exactly
alongside energy and volume. The symmetry must be expressed through a
cyclic coordinate: in other coordinates the splitting generally breaks
the corresponding conservation law.

The change of variables is singular where p1 = p2 = 0, and the scalar
equations are contractions only while h |dF/dq| / rho < 1.

References
----------
Feng, K. and Shang, Z., 1995. Volume-preserving algorithms for
source-free dynamical systems. Numerische Mathematik, 71(4),
pp.451-463.

Ge, Z. and Marsden, J.E., 1988. Lie-Poisson Hamilton-Jacobi theory
and Lie-Poisson integrators. Physics Letters A, 133(3), pp.134-139.

Tupper, P.F., 2006. A Non-Existence Result for Hamiltonian
Integrators. arXiv preprint math/0607641.
'''

import numpy as np
from scipy.optimize import newton

from .integrator import OneStepMethod


class EnergyVolumeSplit(OneStepMethod):
    '''
    Energy- and volume-preserving splitting in (E, phi) variables.

    See the module docstring for the construction. Each step raises
    ValueError if it reaches the singular set p1 = p2 = 0.

    Parameters
    ----------
    symmetric : bool, optional
        True (default) for the symmetric second-order method
        M*_{h/2} o M_{h/2}; False for the first-order method M.
    xtol : float, optional
        Tolerance for the scalar implicit equations and, for a
        non-separable system, the fixed-point iteration. Default 1e-14.
    maxiter : int, optional
        Iteration limit for the scalar implicit equations and the
        fixed-point iteration. Default 100.

    Notes
    -----
    - First- or second-order accurate in h
    - Conserves energy exactly, to round-off
    - Volume-preserving in phase space, but not symplectic
    - Conserves the momentum conjugate to a cyclic coordinate among
      q3, ..., qn
    - Singular where p1 = p2 = 0
    '''

    def __init__(self, symmetric=True, xtol=1e-14, maxiter=100):
        self.symmetric = symmetric
        self.order = 2 if symmetric else 1
        self.name = 'Energy-volume split'
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
            New state z1, shape (2n,), with H(z1) = H(z0) to round-off.

        Raises
        ------
        ValueError
            If the step reaches the singular set p1 = p2 = 0.
        RuntimeError
            If a scalar implicit equation or the fixed-point iteration
            does not converge.
        '''
        # Psi: to the new variables (E, phi, q, p').
        # q and pr are updated in place by the substeps.
        q, p = system.split(np.array(z0, dtype=float))
        pr = p[2:]
        E = system.H(z0)
        phi = np.arctan2(p[1], p[0])

        if self.symmetric:
            phi = self._forward(system, E, phi, q, pr, 0.5 * h)
            phi = self._adjoint(system, E, phi, q, pr, 0.5 * h)
        else:
            phi = self._forward(system, E, phi, q, pr, h)

        # Psi^{-1}: rebuild the pair on {H = E}.
        rho = self._rho(system, E, q, pr)
        pair = [rho * np.cos(phi), rho * np.sin(phi)]
        return np.concatenate([q, pair, pr])

    # --- change of variables ---------------------------------------------

    @staticmethod
    def _rho(system, E, q, pr):
        '''
        Recover rho = sqrt(p1^2 + p2^2) from the energy.

        Parameters
        ----------
        system : HamiltonianSystem
            System supplying V and T.
        E : float
            Energy, held fixed through the step.
        q : np.ndarray
            Positions, shape (n,).
        pr : np.ndarray
            Momenta (p3, ..., pn), shape (n - 2,).

        Returns
        -------
        float
            rho = sqrt(2 (E - V(q) - T(q, pr))).

        Raises
        ------
        ValueError
            If E - V - T <= 0, the singular set p1 = p2 = 0.
        '''
        r2 = 2.0 * (E - system.V(q) - system.T(q, pr))
        if r2 <= 0.0:
            raise ValueError(
                'Energy-volume split reached the singular set '
                'p1 = p2 = 0; reduce the step size'
            )
        return np.sqrt(r2)

    def _solve(self, fun, x0):
        '''
        Solve a scalar equation fun(x) = 0 by the secant method.

        Parameters
        ----------
        fun : callable
            Scalar function.
        x0 : float
            Initial guess.

        Returns
        -------
        float
            Root, to tolerance xtol.

        Raises
        ------
        RuntimeError
            If the secant method does not converge within maxiter steps.
        '''
        try:
            return newton(fun, x0, tol=self.xtol, maxiter=self.maxiter)
        except RuntimeError as e:
            raise RuntimeError(
                f'{self.name} solver failed to converge: {e}'
            ) from e

    @staticmethod
    def _grad_F(system, q, pr):
        '''
        Evaluate the gradient of F = T + V.

        Parameters
        ----------
        system : HamiltonianSystem
            System supplying grad V and grad T.
        q : np.ndarray
            Positions, shape (n,).
        pr : np.ndarray
            Momenta (p3, ..., pn), shape (n - 2,).

        Returns
        -------
        dF_dq : np.ndarray
            Gradient with respect to q, shape (n,).
        dF_dp : np.ndarray
            Gradient with respect to (p3, ..., pn), shape (n - 2,).
        '''
        dT_dq, dT_dp = system.grad_T(q, pr)
        return system.grad_V(q) + dT_dq, dT_dp

    # --- M: symplectic Euler, implicit in phi (resp. p') -----------------

    def _forward(self, system, E, phi, q, pr, h):
        '''
        Apply M: symplectic Euler substeps implicit in phi (resp. p').

        Substeps act in the (q1, phi) and (q2, phi) planes, then on the
        remaining pairs. q and pr are updated in place.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        E : float
            Energy, held fixed.
        phi : float
            Angle of (p1, p2).
        q : np.ndarray
            Positions, shape (n,); updated in place.
        pr : np.ndarray
            Momenta (p3, ..., pn), shape (n - 2,); updated in place.
        h : float
            Step length for this map.

        Returns
        -------
        float
            Updated angle phi.
        '''
        # (q1, phi), K1 = rho(q1) sin(phi):
        #   phi* = phi + h F_q1 sin(phi*) / rho,  q1* = q1 + h rho cos(phi*)
        rho = self._rho(system, E, q, pr)
        c = h * self._grad_F(system, q, pr)[0][0] / rho
        phi0 = phi
        phi = self._solve(lambda f: f - phi0 - c * np.sin(f), phi0)
        q[0] += h * rho * np.cos(phi)

        # (q2, phi), K2 = -rho(q2) cos(phi):
        #   phi* = phi - h F_q2 cos(phi*) / rho,  q2* = q2 + h rho sin(phi*)
        rho = self._rho(system, E, q, pr)
        c = h * self._grad_F(system, q, pr)[0][1] / rho
        phi0 = phi
        phi = self._solve(lambda f: f - phi0 + c * np.cos(f), phi0)
        q[1] += h * rho * np.sin(phi)

        # (q', p'), k >= 3, Hamiltonian F:
        #   p'* = p' - h F_q'(q, p'*),  q'* = q' + h F_p'(q, p'*)
        # A kick then a drift when F is separable.
        if system.n > 2:
            p0 = pr.copy()

            def kick(x):
                return p0 - h * self._grad_F(system, q, x)[0][2:]

            if system.separable:
                pr[:] = kick(p0)
            else:
                pr[:] = self._fixed_point(kick, p0)
            q[2:] += h * self._grad_F(system, q, pr)[1]
        return phi

    # --- M*: adjoint, substeps reversed, implicit in q_k -----------------

    def _adjoint(self, system, E, phi, q, pr, h):
        '''
        Apply M*: the substeps of M reversed, each implicit in q_k.

        q and pr are updated in place.

        Parameters
        ----------
        system : HamiltonianSystem
            System to integrate.
        E : float
            Energy, held fixed.
        phi : float
            Angle of (p1, p2).
        q : np.ndarray
            Positions, shape (n,); updated in place.
        pr : np.ndarray
            Momenta (p3, ..., pn), shape (n - 2,); updated in place.
        h : float
            Step length for this map.

        Returns
        -------
        float
            Updated angle phi.
        '''
        # (q', p'), k >= 3, Hamiltonian F:
        #   q'* = q' + h F_p'(q*, p'),  p'* = p' - h F_q'(q*, p')
        # A drift then a kick when F is separable.
        if system.n > 2:
            q0 = q[2:].copy()
            trial = q.copy()

            def drift(x):
                trial[2:] = x
                return q0 + h * self._grad_F(system, trial, pr)[1]

            if system.separable:
                q[2:] = drift(q0)
            else:
                q[2:] = self._fixed_point(drift, q0)
            pr -= h * self._grad_F(system, q, pr)[0][2:]

        # (q2, phi):  q2* = q2 + h rho(q2*) sin(phi),
        #             phi* = phi - h F_q2(q2*) cos(phi) / rho(q2*)
        s = np.sin(phi)
        q[1] = self._solve_position(system, E, q, pr, 1, h * s)
        rho = self._rho(system, E, q, pr)
        phi = phi - h * self._grad_F(system, q, pr)[0][1] * np.cos(phi) / rho

        # (q1, phi):  q1* = q1 + h rho(q1*) cos(phi),
        #             phi* = phi + h F_q1(q1*) sin(phi) / rho(q1*)
        c = np.cos(phi)
        q[0] = self._solve_position(system, E, q, pr, 0, h * c)
        rho = self._rho(system, E, q, pr)
        phi = phi + h * self._grad_F(system, q, pr)[0][0] * np.sin(phi) / rho
        return phi

    def _solve_position(self, system, E, q, pr, i, a):
        '''
        Solve x = q[i] + a rho(q with q[i] = x) for x.

        Parameters
        ----------
        system : HamiltonianSystem
            System supplying V and T.
        E : float
            Energy, held fixed.
        q : np.ndarray
            Positions, shape (n,); not modified.
        pr : np.ndarray
            Momenta (p3, ..., pn), shape (n - 2,).
        i : int
            Index of the position being solved for.
        a : float
            Coefficient, h cos(phi) or h sin(phi).

        Returns
        -------
        float
            New value of q[i].
        '''
        qi = q[i]
        trial = q.copy()

        def residual(x):
            trial[i] = x
            return x - qi - a * self._rho(system, E, trial, pr)

        return self._solve(residual, qi + a * self._rho(system, E, q, pr))
