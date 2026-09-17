'''
Energy- and volume-preserving splitting method.

For separable H = (p1^2 + p2^2)/2 + T(p3, ..., pn) + V(q), n >= 2, one
step is the conjugation  Phi = Psi^{-1} o M o Psi, where:

Psi : (q, p) -> (E, phi, q, p3, ..., pn),  E = H(q, p),
      phi = atan2(p2, p1).
      Inverse: (p1, p2) = rho (cos phi, sin phi), rho = sqrt(2(E - F)),
      F = T + V. Since rho drho = dE - dF,
      dq ^ dp = dE ^ dphi ^ dq ^ dp3 ^ ... ^ dpn, so Psi has unit
      Jacobian.

M   : holds E fixed and composes symplectic Euler substeps, each acting
      in one plane of the new variables with all else frozen:
        (q1, phi)   planar Hamiltonian K1 =  rho sin(phi) = p2,
        (q2, phi)   planar Hamiltonian K2 = -rho cos(phi) = -p1,
        (q', p')    the remaining pairs k >= 3, Hamiltonian F.
      Each substep has unit Jacobian and leaves E unchanged.

Hence Phi conserves H exactly (to round-off) and preserves phase-space
volume exactly; it is not symplectic. The first-order method is M; the
symmetric second-order method is  M*_{h/2} o M_{h/2}, with M* the
adjoint (substeps reversed, each replaced by its adjoint Euler variant).

The change of variables is singular where p1 = p2 = 0, and the scalar
equations are contractions only while h |grad V| / rho < 1.
'''

import numpy as np
from scipy.optimize import newton

from evolute.integrator import OneStepMethod


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
        Tolerance for the scalar implicit equations.
    '''

    def __init__(self, symmetric=True, xtol=1e-14):
        self.symmetric = symmetric
        self.order = 2 if symmetric else 1
        self.name = "Energy-volume split"
        self.xtol = xtol

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
            rho = sqrt(2 (E - V(q) - T(pr))).

        Raises
        ------
        ValueError
            If E - V - T <= 0, the singular set p1 = p2 = 0.
        '''
        r2 = 2.0 * (E - system.V(q) - system.T(pr))
        if r2 <= 0.0:
            raise ValueError(
                "EnergyVolumeSplit: p1^2 + p2^2 <= 0 in the new "
                "variables (singular set p1 = p2 = 0); reduce the step size"
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
        '''
        return newton(fun, x0, tol=self.xtol, maxiter=100)

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
        #   phi* = phi + h V_q1 sin(phi*) / rho,  q1* = q1 + h rho cos(phi*)
        rho = self._rho(system, E, q, pr)
        c = h * system.grad_V(q)[0] / rho
        phi0 = phi
        phi = self._solve(lambda f: f - phi0 - c * np.sin(f), phi0)
        q[0] += h * rho * np.cos(phi)

        # (q2, phi), K2 = -rho(q2) cos(phi):
        #   phi* = phi - h V_q2 cos(phi*) / rho,  q2* = q2 + h rho sin(phi*)
        rho = self._rho(system, E, q, pr)
        c = h * system.grad_V(q)[1] / rho
        phi0 = phi
        phi = self._solve(lambda f: f - phi0 + c * np.cos(f), phi0)
        q[1] += h * rho * np.sin(phi)

        # (q', p'), k >= 3: kick then drift (separable, so explicit)
        if system.n > 2:
            pr -= h * system.grad_V(q)[2:]
            q[2:] += h * system.grad_T(pr)
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
        # (q', p'), k >= 3: drift then kick
        if system.n > 2:
            q[2:] += h * system.grad_T(pr)
            pr -= h * system.grad_V(q)[2:]

        # (q2, phi):  q2* = q2 + h rho(q2*) sin(phi),
        #             phi* = phi - h V_q2(q2*) cos(phi) / rho(q2*)
        s = np.sin(phi)
        q[1] = self._solve_position(system, E, q, pr, 1, h * s)
        rho = self._rho(system, E, q, pr)
        phi = phi - h * system.grad_V(q)[1] * np.cos(phi) / rho

        # (q1, phi):  q1* = q1 + h rho(q1*) cos(phi),
        #             phi* = phi + h V_q1(q1*) sin(phi) / rho(q1*)
        c = np.cos(phi)
        q[0] = self._solve_position(system, E, q, pr, 0, h * c)
        rho = self._rho(system, E, q, pr)
        phi = phi + h * system.grad_V(q)[0] * np.sin(phi) / rho
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
