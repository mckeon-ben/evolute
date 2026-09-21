'''
Hamiltonian systems with an isotropic momentum pair.

State ordering is z = (q, p) with q, p in R^n, n >= 2, so dim = 2n and
the canonical equations are dz/dt = J grad H(z), J = [[0, I], [-I, 0]].

Every system has the form

    H(q, p) = (p1^2 + p2^2) / 2 + T(q, p3, ..., pn) + V(q),

where (p1, p2) = (p[0], p[1]) is the isotropic momentum pair. The pair
never couples to q, but T may, so the system need not be separable;
the default T = |(p3, ..., pn)|^2 / 2 makes it separable. A system is
built from V and grad V (and optionally T and grad T). System-specific
definitions belong in user scripts, not in this package.
'''

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _canonical_J(n):
    '''
    Build and cache the canonical structure matrix on R^{2n}.

    Parameters
    ----------
    n : int
        Degrees of freedom.

    Returns
    -------
    np.ndarray
        Read-only (2n, 2n) matrix [[0, I], [-I, 0]].
    '''
    J = np.zeros((2 * n, 2 * n))
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    J.flags.writeable = False
    return J


def canonical_J(n):
    '''
    Return the canonical structure matrix on R^{2n}.

    The matrix is cached and read-only, so every caller shares one
    copy.

    Parameters
    ----------
    n : int
        Degrees of freedom.

    Returns
    -------
    np.ndarray
        Read-only (2n, 2n) matrix [[0, I], [-I, 0]].
    '''
    return _canonical_J(n)


def _default_T(q, p_rest):
    '''
    Evaluate the default kinetic energy |(p3, ..., pn)|^2 / 2.

    Parameters
    ----------
    q : np.ndarray
        Positions, shape (n,); unused.
    p_rest : np.ndarray
        Momenta (p3, ..., pn), shape (n - 2,).

    Returns
    -------
    float
        Kinetic energy of the remaining momenta.
    '''
    return 0.5 * (p_rest @ p_rest)


def _default_grad_T(q, p_rest):
    '''
    Evaluate the gradient of the default kinetic energy.

    Parameters
    ----------
    q : np.ndarray
        Positions, shape (n,).
    p_rest : np.ndarray
        Momenta (p3, ..., pn), shape (n - 2,).

    Returns
    -------
    dT_dq : np.ndarray
        Zeros, shape (n,).
    dT_dp : np.ndarray
        The momenta themselves, shape (n - 2,).
    '''
    return np.zeros_like(q), p_rest


class HamiltonianSystem:
    '''
    Hamiltonian H = (p1^2 + p2^2) / 2 + T(q, p3, ..., pn) + V(q).

    The pair (p1, p2) always has the isotropic kinetic energy above;
    EnergyVolumeSplit replaces it by energy and angle variables. The
    remaining momenta carry their own kinetic energy T, which may also
    depend on q, as in curvilinear coordinates.

    Parameters
    ----------
    n : int
        Degrees of freedom, at least 2.
    V, grad_V : callable
        Potential and its gradient, functions of q, shape (n,).
    T, grad_T : callable, optional
        T(q, p_rest), the kinetic energy of p_rest = (p3, ..., pn),
        shape (n - 2,), and grad_T(q, p_rest), returning the pair
        (dT/dq, dT/dp_rest) with shapes (n,) and (n - 2,). Supply both
        or neither; the default is T = |(p3, ..., pn)|^2 / 2.
    separable : bool, optional
        Whether T is independent of q, so that H = K(p) + V(q). The
        default is True when T is omitted and False when it is
        supplied; pass True for a supplied T that depends on the
        momenta alone. For a separable system the methods skip their
        fixed-point iterations: the symplectic methods are then
        explicit, as is the last substep of EnergyVolumeSplit.
    invariants : callable, optional
        invariants(q, p) returning a dict of additional first
        integrals, name -> value.
    name : str, optional
        Display name, used in labels and titles.

    Attributes
    ----------
    n : int
        Degrees of freedom.
    V, grad_V : callable
        Potential and its gradient.
    T, grad_T : callable
        Kinetic energy of (p3, ..., pn) and its gradient, the defaults
        if none were supplied.
    separable : bool
        Whether T is independent of q.
    name : str
        Display name.

    Raises
    ------
    ValueError
        If n < 2, or if only one of T and grad_T is supplied.

    Examples
    --------
    Planar harmonic oscillator, separable by default:

    >>> system = HamiltonianSystem(2, V=lambda q: 0.5 * (q @ q),
    ...                            grad_V=lambda q: q)
    >>> float(system.H(np.array([1., 0., 0., 1.])))
    1.0

    Axisymmetric oscillator in cylindrical coordinates (R, z, phi), with
    T = L^2 / (2 R^2) for the momentum L conjugate to phi:

    >>> def T(q, p_rest):
    ...     return 0.5 * p_rest[0] ** 2 / q[0] ** 2
    >>> def grad_T(q, p_rest):
    ...     L, R = p_rest[0], q[0]
    ...     return np.array([-L * L / R ** 3, 0., 0.]), p_rest / R ** 2
    >>> system = HamiltonianSystem(
    ...     3, V=lambda q: 0.5 * (q[0] ** 2 + q[1] ** 2),
    ...     grad_V=lambda q: np.array([q[0], q[1], 0.]),
    ...     T=T, grad_T=grad_T)
    >>> system.separable
    False
    '''

    def __init__(self, n, V, grad_V, T=None, grad_T=None,
                 separable=None, invariants=None, name=None):
        if n < 2:
            raise ValueError(f'n must be at least 2, got {n}')
        if (T is None) != (grad_T is None):
            raise ValueError('supply both T and grad_T, or neither')
        self.n = n
        self.V = V
        self.grad_V = grad_V
        self.T = T if T is not None else _default_T
        self.grad_T = grad_T if grad_T is not None else _default_grad_T
        self.separable = (T is None) if separable is None else separable
        self._invariants = invariants
        self.name = name or 'system'

    def __repr__(self):
        return (f'HamiltonianSystem(name={self.name!r}, n={self.n}, '
                f'separable={self.separable})')

    @property
    def dim(self):
        '''
        Phase-space dimension, 2n.
        '''
        return 2 * self.n

    def split(self, z):
        '''
        Split a state into positions and momenta.

        Parameters
        ----------
        z : np.ndarray
            State (q, p), shape (2n,).

        Returns
        -------
        q, p : np.ndarray
            Views of z, each shape (n,). Writing to them writes to z.
        '''
        return z[:self.n], z[self.n:]

    # --- full Hamiltonian ------------------------------------------------

    def kinetic(self, q, p):
        '''
        Evaluate the full kinetic energy (p1^2 + p2^2) / 2 + T.

        Parameters
        ----------
        q, p : np.ndarray
            Positions and momenta, each shape (n,).

        Returns
        -------
        float
            Kinetic energy.
        '''
        return 0.5 * (p[0] * p[0] + p[1] * p[1]) + self.T(q, p[2:])

    def grad_kinetic(self, q, p):
        '''
        Evaluate the gradient of the full kinetic energy.

        Parameters
        ----------
        q, p : np.ndarray
            Positions and momenta, each shape (n,).

        Returns
        -------
        dK_dq : np.ndarray
            Gradient with respect to q, shape (n,); zero for a
            separable system.
        dK_dp : np.ndarray
            Gradient with respect to p, shape (n,).
        '''
        dT_dq, dT_dp = self.grad_T(q, p[2:])
        dK_dp = np.empty(self.n)
        dK_dp[:2] = p[:2]
        dK_dp[2:] = dT_dp
        return dT_dq, dK_dp

    def H(self, z):
        '''
        Evaluate the Hamiltonian.

        Parameters
        ----------
        z : np.ndarray
            State (q, p), shape (2n,).

        Returns
        -------
        float
            Energy H(z).
        '''
        q, p = self.split(z)
        return self.kinetic(q, p) + self.V(q)

    def grad_H(self, z):
        '''
        Evaluate the gradient of the Hamiltonian.

        Parameters
        ----------
        z : np.ndarray
            State (q, p), shape (2n,).

        Returns
        -------
        np.ndarray
            (grad V + dK/dq, dK/dp), shape (2n,).
        '''
        q, p = self.split(z)
        dK_dq, dK_dp = self.grad_kinetic(q, p)
        return np.concatenate([self.grad_V(q) + dK_dq, dK_dp])

    def J(self, z):
        '''
        Return the structure matrix at z.

        Canonical, so independent of z; the argument keeps the
        signature open to Poisson systems.

        Parameters
        ----------
        z : np.ndarray
            State, shape (2n,).

        Returns
        -------
        np.ndarray
            Read-only (2n, 2n) canonical matrix.
        '''
        return canonical_J(self.n)

    def vector_field(self, z):
        '''
        Evaluate the Hamiltonian vector field J grad H(z).

        Parameters
        ----------
        z : np.ndarray
            State, shape (2n,).

        Returns
        -------
        np.ndarray
            Time derivative dz/dt, shape (2n,).
        '''
        return self.J(z) @ self.grad_H(z)

    # --- diagnostics -----------------------------------------------------

    def energy_error(self, z0, z):
        '''
        Measure the relative energy error of z against the initial state z0.

        H(z0) must be nonzero.

        Parameters
        ----------
        z0 : np.ndarray
            Initial state, shape (2n,).
        z : np.ndarray
            Current state, shape (2n,).

        Returns
        -------
        float
            |H(z) - H(z0)| / |H(z0)|.
        '''
        H0 = self.H(z0)
        return abs(self.H(z) - H0) / abs(H0)

    def invariants(self, z):
        '''
        Evaluate the additional first integrals supplied at construction.

        Parameters
        ----------
        z : np.ndarray
            State, shape (2n,).

        Returns
        -------
        dict
            Name -> value; empty if none were supplied.
        '''
        if self._invariants is None:
            return {}
        return self._invariants(*self.split(z))
