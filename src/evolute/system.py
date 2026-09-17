'''
Separable Hamiltonian systems.

State ordering is z = (q, p) with q, p in R^n, n >= 2, so dim = 2n and
the canonical equations are dz/dt = J grad H(z), J = [[0, I], [-I, 0]].

Every system has the form

    H(q, p) = (p1^2 + p2^2) / 2 + T(p3, ..., pn) + V(q),

where (p1, p2) = (p[0], p[1]) is the isotropic momentum pair. A system
is built from V and grad V (and optionally T and grad T), so
separability is enforced by the interface. System-specific definitions
belong in user scripts, not in this package.
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


def _default_T(p_rest):
    '''
    Evaluate the default kinetic energy |(p3, ..., pn)|^2 / 2.

    Parameters
    ----------
    p_rest : np.ndarray
        Momenta (p3, ..., pn), shape (n - 2,).

    Returns
    -------
    float
        Kinetic energy of the remaining momenta.
    '''
    return 0.5 * (p_rest @ p_rest)


def _default_grad_T(p_rest):
    '''
    Evaluate the gradient of the default kinetic energy.

    Parameters
    ----------
    p_rest : np.ndarray
        Momenta (p3, ..., pn), shape (n - 2,).

    Returns
    -------
    np.ndarray
        The momenta themselves, shape (n - 2,).
    '''
    return p_rest


class HamiltonianSystem:
    '''
    Separable Hamiltonian H = (p1^2 + p2^2) / 2 + T(p3, ..., pn) + V(q).

    The pair (p1, p2) always has the isotropic kinetic energy above;
    EnergyVolumeSplit replaces it by energy and angle variables. The
    remaining momenta carry their own kinetic energy T.

    Parameters
    ----------
    n : int
        Degrees of freedom, at least 2.
    V, grad_V : callable
        Potential and its gradient, functions of q, shape (n,).
    T, grad_T : callable, optional
        Kinetic energy of (p3, ..., pn), shape (n - 2,), and its
        gradient. Supply both or neither; the default is
        T = |(p3, ..., pn)|^2 / 2. Ignored if n = 2.
    invariants : callable, optional
        invariants(q, p) returning a dict of additional first
        integrals, name -> value.
    name : str, optional
        Display name, used in labels and titles.

    Raises
    ------
    ValueError
        If n < 2, or if only one of T and grad_T is supplied.
    '''

    def __init__(self, n, V, grad_V, T=None, grad_T=None,
                 invariants=None, name=None):
        if n < 2:
            raise ValueError("n must be at least 2")
        if (T is None) != (grad_T is None):
            raise ValueError("supply both T and grad_T, or neither")
        self.n = n
        self.V = V
        self.grad_V = grad_V
        self.T = T if T is not None else _default_T
        self.grad_T = grad_T if grad_T is not None else _default_grad_T
        self._invariants = invariants
        self.name = name or "system"

    def __repr__(self):
        return f"HamiltonianSystem(name={self.name!r}, n={self.n})"

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

    def kinetic(self, p):
        '''
        Evaluate the full kinetic energy (p1^2 + p2^2) / 2 + T.

        Parameters
        ----------
        p : np.ndarray
            Momenta, shape (n,).

        Returns
        -------
        float
            Kinetic energy.
        '''
        return 0.5 * (p[0] * p[0] + p[1] * p[1]) + self.T(p[2:])

    def grad_kinetic(self, p):
        '''
        Evaluate the gradient of the full kinetic energy.

        Parameters
        ----------
        p : np.ndarray
            Momenta, shape (n,).

        Returns
        -------
        np.ndarray
            Gradient with respect to p, shape (n,).
        '''
        g = np.empty(self.n)
        g[:2] = p[:2]
        g[2:] = self.grad_T(p[2:])
        return g

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
        return self.kinetic(p) + self.V(q)

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
            (grad V(q), grad K(p)), shape (2n,).
        '''
        q, p = self.split(z)
        return np.concatenate([self.grad_V(q), self.grad_kinetic(p)])

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

    def energy_error(self, z0, z, relative=True):
        '''
        Measure the energy error of z against the initial state z0.

        Use relative=False when H(z0) is zero in exact arithmetic (for
        example rays on H = 0): in floating point H(z0) is then
        round-off, and dividing by it is meaningless.

        Parameters
        ----------
        z0 : np.ndarray
            Initial state, shape (2n,).
        z : np.ndarray
            Current state, shape (2n,).
        relative : bool, optional
            Divide by |H(z0)| if True (default).

        Returns
        -------
        float
            |H(z) - H(z0)|, relative or absolute.
        '''
        H0 = self.H(z0)
        err = abs(self.H(z) - H0)
        return err / abs(H0) if relative else err

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
