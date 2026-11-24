'''
Hamiltonian systems in canonical coordinates.

State ordering is z = (q, p) with q, p in R^n, n >= 2, so dim = 2n and
the canonical equations are dz/dt = J grad H(z), J = [[0, I], [-I, 0]].

System is the base, and carries everything that follows from H alone:
the structure matrix, the vector field and the energy error. It is not
used directly; HamiltonianSystem is the concrete class.

HamiltonianSystem, the class the methods here are built around, has:

    H(q, p) = (p1^2 + p2^2) / 2 + T(q, p3, ..., pn) + V(q),

where (p1, p2) = (p[0], p[1]) is the isotropic momentum pair. The pair
never couples to q, but T may, so the system need not be separable;
the default T = |(p3, ..., pn)|^2 / 2 makes it separable. A system is
built from V and grad V (and optionally T and grad T).

System-specific definitions belong in user scripts, not in this
package.
'''

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _canonical_J(n):
    '''
    Build the canonical structure matrix on R^{2n}.

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
    J = np.zeros((2 * n, 2 * n))
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    J.flags.writeable = False
    return J


@lru_cache(maxsize=None)
def _zeros(n):
    '''
    Build a zero vector of length n.

    Cached and read-only, as _canonical_J is, so the many calls that
    need a zero gradient share one array instead of allocating. Any
    attempt to write to it raises rather than corrupting the others.

    Parameters
    ----------
    n : int
        Length.

    Returns
    -------
    np.ndarray
        Read-only zero vector, shape (n,).
    '''
    z = np.zeros(n)
    z.flags.writeable = False
    return z


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
        Read-only zeros, shape (n,).
    dT_dp : np.ndarray
        The momenta themselves, shape (n - 2,).
    '''
    return _zeros(q.shape[0]), p_rest


class System:
    '''
    A Hamiltonian system in canonical coordinates on R^{2n}.

    The base of HamiltonianSystem, holding what follows from H alone.
    A subclass supplies H and grad_H, and everything here is derived
    from those and from n.

    A method needing the Hamiltonian in another canonical frame reaches
    it through to_working, from_working and working_grad_H. With no
    transform these are the identity and grad_H itself, so a method
    written against them runs on any system.

    Parameters
    ----------
    n : int
        Degrees of freedom, at least 2.
    name : str, optional
        Display name, shown in the repr.
    transform : np.ndarray, optional
        Canonical matrix S with x = S z, giving the working frame.
        The default is the identity, so the two frames coincide. S
        must satisfy S' J S = J, or the system in the working frame
        is not the one the caller posed.

    Attributes
    ----------
    n : int
        Degrees of freedom.
    name : str
        Display name.
    transform : np.ndarray or None
        The matrix S, or None if the frames coincide.

    Raises
    ------
    ValueError
        If n < 2, if transform is not of shape (2n, 2n), or if it is
        not canonical.
    '''

    def __init__(self, n, name=None, transform=None):
        if n < 2:
            raise ValueError(f'n must be at least 2, got {n}')
        self.n = n
        self.name = name or 'system'
        if transform is None:
            self.transform = None
            self._inverse = None
        else:
            S = np.asarray(transform, dtype=float)
            if S.shape != (2 * n, 2 * n):
                raise ValueError(
                    f'transform must have shape {(2 * n, 2 * n)}, got '
                    f'{S.shape}')
            J = _canonical_J(n)
            if not np.allclose(S.T @ J @ S, J, atol=1e-12):
                raise ValueError("transform is not canonical: "
                                 "S' J S != J")
            self.transform = S
            self._inverse = np.linalg.inv(S)

    def __repr__(self):
        return f'{type(self).__name__}(name={self.name!r}, n={self.n})'

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
        q : np.ndarray
            View of the positions, shape (n,). Writing to it writes
            to z.
        p : np.ndarray
            View of the momenta, shape (n,). Writing to it writes
            to z.
        '''
        return z[:self.n], z[self.n:]

    def H(self, z):
        '''
        Evaluate the Hamiltonian; defined by the subclass.

        Parameters
        ----------
        z : np.ndarray
            State (q, p), shape (2n,).

        Raises
        ------
        NotImplementedError
            Always; System is not used directly.
        '''
        raise NotImplementedError('subclasses define H')

    def grad_H(self, z):
        '''
        Evaluate the Hamiltonian gradient; defined by the subclass.

        Parameters
        ----------
        z : np.ndarray
            State (q, p), shape (2n,).

        Raises
        ------
        NotImplementedError
            Always; System is not used directly.
        '''
        raise NotImplementedError('subclasses define grad_H')

    def J(self, z):
        '''
        Evaluate the structure matrix at z.

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
        return _canonical_J(self.n)

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

    # The working frame: a second canonical frame a method may need.

    def to_working(self, z):
        '''
        Map a state into the working frame.

        Parameters
        ----------
        z : np.ndarray
            State, shape (2n,).

        Returns
        -------
        np.ndarray
            S z, or z itself if there is no transform.
        '''
        return z if self.transform is None else self.transform @ z

    def from_working(self, x):
        '''
        Map a state back from the working frame.

        Parameters
        ----------
        x : np.ndarray
            State in the working frame, shape (2n,).

        Returns
        -------
        np.ndarray
            S^-1 x, or x itself if there is no transform.
        '''
        return x if self.transform is None else self._inverse @ x

    def working_grad_H(self, x):
        '''
        Evaluate the gradient of H in the working frame.

        The Hamiltonian there is H(S^-1 x), so its gradient is
        S^-T grad H(S^-1 x). Nothing need be rewritten in the working
        frame: this follows from grad_H and S alone.

        Parameters
        ----------
        x : np.ndarray
            State in the working frame, shape (2n,).

        Returns
        -------
        np.ndarray
            Gradient in that frame, shape (2n,).
        '''
        if self.transform is None:
            return self.grad_H(x)
        return self._inverse.T @ self.grad_H(self._inverse @ x)

    # Diagnostics

    def energy_error(self, z0, z):
        '''
        Measure the relative energy error of z against z0.

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


class HamiltonianSystem(System):
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
    name : str, optional
        Display name, shown in the repr.
    transform : np.ndarray, optional
        Canonical matrix S with x = S z, as in System, giving the
        frame a method such as McLachlan-Quispel's works in. It does
        not affect the methods here, which all work in z.

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
        If n < 2, if only one of T and grad_T is supplied, or if
        transform is not a canonical (2n, 2n) matrix.
    '''

    def __init__(self, n, V, grad_V, T=None, grad_T=None,
                 separable=None, name=None, transform=None):
        if (T is None) != (grad_T is None):
            raise ValueError(
                'T and grad_T must be given together, got only '
                f'{"T" if T is not None else "grad_T"}')
        super().__init__(n, name, transform)
        self.V = V
        self.grad_V = grad_V
        self.T = T if T is not None else _default_T
        self.grad_T = grad_T if grad_T is not None else _default_grad_T
        self.separable = (T is None) if separable is None else separable
        # With the default T, grad_kinetic is (0, p) and needs no
        # assembly; a supplied T takes the general path.
        self._default_kinetic = T is None

    def __repr__(self):
        return (f'HamiltonianSystem(name={self.name!r}, n={self.n}, '
                f'separable={self.separable})')

    # Full Hamiltonian

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
            separable system, and read-only with the default T.
        dK_dp : np.ndarray
            Gradient with respect to p, shape (n,).
        '''
        if self._default_kinetic:
            # dK/dp is (p1, p2, p3, ..., pn), which is p itself, and
            # dK/dq is zero. Copied, since the caller may write to it.
            return _zeros(self.n), p.copy()
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
        out = np.empty(2 * self.n)
        out[:self.n] = self.grad_V(q) + dK_dq
        out[self.n:] = dK_dp
        return out
