'''
Maxwell fish-eye lens (ray optics): energy error of each method.
'''

import numpy as np

import evolute
from evolute import (EnergyVolumeSplit, Gonzalez, ItohAbe,
                     Symplectic)


def maxwell_fisheye(n0=1.0, R=1.0):
    '''
    Build the Maxwell fish-eye lens as a ray-optics Hamiltonian.

    H = (|p|^2 - n(q)^2) / 2 with refractive index
    n(q) = n0 / (1 + |q|^2 / R^2), in three dimensions. Rays lie on
    H = 0. The default T = p3^2 / 2 completes |p|^2 / 2. The index is
    attached to the system as `index`, for use by ray_state.

    Parameters
    ----------
    n0 : float, optional
        Refractive index at the centre.
    R : float, optional
        Length scale; the index falls to n0 / 2 at |q| = R.

    Returns
    -------
    HamiltonianSystem
        System with n = 3.
    '''

    def index(q):
        return n0 / (1.0 + q @ q / R ** 2)

    def V(q):
        return -0.5 * index(q) ** 2

    def grad_V(q):
        # -n grad n, with grad n = -2 n0 q / (R^2 (1 + |q|^2/R^2)^2)
        return 2.0 * n0 ** 2 * q / (R ** 2 * (1.0 + q @ q / R ** 2) ** 3)

    system = evolute.HamiltonianSystem(3, V, grad_V,
                                       name="Maxwell fish-eye")
    system.index = index
    return system


def ray_state(system, q, direction):
    '''
    Build an initial state on the ray surface H = 0.

    Parameters
    ----------
    system : HamiltonianSystem
        Fish-eye system from maxwell_fisheye.
    q : array_like
        Initial position, shape (3,).
    direction : array_like
        Initial ray direction, shape (3,); need not be normalised.

    Returns
    -------
    np.ndarray
        State (q, p) with p along `direction` and |p| = n(q).
    '''
    q = np.asarray(q, dtype=float)
    d = np.asarray(direction, dtype=float)
    p = system.index(q) * d / np.linalg.norm(d)
    return np.concatenate([q, p])


def main(h=0.05, steps=4000):
    '''
    Print the maximum absolute energy error of each method.

    Parameters
    ----------
    h : float, optional
        Step size.
    steps : int, optional
        Number of steps.
    '''
    system = maxwell_fisheye()
    z0 = ray_state(system, [1.0, 0.2, 0.1], [0.3, 1.0, 0.2])
    methods = (Symplectic(symmetric=False), Symplectic(),
               ItohAbe(), Gonzalez(),
               EnergyVolumeSplit(symmetric=False), EnergyVolumeSplit())
    for method in methods:
        z, err = z0.copy(), 0.0
        for _ in range(steps):
            z = method.step(system, z, h)
            err = max(err, system.energy_error(z0, z, relative=False))
        label = f"{method.name} (order {method.order})"
        print(f"{label:32s} max |H - H0| {err:.2e}")


if __name__ == "__main__":
    main()
