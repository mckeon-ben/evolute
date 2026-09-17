'''
Kepler problem: energy error of each method.
'''

import numpy as np

import evolute
from evolute import (EnergyVolumeSplit, Gonzalez, ItohAbe,
                     Symplectic)


def kepler(mu=1.0):
    '''
    Build the planar Kepler problem, V(q) = -mu / |q|.

    Supplies the angular momentum and the Laplace-Runge-Lenz vector as
    invariants.

    Parameters
    ----------
    mu : float, optional
        Gravitational parameter.

    Returns
    -------
    HamiltonianSystem
        System with n = 2.
    '''

    def V(q):
        return -mu / np.linalg.norm(q)

    def grad_V(q):
        return mu * q / np.linalg.norm(q) ** 3

    def invariants(q, p):
        L = q[0] * p[1] - q[1] * p[0]
        A = np.array([p[1] * L, -p[0] * L]) - mu * q / np.linalg.norm(q)
        return {'angular_momentum': L, 'lrl_x': A[0], 'lrl_y': A[1]}

    return evolute.HamiltonianSystem(2, V, grad_V, invariants=invariants,
                                     name='Kepler')


def main(h=0.05, steps=4000):
    '''
    Print the maximum energy error of each method over one run.

    Parameters
    ----------
    h : float, optional
        Step size.
    steps : int, optional
        Number of steps.
    '''
    system = kepler()
    z0 = np.array([1.0, 0.1, 0.2, 0.9])
    methods = (Symplectic(symmetric=False), Symplectic(),
               ItohAbe(), Gonzalez(),
               EnergyVolumeSplit(symmetric=False), EnergyVolumeSplit())
    for method in methods:
        z, err = z0.copy(), 0.0
        for _ in range(steps):
            z = method.step(system, z, h)
            err = max(err, system.energy_error(z0, z))
        label = f'{method.name} (order {method.order})'
        print(f'{label:32s} max energy error {err:.2e}')


if __name__ == '__main__':
    main()
