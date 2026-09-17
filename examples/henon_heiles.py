'''
Henon-Heiles system: energy error of each method.
'''

import numpy as np

import evolute
from evolute import (EnergyVolumeSplit, Gonzalez, ItohAbe,
                     Symplectic)


def henon_heiles(lam=1.0):
    '''
    Build the Henon-Heiles system.

    The potential is V = (x^2 + y^2) / 2 + lam (x^2 y - y^3 / 3); the
    classical system has lam = 1.

    Parameters
    ----------
    lam : float, optional
        Strength of the cubic coupling.

    Returns
    -------
    HamiltonianSystem
        System with n = 2.
    '''

    def V(q):
        x, y = q
        return 0.5 * (x * x + y * y) + lam * (x * x * y - y ** 3 / 3.0)

    def grad_V(q):
        x, y = q
        return np.array([x + 2.0 * lam * x * y,
                         y + lam * (x * x - y * y)])

    return evolute.HamiltonianSystem(2, V, grad_V, name='Hénon-Heiles')


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
    system = henon_heiles()
    z0 = np.array([0.1, -0.2, 0.3, 0.15])
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
