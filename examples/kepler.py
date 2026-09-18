'''
Kepler problem.

Integrates every method over a fixed time at a sequence of step counts,
and over a long run at a single step size, then writes the final states
and the energy history to a JSON data file.

The reference solution is exact: Kepler's equation is solved by the
Lagrange-Bessel series.
'''

import json
import os
import time

import numpy as np
from scipy.special import jv

import evolute
from evolute import (EnergyVolumeSplit, Gonzalez, ItohAbe, Symplectic)


# Written into the data folder beside this script, wherever it is
# run from.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data')
DATA_FILE = os.path.join(DATA_DIR, 'kepler.json')

# Names the figure.
EXPERIMENT = 'Kepler problem'

# Gravitational parameter and initial state (q1, q2, p1, p2).
MU = 1.0
Z_START = np.array([1.0, 0.1, 0.2, 0.9])

# Convergence study over 5.5 orbital periods. The fraction is
# deliberate: the O(h) error term of some first-order methods vanishes
# whenever the orbit returns to its start, so a whole number of periods
# would make them look second order.
PERIODS = 5.5
N_LIST = [256, 512, 1024, 2048, 4096]

# Long run for the energy history, about 39 orbits, sampled to keep the
# file small.
ENERGY_H = 0.05
ENERGY_STEPS = 4000
ENERGY_SAMPLE = 10
RELATIVE_ENERGY = True

# Display name -> method. The class name is stored in the data file, so
# the two energy-volume split variants stay distinguishable.
METHODS = {
    'Symplectic Euler': Symplectic(symmetric=False),
    'Störmer-Verlet': Symplectic(),
    'Itoh-Abe': ItohAbe(),
    'Gonzalez': Gonzalez(),
    'Energy-volume split (1)': EnergyVolumeSplit(symmetric=False),
    'Energy-volume split (2)': EnergyVolumeSplit(),
}


def kepler(mu=MU):
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


def _elements(z0, mu):
    '''
    Orbital elements of the bound orbit through z0.

    Parameters
    ----------
    z0 : np.ndarray
        Initial state, shape (4,).
    mu : float
        Gravitational parameter.

    Returns
    -------
    a, e : float
        Semi-major axis and eccentricity.
    P, Q : np.ndarray
        Perifocal basis vectors, each shape (2,).
    E0 : float
        Eccentric anomaly at t = 0.

    Raises
    ------
    ValueError
        If the orbit is unbound or circular.
    '''
    q, p = z0[:2], z0[2:]
    r, v2, rv = np.linalg.norm(q), p @ p, q @ p
    energy = 0.5 * v2 - mu / r
    if energy >= 0.0:
        raise ValueError('kepler_exact: orbit is not bound (H >= 0)')
    a = -mu / (2.0 * energy)
    evec = ((v2 - mu / r) * q - rv * p) / mu
    e = np.linalg.norm(evec)
    if e < 1e-12:
        raise ValueError('kepler_exact: circular orbit not supported')
    P = evec / e
    L = q[0] * p[1] - q[1] * p[0]
    Q = np.sign(L) * np.array([-P[1], P[0]])
    E0 = np.arctan2(rv / (e * np.sqrt(mu * a)), (1.0 - r / a) / e)
    return a, e, P, Q, E0


def kepler_period(z0, mu=MU):
    '''
    Orbital period 2 pi sqrt(a^3 / mu).

    Parameters
    ----------
    z0 : np.ndarray
        Initial state, shape (4,).
    mu : float, optional
        Gravitational parameter.

    Returns
    -------
    float
        Orbital period.
    '''
    a = _elements(z0, mu)[0]
    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


def kepler_exact(z0, t, mu=MU, terms=40):
    '''
    Exact state at time t on the elliptic orbit through z0.

    Kepler's equation is solved by the Lagrange-Bessel series

        E = M + sum_k (2 / k) J_k(k e) sin(k M),

    which converges for eccentricities below the Laplace limit, about
    0.6627. Whole turns are dropped from the mean anomaly first, since
    only cos E and sin E are needed afterwards.

    Parameters
    ----------
    z0 : np.ndarray
        Initial state, shape (4,).
    t : float
        Time.
    mu : float, optional
        Gravitational parameter.
    terms : int, optional
        Number of terms of the series.

    Returns
    -------
    np.ndarray
        Exact state, shape (4,).
    '''
    a, e, P, Q, E0 = _elements(z0, mu)
    M = E0 - e * np.sin(E0) + np.sqrt(mu / a ** 3) * t
    M = (M + np.pi) % (2.0 * np.pi) - np.pi
    k = np.arange(1, terms + 1)
    E = M + np.sum(2.0 / k * jv(k, k * e) * np.sin(k * M))
    b = np.sqrt(1.0 - e * e)
    q = a * (np.cos(E) - e) * P + a * b * np.sin(E) * Q
    speed = np.sqrt(mu * a) / (a * (1.0 - e * np.cos(E)))
    p = speed * (-np.sin(E) * P + b * np.cos(E) * Q)
    return np.concatenate([q, p])


def final_state(method, system, z0, T, N):
    '''
    Integrate one method over [0, T] with N steps.

    Parameters
    ----------
    method : OneStepMethod
        Method to run.
    system : HamiltonianSystem
        System to integrate.
    z0 : np.ndarray
        Initial state, shape (2n,).
    T : float
        Final time.
    N : int
        Number of steps.

    Returns
    -------
    np.ndarray
        Final state, shape (2n,).
    '''
    z, h = np.array(z0, dtype=float), T / N
    for _ in range(N):
        z = method.step(system, z, h)
    return z


def energy_history(method, system, z0, h, steps, sample, relative=True):
    '''
    Energy error of one method along a long run, sampled.

    Parameters
    ----------
    method : OneStepMethod
        Method to run.
    system : HamiltonianSystem
        System to integrate.
    z0 : np.ndarray
        Initial state, shape (2n,).
    h : float
        Step size.
    steps : int
        Number of steps.
    sample : int
        Keep every `sample`-th step.
    relative : bool, optional
        Divide by |H(z0)|.

    Returns
    -------
    t : np.ndarray
        Sample times.
    err : np.ndarray
        Energy error at those times.
    '''
    z = np.array(z0, dtype=float)
    t, err = [], []
    for k in range(1, steps + 1):
        z = method.step(system, z, h)
        if k % sample == 0:
            t.append(k * h)
            err.append(system.energy_error(z0, z, relative=relative))
    return np.array(t), np.array(err)


def main(filename=DATA_FILE):
    '''
    Run every method and write the data file.

    Parameters
    ----------
    filename : str, optional
        Path of the JSON data file; by default in the data folder beside
        this script.

    Returns
    -------
    dict
        The record written to the data file.
    '''
    system = kepler()
    z0 = Z_START.copy()
    T = PERIODS * kepler_period(z0)

    print('\n' + EXPERIMENT)
    print('-' * 72)
    print(f'{"Method":<26s} {"Class":<20s} {"Order":>5s} {"Time":>7s}')
    print('-' * 72)
    methods = {}
    for name, method in METHODS.items():
        t0 = time.perf_counter()
        states = [final_state(method, system, z0, T, N) for N in N_LIST]
        t, err = energy_history(method, system, z0, ENERGY_H,
                                ENERGY_STEPS, ENERGY_SAMPLE,
                                relative=RELATIVE_ENERGY)
        methods[name] = {
            'class': type(method).__name__,
            'order': method.order,
            'final_state': [s.tolist() for s in states],
            'energy_error': err.tolist(),
        }
        print(f'{name:<26s} {type(method).__name__:<20s} {method.order:>5d} '
              f'{time.perf_counter() - t0:6.1f}s', flush=True)

    record = {
        'schema': 1,
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'experiment': EXPERIMENT,
        'parameters': {
            'mu': MU, 'T': T, 'periods': PERIODS, 'N_list': N_LIST,
            'energy_h': ENERGY_H, 'energy_steps': ENERGY_STEPS,
        },
        'initial_state': z0.tolist(),
        'relative_energy': RELATIVE_ENERGY,
        'dt': [T / N for N in N_LIST],
        'reference': {'kind': 'exact',
                      'state': kepler_exact(z0, T).tolist()},
        'energy_time': t.tolist(),
        'method_order': list(METHODS),
        'methods': methods,
    }
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filename, 'w') as fh:
        json.dump(record, fh, indent=1)
    print(f'\nWrote {filename}')
    return record


if __name__ == '__main__':
    main()
