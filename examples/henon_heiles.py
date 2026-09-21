'''
Henon-Heiles system.

Integrates every method over a fixed time at a sequence of step counts,
and over a long run at a single step size, then writes the final states
and the energy history to a JSON data file.
'''

import json
import os
import time

import numpy as np

from evolute import (
    EnergyVolumeSplit, Gonzalez, HamiltonianSystem, ItohAbe, Symplectic,
)


# Written into the data folder beside this script, wherever it is
# run from.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data')
DATA_FILE = os.path.join(DATA_DIR, 'henon_heiles.json')

# Names the figure.
EXPERIMENT = 'Henon-Heiles system'

# Cubic coupling and initial state (q1, q2, p1, p2).
LAMBDA = 1.0
Z_START = np.array([0.1, -0.2, 0.3, 0.15])

# Convergence study.
T = 16.0
N_LIST = [512, 1024, 2048, 4096, 8192, 16384]

# Long run for the energy history, at the coarsest step of the
# convergence study, sampled to keep the file small.
ENERGY_H = T / N_LIST[0]
ENERGY_STEPS = 16384
ENERGY_SAMPLE = 64

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


def henon_heiles(lam=LAMBDA):
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

    return HamiltonianSystem(2, V, grad_V, name='Henon-Heiles')


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


def energy_history(method, system, z0, h, steps, sample):
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
            err.append(system.energy_error(z0, z))
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
    system = henon_heiles()
    z0 = Z_START.copy()

    print('\n' + EXPERIMENT)
    print('-' * 72)
    print(f'{"Method":<26s} {"Class":<20s} {"Order":>5s} {"Time":>7s}')
    print('-' * 72)
    methods = {}
    for name, method in METHODS.items():
        t0 = time.perf_counter()
        states = [final_state(method, system, z0, T, N) for N in N_LIST]
        t, err = energy_history(method, system, z0, ENERGY_H,
                                ENERGY_STEPS, ENERGY_SAMPLE)
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
            'lambda': LAMBDA, 'T': T, 'N_list': N_LIST,
            'energy_h': ENERGY_H, 'energy_steps': ENERGY_STEPS,
        },
        'initial_state': z0.tolist(),
        'dt': [T / N for N in N_LIST],
        'reference': {'kind': 'successive', 'state': None},
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
