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
    EnergyVolumeSplit, evolve, HamiltonianSystem, integrate, ItohAbe,
    Symplectic,
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

# Convergence study. The step counts below fall faster than T does:
# a shorter run reaches the asymptotic regime at a coarser step,
# because the terms beyond the leading one grow with the time
# integrated.
T = 8.0

# Each order has its own step counts, chosen so that every method is
# in its asymptotic regime: the observed orders printed by plotting.py
# sit within 0.05 of the nominal order across each list. Each order
# enters that regime at a step of its own -- the first order latest,
# and Itoh-Abe latest of the three classes -- and leaves it again at
# the round-off floor, which a fourth-order method reaches four times
# faster than a second-order one, so no single list serves them all.
# The finest count of a list is the yardstick of the ladder and
# carries no error of its own, so a list of N counts is drawn at
# N - 1 step sizes. Within a family, methods are grouped by class;
# the plot gives each class a color and a marker, each order a line
# style, and the class name is stored in the data file so that the
# variants of a class stay distinguishable.
FIRST_ORDER_METHODS = {
    'Symplectic Euler': Symplectic(order=1),
    'Itoh-Abe': ItohAbe(order=1),
    'Energy-volume split': EnergyVolumeSplit(order=1),
}
SECOND_ORDER_METHODS = {
    'Störmer-Verlet': Symplectic(order=2),
    'Symmetrized Itoh-Abe': ItohAbe(order=2),
    'Symmetric energy-volume split': EnergyVolumeSplit(order=2),
}
FOURTH_ORDER_METHODS = {
    'Forest-Ruth': Symplectic(order=4),
    'Itoh-Abe triple jump': ItohAbe(order=4),
    'Energy-volume split triple jump': EnergyVolumeSplit(order=4),
}

FAMILIES = [
    ('1st order', FIRST_ORDER_METHODS, [512, 1024, 2048, 4096, 8192]),
    ('2nd order', SECOND_ORDER_METHODS, [64, 128, 256, 512, 1024]),
    ('4th order', FOURTH_ORDER_METHODS, [256, 512, 1024, 2048, 4096]),
]

# Long run for the energy history, every step of it. The panel is
# read for how large the energy error grows, which is the upper edge
# of a fast oscillation, and keeping every N-th step samples through
# that oscillation rather than along its edge: it serrates the edge
# and understates the largest error.
# It is set by the time it covers rather than by a count of steps,
# because the step below is chosen for the methods and not for the
# run, so a step count would leave the span implicit and move it
# whenever the step changed. That step is coarser than the
# convergence study needs: a coarse step covers more time per step,
# and it lifts the energy error of the symplectic methods further
# clear of the round-off floor the discrete-gradient methods sit on,
# while their own exact conservation is indifferent to the step. How
# coarse it can be is set by the methods, since an implicit solve
# eventually fails to converge and the energy-volume split has its
# singular set to stay clear of.
ENERGY_T = 512.0
ENERGY_H = T / 256
ENERGY_STEPS = round(ENERGY_T / ENERGY_H)
ENERGY_TIME = ENERGY_H * np.arange(1, ENERGY_STEPS + 1)


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
    return integrate(method, system, z0, T / N, N)


def energy_history(method, system, z0, h, steps):
    '''
    Energy error of one method after every step of a long run.

    The step times are ENERGY_TIME, which follows from the constants
    above.

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

    Returns
    -------
    np.ndarray
        Energy error after each step, shape (steps,).
    '''
    return np.array([system.energy_error(z0, z)
                     for z in evolve(method, system, z0, h, steps)])


def run_family(system, z0, label, methods_in_family, n_list):
    '''
    Run one family of methods over its own step counts.

    Parameters
    ----------
    system : HamiltonianSystem
        System to integrate.
    z0 : np.ndarray
        Initial state, shape (2n,).
    label : str
        Display label of the family.
    methods_in_family : dict
        Display name -> method.
    n_list : list of int
        Step counts of the convergence study for this family.

    Returns
    -------
    dict
        Keys 'label', 'dt', 'method_order' and 'methods', the last
        mapping each display name to its class, order, the seconds it
        took, its final states and its energy error.
    '''
    methods = {}
    for name, method in methods_in_family.items():
        t0 = time.perf_counter()
        states = [final_state(method, system, z0, T, N)
                  for N in n_list]
        err = energy_history(method, system, z0, ENERGY_H, ENERGY_STEPS)
        elapsed = time.perf_counter() - t0
        methods[name] = {
            'class': type(method).__name__,
            'order': method.order,
            'time': round(elapsed, 3),
            'final_state': [s.tolist() for s in states],
            'energy_error': err.tolist(),
        }
        print(f'  {label:>9s}  {name:<31s} '
              f'{type(method).__name__:<17s} {elapsed:6.1f}s', flush=True)
    return {'label': label, 'dt': [T / N for N in n_list],
            'method_order': list(methods_in_family), 'methods': methods}


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
    print(f'{"Family":>11s}  {"Method":<31s} {"Class":<17s} {"Time":>7s}')
    print('-' * 72)
    families = [run_family(system, z0, label, methods, n_list)
                for label, methods, n_list in FAMILIES]

    record = {
        'schema': 2,
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'experiment': EXPERIMENT,
        'parameters': {
            'lambda': LAMBDA, 'T': T,
            'N_list': {label: n_list
                       for label, _, n_list in FAMILIES},
            'energy_h': ENERGY_H, 'energy_steps': ENERGY_STEPS,
        },
        'initial_state': z0.tolist(),
        'reference': {'kind': 'successive', 'state': None},
        'energy_time': ENERGY_TIME.tolist(),
        'families': families,
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
