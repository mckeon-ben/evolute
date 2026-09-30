'''
Axisymmetric logarithmic potential.

Integrates every method over a fixed time at a sequence of step counts,
and over a long run at a single step size, then writes the final states
and the energy histories to a JSON data file.

Cylindrical coordinates, so the kinetic term depends on R and the system
is not separable.
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
DATA_FILE = os.path.join(DATA_DIR, 'logarithmic_potential.json')

# Names the figure.
EXPERIMENT = 'Logarithmic potential'

# Core radius and axis ratio of the potential.
CORE_RADIUS = 0.0
FLATTENING = 0.9

# Energy, angular momentum, starting radius and launch angle in radians.
ENERGY = -0.8
ANGULAR_MOMENTUM = 0.2
R_START = 0.2
LAUNCH_ANGLE = np.pi / 4.0

# Convergence study.
T = 16.0
N_LIST = [16384, 32768, 65536, 131072, 262144, 524288]

# Long run for the energy history, at the coarsest step of the
# convergence study, sampled to keep the file small.
ENERGY_H = T / N_LIST[0]
ENERGY_STEPS = 32768
# The same number of samples in every example, whatever the run's
# length, so every history is drawn at the same resolution.
ENERGY_SAMPLES = 2048
ENERGY_SAMPLE = ENERGY_STEPS // ENERGY_SAMPLES
assert ENERGY_STEPS % ENERGY_SAMPLES == 0, \
    'ENERGY_STEPS must be a multiple of ENERGY_SAMPLES'

# Display name -> method, grouped by class and ordered within it. The
# class name is stored in the data file, so the variants of a class stay
# distinguishable; the plot gives each class a color and a marker, and
# each order a line style.
METHODS = {
    'Symplectic Euler': Symplectic(order=1),
    'Störmer-Verlet': Symplectic(order=2),
    'Itoh-Abe': ItohAbe(order=1),
    'Symmetrized Itoh-Abe': ItohAbe(order=2),
    'Energy-volume split': EnergyVolumeSplit(order=1),
    'Symmetric energy-volume split': EnergyVolumeSplit(order=2),
}


def logarithmic(rc=CORE_RADIUS, qf=FLATTENING):
    '''
    Build the logarithmic potential in cylindrical coordinates.

    State (R, z, phi, pR, pz, Lz); the pair (pR, pz) is isotropic, and
    T = Lz^2 / (2 R^2) depends on R, so the system is not separable.
    phi is cyclic.

    Parameters
    ----------
    rc : float, optional
        Core radius.
    qf : float, optional
        Axis ratio of the equipotentials.

    Returns
    -------
    HamiltonianSystem
        Non-separable system with n = 3.
    '''

    def V(q):
        R, z = q[0], q[1]
        return 0.5 * np.log(rc * rc + R * R + z * z / (qf * qf))

    def grad_V(q):
        R, z = q[0], q[1]
        D = rc * rc + R * R + z * z / (qf * qf)
        return np.array([R / D, z / (qf * qf * D), 0.0])

    def T(q, p_rest):
        return 0.5 * p_rest[0] ** 2 / q[0] ** 2

    def grad_T(q, p_rest):
        L, R = p_rest[0], q[0]
        return np.array([-L * L / R ** 3, 0.0, 0.0]), np.array([L / R ** 2])

    return HamiltonianSystem(3, V, grad_V, T, grad_T, name='Logarithmic')


def to_cartesian(z):
    '''
    Convert a cylindrical state to Cartesian coordinates.

    Parameters
    ----------
    z : np.ndarray
        State (R, z, phi, pR, pz, Lz).

    Returns
    -------
    np.ndarray
        State (x, y, z, px, py, pz).
    '''
    R, zz, phi, pR, pz, L = z
    c, s = np.cos(phi), np.sin(phi)
    return np.array([R * c, R * s, zz,
                     pR * c - L / R * s, pR * s + L / R * c, pz])


def launch(rc=CORE_RADIUS, qf=FLATTENING):
    '''
    Initial state in cylindrical coordinates.

    Places the orbit at (R, z, phi) = (R_START, 0, 0) with angular
    momentum ANGULAR_MOMENTUM and scales the meridional momentum onto
    H = ENERGY.

    Parameters
    ----------
    rc : float, optional
        Core radius.
    qf : float, optional
        Axis ratio of the equipotentials.

    Returns
    -------
    np.ndarray
        State (R, z, phi, pR, pz, Lz).

    Raises
    ------
    ValueError
        If the launch point is outside the zero-velocity curve.
    '''
    system = logarithmic(rc, qf)
    L = ANGULAR_MOMENTUM
    q = np.array([R_START, 0.0, 0.0])
    r2 = 2.0 * (ENERGY - system.V(q)) - (L / R_START) ** 2
    if r2 <= 0.0:
        raise ValueError('launch: point outside the zero-velocity curve')
    rho = np.sqrt(r2)
    return np.array([R_START, 0.0, 0.0, rho * np.cos(LAUNCH_ANGLE),
                     rho * np.sin(LAUNCH_ANGLE), L])


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
    t, err = [], []
    for k, z in enumerate(evolve(method, system, z0, h, steps), start=1):
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
    system = logarithmic()
    z0 = launch()

    print('\n' + EXPERIMENT)
    print('-' * 67)
    print(f'{"Method":<30s} {"Class":<20s} {"Order":>5s} {"Time":>7s}')
    print('-' * 67)
    methods = {}
    for name, method in METHODS.items():
        t0 = time.perf_counter()
        states = [to_cartesian(final_state(method, system, z0, T, N))
                  for N in N_LIST]
        t, err = energy_history(method, system, z0, ENERGY_H,
                                ENERGY_STEPS, ENERGY_SAMPLE)
        elapsed = time.perf_counter() - t0
        methods[name] = {
            'class': type(method).__name__,
            'order': method.order,
            'time': round(elapsed, 3),
            'final_state': [s.tolist() for s in states],
            'energy_error': err.tolist(),
        }
        print(f'{name:<30s} {type(method).__name__:<20s} {method.order:>5d} '
              f'{elapsed:6.1f}s', flush=True)

    record = {
        'schema': 1,
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'experiment': EXPERIMENT,
        'parameters': {
            'core_radius': CORE_RADIUS, 'flattening': FLATTENING,
            'energy': ENERGY, 'angular_momentum': ANGULAR_MOMENTUM,
            'R_start': R_START, 'launch_angle': LAUNCH_ANGLE,
            'T': T, 'N_list': N_LIST,
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
