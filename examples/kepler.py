'''
Kepler problem.

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
DATA_FILE = os.path.join(DATA_DIR, 'kepler.json')

# Names the figure.
EXPERIMENT = 'Kepler problem'

# Gravitational parameter and initial state (q1, q2, p1, p2).
MU = 1.0
Z_START = np.array([1.0, 0.1, 0.2, 0.9])

# Convergence study. The step counts fall faster than T: a shorter
# run reaches the asymptotic regime at a coarser step.
T = 8.0

# Each order has its own step counts, chosen so that the orders
# plotting.py prints stay within 0.05 of nominal; the finest count is
# the yardstick, so N counts are drawn at N - 1 step sizes.
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

# One entry per figure column: label, methods, step counts.
FAMILIES = [
    ('1st order', FIRST_ORDER_METHODS,
     [16384, 32768, 65536, 131072, 262144, 524288]),
    ('2nd order', SECOND_ORDER_METHODS,
     [2048, 4096, 8192, 16384, 32768, 65536]),
    ('4th order', FOURTH_ORDER_METHODS,
     [256, 512, 1024, 2048, 4096, 8192]),
]

# Long run for the energy history, every step of it. A step coarser
# than the convergence study needs lifts the symplectic energy error
# clear of the round-off floor the discrete-gradient methods sit on.
ENERGY_T = 512.0
ENERGY_H = T / 256
ENERGY_STEPS = round(ENERGY_T / ENERGY_H)
ENERGY_TIME = ENERGY_H * np.arange(1, ENERGY_STEPS + 1)


def kepler(mu=MU):
    '''
    Build the planar Kepler problem, V(q) = -mu / |q|.

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

    return HamiltonianSystem(2, V, grad_V, name='Kepler')


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
    a : float
        Semi-major axis.
    e : float
        Eccentricity.
    P : np.ndarray
        Perifocal basis vector toward perihelion, shape (2,).
    Q : np.ndarray
        Perifocal basis vector a quarter turn on, shape (2,).
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
        raise ValueError('orbit is not bound (H >= 0)')
    a = -mu / (2.0 * energy)
    evec = ((v2 - mu / r) * q - rv * p) / mu
    e = np.linalg.norm(evec)
    if e < 1e-12:
        raise ValueError('circular orbit not supported')
    P = evec / e
    L = q[0] * p[1] - q[1] * p[0]
    Q = np.sign(L) * np.array([-P[1], P[0]])
    E0 = np.arctan2(rv / (e * np.sqrt(mu * a)), (1.0 - r / a) / e)
    return a, e, P, Q, E0


def _eccentric_anomaly(M, e, xtol=1e-15, maxiter=50):
    '''
    Solve Kepler's equation M = E - e sin E by Newton's method.

    The left-hand side has derivative 1 - e cos E, which is at least
    1 - e > 0 for an ellipse, so the equation has one root and Newton's
    method converges quadratically: from the first-order guess
    E = M + e sin M it reaches round-off in a handful of iterations.

    Parameters
    ----------
    M : float
        Mean anomaly.
    e : float
        Eccentricity, below 1.
    xtol : float, optional
        Stop once a Newton correction falls to this size. Default 1e-15.
    maxiter : int, optional
        Iteration limit. Default 50.

    Returns
    -------
    float
        Eccentric anomaly E.

    Raises
    ------
    RuntimeError
        If the iteration does not converge within maxiter steps.
    '''
    E = M + e * np.sin(M)
    for _ in range(maxiter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) <= xtol:
            return E
    raise RuntimeError("Kepler's equation failed to converge")


def kepler_exact(z0, t, mu=MU):
    '''
    Exact state at time t on the elliptic orbit through z0.

    Advances the mean anomaly, drops whole turns from it, since only
    cos E and sin E are needed afterwards, and inverts Kepler's equation
    for the eccentric anomaly.

    In the perifocal frame (P, Q) the ellipse is
    q = a (cos E - e) P + b sin E Q, with b its semi-minor axis, and the
    radius is r = a (1 - e cos E). The velocity is dq/dE scaled by
    dE/dt = sqrt(mu / a) / r.

    Parameters
    ----------
    z0 : np.ndarray
        Initial state, shape (4,).
    t : float
        Time.
    mu : float, optional
        Gravitational parameter.

    Returns
    -------
    np.ndarray
        Exact state, shape (4,).
    '''
    a, e, P, Q, E0 = _elements(z0, mu)
    M = E0 - e * np.sin(E0) + np.sqrt(mu / a ** 3) * t
    M = (M + np.pi) % (2.0 * np.pi) - np.pi
    E = _eccentric_anomaly(M, e)
    r = a * (1.0 - e * np.cos(E))
    b = a * np.sqrt(1.0 - e * e)
    q = a * (np.cos(E) - e) * P + b * np.sin(E) * Q
    scale = np.sqrt(mu / a) / r
    p = scale * (-a * np.sin(E) * P + b * np.cos(E) * Q)
    return np.concatenate([q, p])


def _cartesian(z):
    '''
    The state in the coordinates the error is measured in.

    This system is integrated in Cartesian coordinates already, so the
    state passes through unchanged. Every example takes this step, so
    that the machinery below it does not vary between them.

    Parameters
    ----------
    z : np.ndarray
        State, shape (2n,).

    Returns
    -------
    np.ndarray
        The same state.
    '''
    return z


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
        states = [_cartesian(final_state(method, system, z0, T, N))
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
    system = kepler()
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
            'mu': MU, 'T': T,
            'N_list': {label: n_list
                       for label, _, n_list in FAMILIES},
            'energy_h': ENERGY_H, 'energy_steps': ENERGY_STEPS,
        },
        'initial_state': z0.tolist(),
        'reference': {'kind': 'exact',
                      'state': kepler_exact(z0, T).tolist()},
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
