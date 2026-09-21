'''
Maxwell fish-eye lens (ray optics).

Integrates every method over a fixed parameter interval at a sequence of
step counts, and over a long run at a single step size, then writes the
final states and the energy history to a JSON data file.

The reference solution is exact. Liu (2022) gives the ray path in polar
form; the parameter of this Hamiltonian follows from it by a quadrature
(see fisheye_exact).

The run covers one pass through the lens: the ray enters at the rim and
the run ends at the antipodal point, where every ray of a fish-eye
returns to the surface.

Rays lie on H = 0, so the energy error is reported as an absolute one.

References
----------
Liu, W., 2022. Ray tracing in concentric gradient-index media: optical
Binet equation. Journal of the Optical Society of America A, 39(6),
pp.1025-1033.
'''

import json
import os
import time

import numpy as np
from scipy.optimize import brentq

from evolute import (
    EnergyVolumeSplit, Gonzalez, HamiltonianSystem, ItohAbe, Symplectic,
)


# Written into the data folder beside this script, wherever it is
# run from.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data')
DATA_FILE = os.path.join(DATA_DIR, 'maxwell_fisheye.json')

# Names the figure.
EXPERIMENT = 'Maxwell fish-eye lens'

# Lens parameters. With n0 = 2 the index is 2 R^2 / (R^2 + r^2), which
# is one at the rim, so a ray entering from vacuum does not refract.
N0 = 2.0
RADIUS = 1.0

# The ray is launched from the rim, |q| = R, at 75 degrees from the
# inward radius, with a small out-of-plane tilt so that the third
# momentum is not idle. The momentum is scaled onto H = 0 by ray_state.
Q_START = (1.0, 0.0, 0.0)
INJECTION_ANGLE = 75.0
TILT = 0.12

# Both studies cover one pass through the lens: from the incident point
# to the conjugate point on the far side, which every ray of a fish-eye
# reaches together. fisheye_pass computes that parameter interval, so
# the run ends where the ray leaves the lens rather than at a round
# number.
N_LIST = [64, 128, 256, 512, 1024]

# Energy history over the same pass, sampled to keep the file small.
ENERGY_H = 0.01
ENERGY_SAMPLE = 2
RELATIVE_ENERGY = False

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


def maxwell_fisheye(n0=N0, R=RADIUS):
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

    system = HamiltonianSystem(3, V, grad_V,
                               name='Maxwell fish-eye')
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


def injection_direction(angle=INJECTION_ANGLE, tilt=TILT):
    '''
    Ray direction at the rim point Q_START.

    The direction is `angle` degrees from the inward radius, rotated
    towards a tangent tilted out of the plane z = 0, so the angle of
    incidence is exactly `angle` whatever the tilt.

    Parameters
    ----------
    angle : float, optional
        Angle of incidence in degrees, measured from the inward radius.
    tilt : float, optional
        Out-of-plane component of the tangent direction.

    Returns
    -------
    np.ndarray
        Unit direction vector, shape (3,).
    '''
    q = np.asarray(Q_START, dtype=float)
    inward = -q / np.linalg.norm(q)
    tangent = np.array([0.0, 1.0, tilt])
    tangent /= np.linalg.norm(tangent)
    phi = np.radians(angle)
    return np.cos(phi) * inward + np.sin(phi) * tangent


def _path_constants(z0, R):
    '''
    Constants of the polar ray path R / r - r / R = C sin(theta + C0).

    The path is Eq. (14) of Liu (2022). C and C0 follow from the initial
    radius and the initial radial momentum.

    Parameters
    ----------
    z0 : np.ndarray
        Initial state, shape (6,).
    R : float
        Lens length scale.

    Returns
    -------
    L : float
        Magnitude of the angular momentum q x p.
    e1, e2 : np.ndarray
        Orthonormal basis of the orbital plane, with e1 along q0.
    C, C0 : float
        Amplitude and phase of the path.
    '''
    q, p = z0[:3], z0[3:]
    Lvec = np.cross(q, p)
    L = np.linalg.norm(Lvec)
    e1 = q / np.linalg.norm(q)
    e2 = np.cross(Lvec / L, e1)
    r0 = np.linalg.norm(q)
    dr = (p @ e1) * r0 ** 2 / L                 # dr/dtheta at theta = 0
    w = R / r0 - r0 / R
    dw = -(R / r0 ** 2 + 1.0 / R) * dr
    return L, e1, e2, np.hypot(w, dw), np.arctan2(w, dw)


def _radius(phi, C, R):
    '''
    Positive root r of r^2 + C R sin(phi) r - R^2 = 0.

    Parameters
    ----------
    phi : float
        Polar angle plus the path phase.
    C, R : float
        Path amplitude and lens length scale.

    Returns
    -------
    float
        Radius on the ray path.
    '''
    b = C * R * np.sin(phi)
    return 0.5 * (-b + np.sqrt(b * b + 4.0 * R ** 2))


def _sigma_integral(phi, C, R):
    '''
    Antiderivative of r(phi)^2 with respect to phi.

    Since dq/dsigma = p with |p| = n on H = 0, the path element gives
    dsigma = r^2 dtheta / L, so the parameter follows from this
    integral. Both pieces are elementary.

    Parameters
    ----------
    phi : float
        Polar angle plus the path phase.
    C, R : float
        Path amplitude and lens length scale.

    Returns
    -------
    float
        Value of the antiderivative at phi.
    '''
    A = C * R
    B2 = A * A + 4.0 * R * R
    c = np.cos(phi)
    quad = A * A * (0.5 * phi - 0.25 * np.sin(2.0 * phi))
    const = 2.0 * R * R * phi
    surd = -A * (0.5 * c * np.sqrt(B2 - A * A * c * c)
                 + B2 / (2.0 * A) * np.arcsin(A * c / np.sqrt(B2)))
    return 0.5 * (quad + const - surd)


def fisheye_pass(z0, R=RADIUS):
    '''
    Parameter interval of one pass through the lens.

    A ray entering at the rim leaves at the conjugate point, half a turn
    round in polar angle, so the interval is sigma(theta = pi).

    Parameters
    ----------
    z0 : np.ndarray
        Initial state on H = 0, shape (6,).
    R : float, optional
        Lens length scale.

    Returns
    -------
    float
        Parameter value at the exit point.
    '''
    L, _, _, C, C0 = _path_constants(z0, R)
    return (_sigma_integral(np.pi + C0, C, R)
            - _sigma_integral(C0, C, R)) / L


def fisheye_exact(z0, sigma, R=RADIUS):
    '''
    Exact ray state at parameter sigma.

    The polar angle is recovered by inverting sigma(theta), which is
    strictly increasing, with a bracketing root solve; the state then
    follows from the path and from dtheta/dsigma = L / r^2.

    Parameters
    ----------
    z0 : np.ndarray
        Initial state on H = 0, shape (6,).
    sigma : float
        Parameter value, non-negative.
    R : float, optional
        Lens length scale.

    Returns
    -------
    np.ndarray
        Exact state, shape (6,).
    '''
    L, e1, e2, C, C0 = _path_constants(z0, R)
    base = _sigma_integral(C0, C, R)

    def residual(theta):
        return (_sigma_integral(theta + C0, C, R) - base) / L - sigma

    lo, hi = 0.0, 1.0
    while residual(hi) < 0.0:
        hi *= 2.0
    while residual(lo) > 0.0:
        lo -= 1.0
    theta = brentq(residual, lo, hi, xtol=1e-14, rtol=8.9e-16)

    phi = theta + C0
    r = _radius(phi, C, R)
    b, db = C * R * np.sin(phi), C * R * np.cos(phi)
    dr = -db * r / (2.0 * r + b)                # dr/dtheta
    dtheta = L / r ** 2                         # dtheta/dsigma
    radial = np.cos(theta) * e1 + np.sin(theta) * e2
    angular = -np.sin(theta) * e1 + np.cos(theta) * e2
    return np.concatenate([r * radial,
                           dr * dtheta * radial + r * dtheta * angular])


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
    system = maxwell_fisheye()
    z0 = ray_state(system, Q_START, injection_direction())
    T = fisheye_pass(z0)
    energy_steps = int(round(T / ENERGY_H))

    print('\n' + EXPERIMENT)
    print('-' * 72)
    print(f'{"Method":<26s} {"Class":<20s} {"Order":>5s} {"Time":>7s}')
    print('-' * 72)
    methods = {}
    for name, method in METHODS.items():
        t0 = time.perf_counter()
        states = [final_state(method, system, z0, T, N) for N in N_LIST]
        t, err = energy_history(method, system, z0, ENERGY_H,
                                energy_steps, ENERGY_SAMPLE,
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
            'n0': N0, 'R': RADIUS,
            'q_start': list(Q_START), 'angle': INJECTION_ANGLE,
            'tilt': TILT, 'T': T, 'N_list': N_LIST,
            'energy_h': ENERGY_H, 'energy_steps': energy_steps,
        },
        'initial_state': z0.tolist(),
        'relative_energy': RELATIVE_ENERGY,
        'dt': [T / N for N in N_LIST],
        'reference': {'kind': 'exact',
                      'state': fisheye_exact(z0, T).tolist()},
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
