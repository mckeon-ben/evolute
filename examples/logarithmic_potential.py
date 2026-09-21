'''
Axisymmetric logarithmic potential (momentum conservation).

Integrates every method over a fixed time at a sequence of step counts,
and over a long run at a single step size, then writes the final states
and the energy and angular momentum histories to a JSON data file.

The flattened logarithmic potential

    V = ln(Rc^2 + R^2 + z^2 / qf^2) / 2

is axisymmetric. In cylindrical coordinates (R, z, phi) the azimuth phi
is cyclic, so its conjugate momentum, the angular momentum Lz, is
conserved, and

    H = (pR^2 + pz^2) / 2 + Lz^2 / (2 R^2) + V(R, z)

is not separable, so the symplectic methods become implicit. Because
the derivative of H with respect to phi vanishes identically, the
symplectic methods, Itoh-Abe and EnergyVolumeSplit conserve Lz exactly;
Gonzalez does not, since its correction term is parallel to the whole
step. Only EnergyVolumeSplit conserves energy, volume and Lz together:
Itoh-Abe conserves energy but not volume, and the symplectic methods
preserve volume but not energy. EnergyVolumeSplit needs the symmetry
expressed through a cyclic coordinate: in Cartesian coordinates, where
the rotation mixes x and y, it does not conserve Lz.

Final states are converted to Cartesian coordinates before they are
stored, so the convergence errors are distances in space, about
sqrt(dR^2 + dz^2 + (R dphi)^2). A norm on (R, z) alone would miss the
error in phi. From this launch it would hide the first-order error of
symplectic Euler: the launch is at the minimum of the effective
potential, where grad_q H vanishes, so that error goes entirely into
phi.

Itoh-Abe's energy error is about 3e-13 here rather than 1e-14: at the
inner radial turning points R changes by only a few 1e-6 per step, and
its difference quotient in R loses digits to round-off there.

The potential, energy and angular momentum are those of the orbits in
Binney and Tremaine's Figure 3.4: their equation (3.70), which is the
potential above with Rc = 0, with qf = 0.9, E = -0.8 and Lz = 0.2, in
units with v0 = 1. The launch is our own. Their left-hand orbit nearly
touches the zero-velocity curve, where the meridional speed vanishes
and EnergyVolumeSplit is singular, so this orbit lies on a neighbouring
invariant curve of their Figure 3.5 instead.

The system is non-integrable, so there is no closed-form reference; the
convergence errors are estimated from successive step sizes.

References
----------
Binney, J. and Tremaine, S., 2008. Galactic dynamics. 2nd ed.,
Princeton University Press, Section 3.2.
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
DATA_FILE = os.path.join(DATA_DIR, 'logarithmic_potential.json')

# Names the figure.
EXPERIMENT = 'Logarithmic potential'

# Core radius Rc and axis ratio qf of the potential. Without a core, V
# is singular at the origin, which an orbit with Lz != 0 never reaches.
CORE_RADIUS = 0.0
FLATTENING = 0.9

# The orbit has energy E and angular momentum Lz, and is launched from
# the midplane at radius R_START, its meridional momentum (pR, pz) at
# LAUNCH_ANGLE degrees from the R axis; R_START is the radius of the
# circular orbit with this Lz. The orbit is a tube around the z axis
# whose meridional speed sqrt(pR^2 + pz^2), the rho of
# EnergyVolumeSplit, stays above about 0.03 over the long run. The
# left-hand orbit of Binney and Tremaine's Figure 3.4, through
# (R, pR) = (0.16, 0) on their surface of section, drops to about 0.002
# at its corners, and the second-order split fails there whatever the
# step size.
ENERGY = -0.8
ANGULAR_MOMENTUM = 0.2
R_START = 0.2
LAUNCH_ANGLE = 45.0

# Convergence study, about two radial oscillations. The steps are
# small enough for symplectic Euler to reach its asymptotic first-order
# regime.
T = 2.5
N_LIST = [512, 1024, 2048, 4096, 8192]

# Long run for the energy and angular momentum histories, sampled to
# keep the file small. The step is about half the largest at which the
# first-order energy-volume split completes the run, 0.0035: its scalar
# equations stop being contractions once h |dF/dq| / rho approaches
# one.
ENERGY_H = 0.002
ENERGY_STEPS = 25000
ENERGY_SAMPLE = 50
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


def logarithmic(rc=CORE_RADIUS, qf=FLATTENING):
    '''
    Build the logarithmic potential in cylindrical coordinates.

    State (R, z, phi, pR, pz, Lz); the pair (pR, pz) is isotropic, and
    T = Lz^2 / (2 R^2) depends on R, so the system is not separable.
    phi is cyclic. Supplies the angular momentum Lz as an invariant.

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

    def invariants(q, p):
        return {'angular_momentum': p[2]}

    return HamiltonianSystem(3, V, grad_V, T, grad_T,
                             invariants=invariants, name='Logarithmic')


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
    a = np.radians(LAUNCH_ANGLE)
    rho = np.sqrt(r2)
    return np.array([R_START, 0.0, 0.0, rho * np.cos(a), rho * np.sin(a),
                     L])


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


def histories(method, system, z0, h, steps, sample, relative=True):
    '''
    Energy and angular momentum errors along a long run, sampled.

    Parameters
    ----------
    method : OneStepMethod
        Method to run.
    system : HamiltonianSystem
        System to integrate; must supply 'angular_momentum' as an
        invariant.
    z0 : np.ndarray
        Initial state, shape (2n,).
    h : float
        Step size.
    steps : int
        Number of steps.
    sample : int
        Keep every `sample`-th step.
    relative : bool, optional
        Divide the energy error by |H(z0)|.

    Returns
    -------
    t : np.ndarray
        Sample times.
    energy : np.ndarray
        Energy error at those times.
    momentum : np.ndarray
        Absolute angular momentum error at those times.
    '''
    z = np.array(z0, dtype=float)
    L0 = system.invariants(z)['angular_momentum']
    t, energy, momentum = [], [], []
    for k in range(1, steps + 1):
        z = method.step(system, z, h)
        if k % sample == 0:
            t.append(k * h)
            energy.append(system.energy_error(z0, z, relative=relative))
            momentum.append(
                abs(system.invariants(z)['angular_momentum'] - L0))
    return np.array(t), np.array(energy), np.array(momentum)


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
    print('-' * 72)
    print(f'{"Method":<26s} {"Class":<20s} {"Order":>5s} {"Time":>7s}')
    print('-' * 72)
    methods = {}
    for name, method in METHODS.items():
        t0 = time.perf_counter()
        states = [to_cartesian(final_state(method, system, z0, T, N))
                  for N in N_LIST]
        t, energy, momentum = histories(method, system, z0, ENERGY_H,
                                        ENERGY_STEPS, ENERGY_SAMPLE,
                                        relative=RELATIVE_ENERGY)
        methods[name] = {
            'class': type(method).__name__,
            'order': method.order,
            'final_state': [s.tolist() for s in states],
            'energy_error': energy.tolist(),
            'momentum_error': momentum.tolist(),
        }
        print(f'{name:<26s} {type(method).__name__:<20s} {method.order:>5d} '
              f'{time.perf_counter() - t0:6.1f}s', flush=True)

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
        'relative_energy': RELATIVE_ENERGY,
        'momentum_symbol': 'L_z',
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
