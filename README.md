# EVOLUTE

**E**nergy- and **VOLU**me-preserving **T**ime integration **E**ngine.

`evolute` provides one-step integrators for separable Hamiltonian
systems. Its central method, `EnergyVolumeSplit`, conserves energy and
preserves phase-space volume exactly, for systems with two or more
degrees of freedom. It is not symplectic, which is what allows both
properties at once. Symplectic and discrete gradient methods are
included as comparison baselines, alongside scripts for three test
problems.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Systems](#systems)
- [Methods](#methods)
- [How `EnergyVolumeSplit` works](#how-energyvolumesplit-works)
- [Limitations](#limitations)
- [Experiments](#experiments)
- [Package layout](#package-layout)
- [References](#references)

## Installation

`evolute` needs Python 3.9 or later, NumPy and SciPy. From the
repository root:

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from evolute import EnergyVolumeSplit, HamiltonianSystem


def V(q):
    x, y = q
    return 0.5 * (x * x + y * y) + x * x * y - y ** 3 / 3.0


def grad_V(q):
    x, y = q
    return np.array([x + 2.0 * x * y, y + x * x - y * y])


system = HamiltonianSystem(2, V, grad_V, name="Henon-Heiles")
method = EnergyVolumeSplit()          # symmetric, second order

z0 = np.array([0.1, -0.2, 0.3, 0.15])  # (q1, q2, p1, p2)
z = z0.copy()
for _ in range(1000):
    z = method.step(system, z, 0.05)

print(system.energy_error(z0, z))
```

The printed relative energy error is at round-off, about `1e-14`.

## Systems

A `HamiltonianSystem` has the form

```text
H(q, p) = (p1^2 + p2^2) / 2 + T(p3, ..., pn) + V(q),    n >= 2,
```

with state `z = (q, p)` and canonical equations `dz/dt = J grad H(z)`.
The pair `(p1, p2)` always has the isotropic kinetic energy shown.
A system is built from:

- `n`, the number of degrees of freedom;
- `V` and `grad_V`, functions of `q`;
- optionally `T` and `grad_T`, functions of `(p3, ..., pn)`. The
  default is `T = |(p3, ..., pn)|^2 / 2`;
- optionally `invariants(q, p)`, returning a dictionary of further
  first integrals, and a display `name`.

Because the potential depends only on `q` and `T` only on the remaining
momenta, every system is separable by construction.

`HamiltonianSystem` also provides `H`, `grad_H`, `vector_field` and
`energy_error(z0, z, relative=True)`. Use `relative=False` when the
initial energy is zero in exact arithmetic, as for rays on `H = 0`: in
floating point it is then round-off.

System-specific definitions live in user scripts rather than in the
package; see [Experiments](#experiments).

## Methods

Every method has `step(system, z0, h)`, returning the new state, and
the attributes `name` and `order`.

| Class               | `symmetric` | Order | Energy | Volume | Symplectic |
| ------------------- | ----------- | ----- | ------ | ------ | ---------- |
| `EnergyVolumeSplit` | `True`      | 2     | exact  | exact  | no         |
| `EnergyVolumeSplit` | `False`     | 1     | exact  | exact  | no         |
| `Symplectic`        | `True`      | 2     | no     | exact  | yes        |
| `Symplectic`        | `False`     | 1     | no     | exact  | yes        |
| `Gonzalez`          | n/a         | 2     | exact  | no     | no         |
| `ItohAbe`           | n/a         | 1     | exact  | no     | no         |

`symmetric` defaults to `True`. With `symmetric=True`, `Symplectic` is
Stormer-Verlet; with `False`, it is symplectic Euler.

"Exact" means up to round-off, and for the implicit methods up to the
solver tolerance `xtol`. The symplectic methods do not conserve energy,
but for these problems their energy error stays bounded rather than
drifting.

The second-order methods `EnergyVolumeSplit()` and `Symplectic()` are
each the symmetric composition `M*_{h/2} o M_{h/2}` of a first-order
map `M` with its adjoint `M*`. `Gonzalez` is already symmetric.

The base classes `ExplicitMethod` and `ImplicitMethod` are exported for
writing new methods. `ImplicitMethod` subclasses define a residual
`F(z0, z1)` and inherit a SciPy solve.

## How `EnergyVolumeSplit` works

One step is the conjugation `Phi = Psi^{-1} o M o Psi`.

1. `Psi` replaces the pair `(p1, p2)` by the energy `E = H(q, p)` and
   the angle `phi = atan2(p2, p1)`. It is not canonical, but it has
   unit Jacobian.
2. `M` holds `E` fixed and composes symplectic Euler substeps, each
   acting in one plane of the new variables with everything else
   frozen: `(q1, phi)`, `(q2, phi)`, and the remaining pairs
   `(qk, pk)` for `k >= 3`. Each substep preserves area in its plane.
3. `Psi^{-1}` rebuilds `(p1, p2)` on the level set `H = E`.

Energy is conserved because `E` is a variable that no substep changes.
Volume is preserved because `Psi` and every substep have unit
Jacobian. Energy and area are never required to be preserved in the
same plane, which is why the method can have both properties without
being symplectic. The module docstring of `energy_volume_split.py`
gives the substeps in full.

## Limitations

- Systems must be separable and have the form above, with `n >= 2`.
- The change of variables is singular where `p1 = p2 = 0`. A step that
  reaches it raises `ValueError`; reduce the step size.
- The scalar equations in each substep are contractions only while
  `h |grad V| / rho < 1`, with `rho = sqrt(p1^2 + p2^2)`.
- The momentum pair is fixed for the whole run. Switching pairs
  adaptively to avoid the singular set would break volume
  preservation.

## Experiments

The scripts in `experiments/` define their test problems locally and
print the maximum energy error of every method over 4000 steps of size
`h = 0.05`:

- `kepler.py`: the planar Kepler problem, with angular momentum and
  the Laplace-Runge-Lenz vector as invariants;
- `henon_heiles.py`: the Henon-Heiles system;
- `maxwell_fisheye.py`: the Maxwell fish-eye lens in ray optics, in
  three dimensions. Rays lie on `H = 0`, so the script reports the
  absolute energy error.

Run them from the `experiments/` directory, for example:

```bash
cd experiments
python kepler.py
```

## Package layout

```text
src/evolute/
    __init__.py              public API
    system.py                HamiltonianSystem, canonical_J
    integrator.py            OneStepMethod, ExplicitMethod, ImplicitMethod
    energy_volume_split.py   EnergyVolumeSplit
    symplectic.py            Symplectic (baseline)
    discrete_gradient.py     ItohAbe, Gonzalez (baselines)
experiments/
    kepler.py, henon_heiles.py, maxwell_fisheye.py
```

## References

- K. Feng and Z. Shang, Volume-preserving algorithms for source-free
  dynamical systems, *Numer. Math.* 71 (1995).
- Z. Ge and J. E. Marsden, Lie-Poisson Hamilton-Jacobi theory and
  Lie-Poisson integrators, *Phys. Lett. A* 133 (1988).
- O. Gonzalez, Time integration and discrete Hamiltonian systems,
  *J. Nonlinear Sci.* 6 (1996).
- E. Hairer, C. Lubich and G. Wanner, *Geometric Numerical
  Integration*, 2nd ed., Springer (2006).
- T. Itoh and K. Abe, Hamiltonian-conserving discrete canonical
  equations based on variational difference quotients, *J. Comput.
  Phys.* 76 (1988).
- P. F. Tupper, arXiv:math/0607641.
