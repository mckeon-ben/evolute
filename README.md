# evolute

**E**nergy- and **VOLU**me-preserving **T**ime integration **E**ngine.

`evolute` provides one-step integrators for Hamiltonian systems. Its
central method, `EnergyVolumeSplit`, conserves energy and preserves
phase-space volume exactly, for systems with two or more degrees of
freedom, and also conserves the momentum conjugate to a cyclic
coordinate outside the isotropic pair. It is not symplectic, which is
what allows energy and volume preservation at once. Symplectic
and discrete gradient methods are included as comparison baselines,
alongside scripts for four test problems.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Systems](#systems)
- [Methods](#methods)
- [How `EnergyVolumeSplit` works](#how-energyvolumesplit-works)
- [Things to know](#things-to-know)
- [Examples](#examples)
- [Package layout](#package-layout)
- [Licence](#licence)
- [References](#references)

## Installation

`evolute` needs Python 3.9 or later, NumPy, SciPy and matplotlib:

```bash
git clone https://github.com/mckeon-ben/evolute.git
cd evolute
pip install .
```

Use `pip install -e .` instead to work on the code in place. The
scripts in `examples/` also need a LaTeX installation; see
[Examples](#examples).

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


system = HamiltonianSystem(2, V, grad_V, name='Henon-Heiles')
method = EnergyVolumeSplit()          # symmetric, second order

z0 = np.array([0.1, -0.2, 0.3, 0.15])  # (q1, q2, p1, p2)
z = z0.copy()
for _ in range(1000):
    z = method.step(system, z, 0.05)

print(system.energy_error(z0, z))
```

The printed relative energy error is at round-off, below `1e-14`.

## Systems

A `HamiltonianSystem` has the form

```text
H(q, p) = (p1^2 + p2^2) / 2 + T(q, p3, ..., pn) + V(q),    n >= 2,
```

with state `z = (q, p)` and canonical equations `dz/dt = J grad H(z)`.
The pair `(p1, p2)` always has the isotropic kinetic energy shown.
A system is built from:

- `n`, the number of degrees of freedom;
- `V` and `grad_V`, functions of *q*;
- optionally `T` and `grad_T`, functions of *q* and
  `p_rest = (p3, ..., pn)`; `grad_T` returns the pair
  `(dT/dq, dT/dp_rest)`. The default is `T = |(p3, ..., pn)|^2 / 2`;
- optionally `separable`, which records whether *T* is independent of
  *q*. It defaults to `True` when *T* is omitted and `False` when it is
  supplied, so pass `True` for a supplied *T* of the momenta alone;
- optionally `invariants(q, p)`, returning a dictionary of further
  first integrals, and a display `name`.

Letting *T* depend on *q* admits curvilinear coordinates. In
cylindrical coordinates `(R, z, phi)`, for example, an axisymmetric
potential gives `T = L^2 / (2 R^2)`, with `L` the momentum conjugate
to `phi`. Such a system is not separable.

`HamiltonianSystem` also provides `H`, `grad_H`, `vector_field` and
`energy_error(z0, z, relative=True)`. Use `relative=False` when the
initial energy is zero in exact arithmetic, as for rays on `H = 0`: in
floating point it is then round-off.

System-specific definitions live in user scripts rather than in the
package; see [Examples](#examples).

## Methods

Every method has `step(system, z0, h)`, which returns the new state as
a new array and leaves `z0` unchanged, and the attributes `name` and
`order`. A run is a loop over `step`, as in the
[quick start](#quick-start).

| Method              | Class               | Order | Energy | Volume |
| ------------------- | ------------------- | ----- | ------ | ------ |
| Energy-volume split | `EnergyVolumeSplit` | 2     | exact  | exact  |
| Energy-volume split | `EnergyVolumeSplit` | 1     | exact  | exact  |
| Störmer–Verlet      | `Symplectic`        | 2     | no     | exact  |
| Symplectic Euler    | `Symplectic`        | 1     | no     | exact  |
| Gonzalez            | `Gonzalez`          | 2     | exact  | no     |
| Itoh–Abe            | `ItohAbe`           | 1     | exact  | no     |

`EnergyVolumeSplit` and `Symplectic` take `symmetric`, which defaults
to `True` and gives the second-order method; `False` gives the
first-order one. Only Störmer–Verlet and symplectic Euler are
symplectic. They are explicit for a separable system; otherwise they
take the implicit forms of Hairer, Lubich and Wanner, solved by
fixed-point iteration, and remain symplectic.

"Exact" means up to round-off, and for the implicit methods up to the
solver tolerance `xtol`. The symplectic methods do not conserve energy,
but for these problems their energy error stays bounded rather than
drifting.

The second-order methods `EnergyVolumeSplit()` and `Symplectic()` are
each the symmetric composition `M*_{h/2} o M_{h/2}` of a first-order
map *M* with its adjoint *M*\*. `Gonzalez` is already symmetric.

The base classes `PartitionedMethod` and `ImplicitMethod` are exported
for writing new methods. `PartitionedMethod` subclasses inherit kick
and drift substeps, explicit or implicit; `ImplicitMethod` subclasses
define a residual `F(z0, z1)` and inherit a SciPy solve.

## How `EnergyVolumeSplit` works

One step is the conjugation `Phi = Psi^{-1} o M o Psi`.

1. *Ψ* replaces the pair `(p1, p2)` by the energy `E = H(q, p)` and
   the angle `phi = atan2(p2, p1)`. It is not canonical, but it has
   unit Jacobian.
2. *M* holds *E* fixed and composes symplectic Euler substeps, each
   acting in one plane of the new variables with everything else
   frozen: `(q1, phi)`, `(q2, phi)`, and the remaining pairs
   `(qk, pk)` for `k >= 3`. Each substep preserves area in its plane.
   The last substep is explicit for a separable system and is solved
   by fixed-point iteration otherwise.
3. `Psi^{-1}` rebuilds `(p1, p2)` on the level set `H = E`.

Energy is conserved because *E* is a variable that no substep changes.
Volume is preserved because *Ψ* and every substep have unit
Jacobian. Energy and area are never required to be preserved in the
same plane, which is why the method can have both properties without
being symplectic. The module docstring of `energy_volume_split.py`
gives the substeps in full.

If *H* does not depend on some `qk` with `k >= 3`, the last substep
leaves `pk` unchanged and no other substep touches it, so that momentum
is conserved exactly too. A spatial symmetry therefore carries over
only when it is expressed through a cyclic coordinate among
`q3, ..., qn`: for an axisymmetric potential, the azimuth of
cylindrical coordinates. In Cartesian coordinates the splitting breaks
the symmetry, and the angular momentum is no longer conserved.

## Things to know

- Systems must have the form above, with `n >= 2`.
- The change of variables is singular where `p1 = p2 = 0`. A step that
  reaches it raises `ValueError`; reduce the step size.
- The scalar equations in each substep are contractions only while
  `h |dF/dq| / rho < 1`, with `F = T + V` and
  `rho = sqrt(p1^2 + p2^2)`.
- The implicit solves need a small enough step. A solve that does not
  converge raises `RuntimeError`: the scalar equations of
  `EnergyVolumeSplit` always, and the fixed-point iterations of
  `EnergyVolumeSplit` and `Symplectic` for a non-separable system.
- The momentum pair is fixed for the whole run. Switching pairs
  adaptively to avoid the singular set would break volume
  preservation.

## Examples

The scripts in `examples/` integrate every method twice over and write
the results to JSON: a convergence study to a fixed final time at a
sequence of step counts, and a long run at a single step size recording
the energy error (and, in `logarithmic_potential.py`, the angular
momentum error):

| Script                     | System                                     |
| -------------------------- | ------------------------------------------ |
| `kepler.py`                | Planar Kepler problem                      |
| `henon_heiles.py`          | Hénon–Heiles system                        |
| `maxwell_fisheye.py`       | Maxwell fish-eye lens in ray optics, n = 3 |
| `logarithmic_potential.py` | Axisymmetric logarithmic potential, n = 3  |

`plotting.py` turns the data files into error estimates, observed
orders and figures, each pairing the energy error against time with the
position error against step size, with a momentum-error panel added
when the data file has one. The scripts can be run from any directory:
data files always go to `examples/data/` and figures to
`examples/plots/`.

```bash
python examples/kepler.py
python examples/plotting.py kepler          # one data file
python examples/plotting.py                 # every data file
```

`plotting.py` typesets through LaTeX by default, needing an
installation with the `helvet` and `sansmath` packages. Set
`USETEX = False` at the top of the script to use matplotlib's own
renderer instead.

## Package layout

```text
pyproject.toml
README.md
LICENSE
evolute/
    __init__.py              public API
    system.py                HamiltonianSystem, canonical_J
    integrator.py            OneStepMethod, PartitionedMethod,
                             ImplicitMethod
    energy_volume_split.py   EnergyVolumeSplit
    symplectic.py            Symplectic (baseline)
    discrete_gradient.py     ItohAbe, Gonzalez (baselines)
examples/
    <test problem>.py        four simulation scripts
    plotting.py              error estimates and figures
```

## Licence

MIT; see [LICENSE](LICENSE).

## References

- Feng, K. and Shang, Z., 1995. Volume-preserving algorithms for
  source-free dynamical systems. *Numerische Mathematik, 71*(4),
  pp.451-463.
- Ge, Z. and Marsden, J.E., 1988. Lie-Poisson Hamilton-Jacobi
  theory and Lie-Poisson integrators. *Physics Letters A, 133*(3),
  pp.134-139.
- Gonzalez, O., 1996. Time integration and discrete Hamiltonian
  systems. *Journal of Nonlinear Science, 6*(5), pp.449-467.
- Hairer, E., Lubich, C. and Wanner, G., 2006. *Geometric numerical
  integration: Structure-preserving algorithms for ordinary
  differential equations*. 2nd ed., Springer.
- Itoh, T. and Abe, K., 1988. Hamiltonian-conserving discrete
  canonical equations based on variational difference quotients.
  *Journal of Computational Physics, 76*(1), pp.85-102.
- Tupper, P.F., 2006. A Non-Existence Result for Hamiltonian
  Integrators. *arXiv preprint math/0607641*.
