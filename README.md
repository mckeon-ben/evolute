# evolute

**E**nergy- and **VOLU**me-preserving **T**ime integration **E**ngine.

`evolute` provides one-step integrators for Hamiltonian systems. Its
central method, `EnergyVolumeSplit`, conserves energy and preserves
phase-space volume exactly, for systems with two or more degrees of
freedom. It is not symplectic, which is what allows energy and volume
preservation at once. Symplectic and discrete gradient methods are
included as comparison baselines, alongside scripts for three test
problems.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Systems](#systems)
- [Methods](#methods)
- [How `EnergyVolumeSplit` works](#how-energyvolumesplit-works)
- [Things to know](#things-to-know)
- [Examples](#examples)
- [Package layout](#package-layout)
- [License](#license)
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
from evolute import EnergyVolumeSplit, HamiltonianSystem, integrate


def V(q):
    x, y = q
    return 0.5 * (x * x + y * y) + x * x * y - y ** 3 / 3.0


def grad_V(q):
    x, y = q
    return np.array([x + 2.0 * x * y, y + x * x - y * y])


system = HamiltonianSystem(2, V, grad_V, name='Henon-Heiles')
method = EnergyVolumeSplit()          # symmetric, second order

z0 = np.array([0.1, -0.2, 0.3, 0.15])  # (q1, q2, p1, p2)
z = integrate(method, system, z0, 0.05, 1000)

print(system.energy_error(z0, z))
```

The printed relative energy error is at round-off, here exactly zero.
For fourth order, pass `order=4`; nothing else changes.

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
- optionally a display `name`.

Letting *T* depend on *q* admits curvilinear coordinates. In
cylindrical coordinates `(R, z, phi)`, for example, an axisymmetric
potential gives `T = L^2 / (2 R^2)`, with `L` the momentum conjugate
to `phi`. Such a system is not separable.

`HamiltonianSystem` also provides `H`, `grad_H`, `vector_field` and
`energy_error(z0, z)`, the relative energy error
`|H(z) - H(z0)| / |H(z0)|`.

System-specific definitions live in user scripts rather than in the
package; see [Examples](#examples).

## Methods

Every method has `step(system, z0, h, E=None)`, which returns the new
state as a new array and leaves `z0` unchanged, and the attributes
`name` and `order`. `E` is the energy an energy-conserving method holds
through the step, and the other methods ignore it.

A run goes through `integrate(method, system, z0, h, steps)`, which
returns the final state, or `evolve`, which yields the state after
every step, as in the [quick start](#quick-start).

| Method              | Class               | Orders  | Energy | Volume |
| ------------------- | ------------------- | ------- | ------ | ------ |
| Energy-volume split | `EnergyVolumeSplit` | 1, 2, 4 | exact  | exact  |
| Symplectic          | `Symplectic`        | 1, 2, 4 | no     | exact  |
| Itoh–Abe            | `ItohAbe`           | 1, 2, 4 | exact  | no     |

`EnergyVolumeSplit` and `ItohAbe` are implicit; the `Symplectic`
methods are explicit for a separable system, and otherwise take the
implicit forms of Hairer, Lubich and Wanner, solved by fixed-point
iteration.

"Exact" means up to round-off, and for the implicit methods up to the
solver tolerance `xtol`. The symplectic methods do not conserve energy,
but for these problems their energy error stays bounded rather than
drifting.

### Orders

Each class takes `order`, which is 1, 2 or 4. It defaults to 2 for
`EnergyVolumeSplit` and `Symplectic`, and to 1 for `ItohAbe`, whose
first-order map is the classical method. At orders 1, 2 and 4 the
`Symplectic` methods are symplectic Euler, Störmer–Verlet and
Forest–Ruth.

`EnergyVolumeSplit` and `Symplectic` reach their higher orders by
composing the first-order map *M*. Order 2 is `M*_{h/2} o M_{h/2}`,
half a step of *M* followed by half a step of its adjoint *M*\*, which
is symmetric and therefore of even order. Order 4 composes that
symmetric map three times, at step lengths chosen to ensure
third-order errors vanish. The construction needs a symmetric base
map, which is why it builds on order 2 and not on *M*, and one of its
three step lengths is negative, so a fourth-order step runs partly
backwards in time.

Energy, volume and symplecticity are inherited by such a composition,
since every factor has them at any step length and either sign; only
the `Symplectic` methods are symplectic.

For `Symplectic`, *M* is symplectic Euler, which applies the kick
first, and *M*\* applies the drift first.

`ItohAbe` reaches order 2 differently. Its first-order map sweeps the
coordinates in increasing order; sweeping them in reverse gives the
adjoint, available on its own as `ItohAbe(reverse=True)`. Rather than
composing the two maps, order 2 averages their discrete gradients,
which is symmetric and so second order at one solve per step: the
symmetrized Itoh–Abe gradient of Eidnes (2022). Order 4 is the triple
jump of that symmetric map.

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

## Things to know

- Systems must have the form above, with `n >= 2`.
- The change of variables is singular where `p1 = p2 = 0`. A step that
  reaches it raises `ValueError`; reduce the step size.
- The scalar equations in each substep are contractions only while
  `h |dF/dq| / rho < 1`, with `F = T + V` and
  `rho = sqrt(p1^2 + p2^2)`.
- The implicit solves need a small enough step. A solve that does not
  converge raises `RuntimeError`: the scalar equations of
  `EnergyVolumeSplit`, the root solve of `ItohAbe`, and, for a
  non-separable system, the fixed-point iterations of
  `EnergyVolumeSplit` and `Symplectic`.
- The momentum pair is fixed for the whole run. Switching pairs
  adaptively to avoid the singular set would break volume
  preservation.
- `integrate` and `evolve` take the target energy from the initial
  state and pass it to every step. Recomputing it each step would make
  the rounding of one step the target of the next; the energy error
  would grow like the square root of the number of steps.

## Examples

The scripts in `examples/` integrate every method twice over and write
the results to JSON: a convergence study to a fixed final time at a
sequence of step counts, and a long run at a single step size recording
the energy error:

| Script                     | System                                      |
| -------------------------- | ------------------------------------------- |
| `kepler.py`                | Planar Kepler problem                       |
| `henon_heiles.py`          | Hénon–Heiles system                         |
| `logarithmic_potential.py` | Logarithmic potential, n = 3, not separable |

Each script runs three families, one per order, over step counts of
their own. Each order enters its asymptotic regime at a step of its
own — the first order latest, and `ItohAbe` latest of the three
classes — and leaves it again at the round-off floor, which a
fourth-order method reaches four times faster than a second-order one,
so no single list of step counts serves them all.

`plotting.py` turns the data files into error estimates, observed
orders and figures, each pairing the energy error against time with
the position error against step size. The scripts can be run from any
directory: data files always go to `examples/data/` and figures to
`examples/plots/`.

```bash
python examples/kepler.py
python examples/plotting.py kepler            # one data file
python examples/plotting.py                   # every data file
python examples/plotting.py kepler --thesis   # for the thesis
python examples/plotting.py kepler --journal  # for the journal
```

`plotting.py` draws a figure for one of three pages, and the
arrangement is the same in all three: the panels, their labels and the
legend below them do not move, so a figure drawn for the thesis
differs from the journal one only in its width and its lettering.

- No flag: matplotlib's own figure, 6.4 in wide, in DejaVu Sans.
  Nothing beyond matplotlib is needed, so anyone can redraw these
  figures from the data files.
- `--thesis`: the text width of an A4 page with 25 mm margins,
  160 mm, in Computer Modern, the body typeface of a LaTeX thesis.
- `--journal`: the text width of the *SIAM Journal on
  Numerical Analysis*, 5.125 in, in Computer Modern, as PDF.

The two LaTeX layouts need a local installation, with the `cm-super`
fonts for Computer Modern. Each layout ends the figure's name its own
way, so drawing the same data for two pages leaves two files rather
than one. The `LAYOUTS` table at the top of the script holds each
page's width, lettering and renderer; a thesis class with margins
other than 25 mm wants its own `\textwidth` in `THESIS_WIDTH`.

## Package layout

```text
pyproject.toml
README.md
LICENSE
evolute/
    __init__.py              public API
    system.py                HamiltonianSystem, canonical_J
    integrator.py            OneStepMethod, PartitionedMethod,
                             ImplicitMethod, integrate, evolve
    energy_volume_split.py   EnergyVolumeSplit
    symplectic.py            Symplectic (baseline)
    discrete_gradient.py     ItohAbe (baseline)
examples/
    <test problem>.py        three simulation scripts
    plotting.py              error estimates and figures
```

## License

MIT; see [LICENSE](LICENSE).

## References

- Eidnes, S., 2022. Order theory for discrete gradient methods.
  *BIT Numerical Mathematics, 62*(4), pp.1207-1255.
- Feng, K. and Shang, Z., 1995. Volume-preserving algorithms for
  source-free dynamical systems. *Numerische Mathematik, 71*(4),
  pp.451-463.
- Forest, E. and Ruth, R.D., 1990. Fourth-order symplectic
  integration. *Physica D, 43*(1), pp.105-117.
- Ge, Z. and Marsden, J.E., 1988. Lie-Poisson Hamilton-Jacobi
  theory and Lie-Poisson integrators. *Physics Letters A, 133*(3),
  pp.134-139.
- Hairer, E., Lubich, C. and Wanner, G., 2006. *Geometric numerical
  integration: Structure-preserving algorithms for ordinary
  differential equations*. 2nd ed., Springer.
- Itoh, T. and Abe, K., 1988. Hamiltonian-conserving discrete
  canonical equations based on variational difference quotients.
  *Journal of Computational Physics, 76*(1), pp.85-102.
- Tupper, P.F., 2006. A Non-Existence Result for Hamiltonian
  Integrators. *arXiv preprint math/0607641*.
- Yoshida, H., 1990. Construction of higher order symplectic
  integrators. *Physics Letters A, 150*(5-7), pp.262-268.
