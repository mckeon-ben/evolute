'''
Reads JSON data files, turning the stored final states into error
estimates, observed orders and convergence figures, and the stored
energy histories into energy-error plots. Data files that also store
momentum histories get a momentum-error panel as well.

Run with no arguments to plot every data file in the data folder, or
name one or more files.

File contract
-------------
Required: 'schema', 'experiment', 'dt', 'reference', 'energy_time',
'method_order' and 'methods'. 'reference' carries 'kind' and 'state',
the exact final state or None. 'methods' maps each display name to
'class', 'order', 'final_state' -- one final state per entry of dt --
and 'energy_error', sampled at the times in 'energy_time'. 'parameters'
must carry 'T', the final time of the convergence study, and
'energy_h', the step of the long run.

Optional: 'relative_energy' (default True) chooses the energy axis
label; a method's 'coordinates' separates its colour from another
method of the same class; and 'momentum_error', present for every
method, adds a momentum panel, labelled with 'momentum_symbol'
(default 'L'). Anything else in the file is ignored.

From final states to errors
---------------------------
With an exact reference the error is the position error at each step
size. Otherwise, with q(dt) the positions of the final state stored
for each method at step size dt, the successive-difference norm

    delta(dt) = || q(dt) - q(dt/2) ||

is the error up to a known constant. Writing e(dt) for the error,
delta(dt) = e(dt) - e(dt/2) = e(dt) (1 - 2**-p), so

    e(dt) ~ delta(dt) / (1 - 2**-p),

the error at the coarser step of each pair.

The observed order is taken from the ratio of consecutive errors
against the ratio of their step sizes,

    p = log(e_i / e_{i+1}) / log(dt_i / dt_{i+1}),

rather than as log2 of the error ratio, so a step list that is not
exactly halved throughout does not silently misreport the order.
'''

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCHEMA = 1

# Where data files live when they are not given by an explicit path.
# A bare name on the command line is resolved against the working
# directory first and then here, and passing no name at all plots every
# data file in this folder.
DATA_DIR = Path(__file__).resolve().parent / 'data'

# Where figures are written, kept apart from the data folder so one
# directory holds only inputs and the other only outputs.
PLOT_DIR = Path(__file__).resolve().parent / 'plots'

# Default output format. Vector, so the figure stays sharp at any zoom
# and at whatever size a paper puts it.
FIGURE_SUFFIX = '.pdf'

# Smallest value shown on the energy axis. Steps where the energy error
# is exactly zero in floating point are drawn at this floor.
FLOOR = 1e-17

# Typeset through a local LaTeX installation instead of matplotlib's
# own engine. Set to False on a machine without LaTeX.
USETEX = True

LATEX_PREAMBLE = r'''
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{sansmath}
\sansmath
'''

FONT_STACK = ['Helvetica', 'Arial', 'TeX Gyre Heros', 'Nimbus Sans',
              'Liberation Sans', 'FreeSans', 'DejaVu Sans']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': FONT_STACK,
    'mathtext.fontset': 'dejavusans',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})


def use_latex(enabled=USETEX):
    '''
    Switch LaTeX typesetting on or off.

    Under usetex the mathtext and font.sans-serif settings above are
    ignored, since the preamble decides both; they stay in place as the
    fallback for when this is off.

    Parameters
    ----------
    enabled : bool, optional
        Whether to typeset through LaTeX.
    '''
    plt.rcParams.update({
        'text.usetex': enabled,
        'text.latex.preamble': LATEX_PREAMBLE if enabled else '',
    })


def resolve(name):
    '''
    Find a data file given a path, a filename, or a bare stem.

    Tries the working directory before DATA_DIR so an explicit path
    always wins, and appends the .json suffix only when the name does
    not already carry one.

    Parameters
    ----------
    name : str
        Path, filename or bare stem of the data file.

    Returns
    -------
    Path
        The data file found.

    Raises
    ------
    SystemExit
        If no matching file exists.
    '''
    candidates = [Path(name), DATA_DIR / name]
    if not name.endswith('.json'):
        candidates += [Path(name + '.json'), DATA_DIR / (name + '.json')]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(f'no data file matching {name!r} in . or {DATA_DIR}/')


def find_all():
    '''
    Every data file in DATA_DIR, sorted by name.

    Returns
    -------
    list of Path
        The data files found.

    Raises
    ------
    SystemExit
        If DATA_DIR holds no .json files.
    '''
    files = sorted(DATA_DIR.glob('*.json'))
    if not files:
        raise SystemExit(f'no data files in {DATA_DIR}/')
    return files


def load(filename):
    '''
    Read a data file, checking the schema and the required keys.

    Parameters
    ----------
    filename : str or Path
        Data file to read.

    Returns
    -------
    dict
        The record.

    Raises
    ------
    ValueError
        If the schema is wrong, or a required key is missing.
    '''
    with open(filename) as fh:
        record = json.load(fh)
    if record.get('schema') != SCHEMA:
        raise ValueError(f'{filename}: schema {record.get("schema")!r}, '
                         f'expected {SCHEMA}')
    for key in ('experiment', 'dt', 'reference', 'energy_time',
                'method_order', 'methods'):
        if key not in record:
            raise ValueError(f'{filename}: missing key {key!r}')
    return record


def assign_styles(record):
    '''
    Line style, marker and colour for every method.

    Methods of the same class and coordinates share a colour, so a
    first-order method and its symmetric composition are drawn in one
    colour; first-order methods are dashed and second-order ones solid.

    Parameters
    ----------
    record : dict
        Record read from a data file.

    Returns
    -------
    dict
        Display name -> dict of matplotlib line properties.
    '''
    colours, styles = {}, {}
    for name in record['method_order']:
        entry = record['methods'][name]
        key = (entry['class'], entry.get('coordinates'))
        colours.setdefault(key, f'C{len(colours)}')
        styles[name] = {
            'color': colours[key],
            'linestyle': '--' if entry['order'] == 1 else '-',
            'marker': 'o' if entry['order'] == 2 else 's',
        }
    return styles


def differences(states, dt, order, reference=None):
    '''
    Position errors and observed orders for one method.

    With a reference state the error is the L2 position error at each
    step size. Without one, successive step sizes are compared and the
    difference rescaled into an error estimate, which is then indexed by
    the coarser step of each pair, so it aligns with dt[:-1].

    Parameters
    ----------
    states : array_like
        Final states, shape (len(dt), 2n), one per step size.
    dt : array_like
        Step sizes, each half the one before.
    order : int
        Nominal order, used to rescale the successive differences.
    reference : array_like, optional
        Exact final state, shape (2n,).

    Returns
    -------
    h : np.ndarray
        Step sizes the errors correspond to.
    err : np.ndarray
        Position errors.
    orders : np.ndarray
        Observed orders, shorter than err by one.
    '''
    states = np.asarray(states, dtype=float)
    dt = np.asarray(dt, dtype=float)
    half = states.shape[1] // 2
    q = states[:, :half]
    if reference is not None:
        h = dt
        err = np.linalg.norm(q - np.asarray(reference)[:half], axis=1)
    else:
        h = dt[:-1]
        err = np.linalg.norm(np.diff(q, axis=0), axis=1) / (1 - 2.0 ** -order)
    with np.errstate(divide='ignore', invalid='ignore'):
        orders = np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])
    return h, err, orders


def analyse(record):
    '''
    Errors and observed orders for every method in a record.

    Parameters
    ----------
    record : dict
        Record read from a data file.

    Returns
    -------
    dict
        Display name -> dict with the step sizes 'h', the position
        errors 'err', the observed 'orders', the sampled energy error
        'energy' and, if stored, the sampled momentum error 'momentum'.
    '''
    ref = record['reference']['state']
    out = {}
    for name in record['method_order']:
        entry = record['methods'][name]
        h, err, orders = differences(entry['final_state'], record['dt'],
                                     entry['order'], ref)
        out[name] = {'h': h, 'err': err, 'orders': orders,
                     'energy': np.asarray(entry['energy_error'])}
        if 'momentum_error' in entry:
            out[name]['momentum'] = np.asarray(entry['momentum_error'])
    return out


def has_momentum(panels):
    '''
    Whether every method in a record has a momentum history.

    Parameters
    ----------
    panels : dict
        Per-method results, as returned by analyse.

    Returns
    -------
    bool
        True if a momentum panel can be drawn.
    '''
    return all('momentum' in p for p in panels.values())


def print_tables(record, panels):
    '''
    Print the maximum errors and the observed order of each method.

    The maximum momentum error is included when every method has one.

    Parameters
    ----------
    record : dict
        Record read from a data file.
    panels : dict
        Per-method results, as returned by analyse.
    '''
    kind = record['reference']['kind']
    momentum = has_momentum(panels)
    width = max(26, max(len(name) for name in record['method_order']))
    rule = '-' * (width + 46 + (21 if momentum else 0))
    print(f'\n{record["experiment"]}  (reference: {kind})')
    print(rule)
    print(f'{"Method":<{width}s} {"max energy error":>17s}'
          + (f' {"max momentum error":>20s}' if momentum else '')
          + f' {"order":>7s} {"finest error":>14s}')
    print(rule)
    for name in record['method_order']:
        p = panels[name]
        print(f'{name:<{width}s} {p["energy"].max():>17.2e}'
              + (f' {p["momentum"].max():>20.2e}' if momentum else '')
              + f' {p["orders"][-1]:>7.2f} {p["err"][-1]:>14.2e}')


def plot(record, panels, filename):
    '''
    Energy error against time, and position error against step size.

    When every method has a momentum history, a panel of the momentum
    error against time is drawn between the two.

    Parameters
    ----------
    record : dict
        Record read from a data file.
    panels : dict
        Per-method results, as returned by analyse.
    filename : str or Path
        Output figure path.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, already saved to filename.
    '''
    styles = assign_styles(record)
    t = np.asarray(record['energy_time'])
    relative = record.get('relative_energy', True)
    momentum = has_momentum(panels)
    if momentum:
        fig, (ax_e, ax_m, ax_c) = plt.subplots(1, 3, figsize=(16, 4.4))
    else:
        fig, (ax_e, ax_c) = plt.subplots(1, 2, figsize=(11, 4.4))

    for name in record['method_order']:
        style = dict(styles[name], marker=None, linewidth=1.0)
        ax_e.semilogy(t, np.maximum(panels[name]['energy'], FLOOR),
                      label=name, **style)
        if momentum:
            ax_m.semilogy(t, np.maximum(panels[name]['momentum'], FLOOR),
                          label=name, **style)
    ax_e.set_xlabel('time $t$')
    ax_e.set_ylabel(r'$|H(z_n) - H(z_0)| / |H(z_0)|$' if relative
                    else r'$|H(z_n) - H(z_0)|$')
    ax_e.set_title(f'energy error, $h = {record["parameters"]["energy_h"]}$')
    ax_e.set_ylim(bottom=FLOOR)
    if momentum:
        symbol = record.get('momentum_symbol', 'L')
        ax_m.set_xlabel('time $t$')
        ax_m.set_ylabel(rf'$|{symbol}(z_n) - {symbol}(z_0)|$')
        ax_m.set_title(
            f'momentum error, $h = {record["parameters"]["energy_h"]}$')
        ax_m.set_ylim(bottom=FLOOR)

    finest = {}
    for name in record['method_order']:
        p = panels[name]
        ax_c.loglog(p['h'], p['err'], label=name, markersize=3,
                    linewidth=1.0, **styles[name])
        order = record['methods'][name]['order']
        finest.setdefault(order, []).append(p['err'][-1])

    h = panels[record['method_order'][0]]['h']
    fine = h[len(h) // 2 - 1:]
    lowest = min(finest)
    for order, errs in sorted(finest.items()):
        above = order == lowest
        e0 = 2.0 * max(errs) if above else 0.5 * min(errs)
        ax_c.loglog(fine, e0 * (fine / fine[-1]) ** order, color='0.5',
                    linestyle=':', linewidth=1.0)
        h_mid = np.sqrt(fine[0] * fine[-1])
        ax_c.annotate(rf'$\mathcal{{O}}(h^{order})$' if order > 1
                      else r'$\mathcal{O}(h)$',
                      xy=(h_mid, e0 * (h_mid / fine[-1]) ** order),
                      xytext=(0, 5 if above else -5),
                      textcoords='offset points', ha='center',
                      va='bottom' if above else 'top', color='0.4',
                      fontsize=9)
    ax_c.set_xlabel('step size $h$')
    ax_c.set_ylabel(r'$\|q_N - q(T)\|_2$' if record['reference']['state']
                    else r'estimated $\|q_N - q(T)\|_2$')
    ax_c.set_title(f'position error at $T = {record["parameters"]["T"]:.4g}$')
    ax_c.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    fig.suptitle(record['experiment'])
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    print(f'Saved {filename}')
    return fig


def main():
    '''
    Plot the data files named on the command line, or all of them.

    Raises
    ------
    SystemExit
        If -o is given with several data files.
    '''
    parser = argparse.ArgumentParser(
        description=' '.join(__doc__.split('\n\n')[0].split()))
    parser.add_argument('results', nargs='*',
                        help='data files; default is every file in '
                             f'{DATA_DIR}/')
    parser.add_argument('-o', '--output', help='output figure path')
    args = parser.parse_args()

    files = [resolve(name) for name in args.results] if args.results \
        else find_all()
    if args.output and len(files) > 1:
        raise SystemExit('-o takes a single data file')

    use_latex()
    PLOT_DIR.mkdir(exist_ok=True)
    for path in files:
        record = load(path)
        panels = analyse(record)
        print_tables(record, panels)
        out = args.output or PLOT_DIR / (path.stem + FIGURE_SUFFIX)
        plot(record, panels, out)


if __name__ == '__main__':
    main()
