'''
Reads JSON data files, turning the stored final states into error
estimates and observed orders and the stored energy histories into
energy-error traces, prints the tables and draws the figure.

File contract
-------------
Required: 'schema', 'experiment', 'reference', 'energy_time',
'parameters' and 'families'. 'reference' carries 'kind' and 'state',
the exact final state or None. 'parameters' carries 'T', the final
time of the convergence study, and 'energy_h', the step of the long
run. Each family carries 'label', its own 'dt' (its step sizes),
'method_order' and 'methods', which maps each display name to 'class',
'order', 'time' (seconds taken), 'final_state', one final state per
entry of 'dt', and 'energy_error', one value per entry of
'energy_time'. Each family has its own 'dt' because each order enters
its asymptotic regime at a step of its own and leaves it at the
round-off floor, which a fourth-order method reaches four times
faster.
'''

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEMA = 2

# Inputs and outputs, beside this script wherever it is run from.
DATA_DIR = Path(__file__).resolve().parent / 'data'
PLOT_DIR = Path(__file__).resolve().parent / 'plots'

# Smallest value shown on the energy axis; an error of exactly zero in
# floating point is drawn at this floor.
FLOOR = 1e-17

# Page widths in inches, drawn at final size so nothing is rescaled:
# the SIAM text width (5.125 in), an A4 page with 25 mm margins
# (160 mm), and matplotlib's default. The layouts differ only in width,
# lettering and renderer; only the default needs no LaTeX.
JOURNAL_WIDTH = 5.125
THESIS_WIDTH = 160 / 25.4
DEFAULT_WIDTH = 6.4

# Computer Modern is LaTeX's default, and the SIAM text font, so
# neither page that goes through LaTeX needs a package.
CM_PREAMBLE = ''
CM_FONTS = ['cmr10', 'DejaVu Serif']

DEFAULT_FONTS = ['DejaVu Sans']

# Distinct suffixes, so drawing for two pages leaves two files.
LAYOUTS = {
    'journal': {'width': JOURNAL_WIDTH, 'font_size': 8, 'line_width': 1.0,
                'suffix': '-journal.pdf', 'usetex': True,
                'preamble': CM_PREAMBLE, 'family': 'serif',
                'fonts': CM_FONTS, 'mathtext': 'cm'},
    'thesis': {'width': THESIS_WIDTH, 'font_size': 10, 'line_width': 1.0,
               'suffix': '-thesis.pdf', 'usetex': True,
               'preamble': CM_PREAMBLE, 'family': 'serif',
               'fonts': CM_FONTS, 'mathtext': 'cm'},
    'default': {'width': DEFAULT_WIDTH, 'font_size': 10, 'line_width': 1.0,
                'suffix': '.pdf', 'usetex': False, 'preamble': '',
                'family': 'sans-serif', 'fonts': DEFAULT_FONTS,
                'mathtext': 'dejavusans'},
}

# Height over width, the same on every page.
FIGURE_RATIO = 0.94

# Settings shared by every layout.
plt.rcParams.update({
    'axes.grid': True,
    'grid.color': '0.9',
    'figure.dpi': 150,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def use_layout(name):
    '''
    Set the lettering, line widths and renderer of one layout.

    The font list and mathtext set matter only when LaTeX is off.

    Parameters
    ----------
    name : str
        Key of LAYOUTS.

    Returns
    -------
    dict
        The layout, whose width and suffix the caller still needs.
    '''
    layout = LAYOUTS[name]
    size, width = layout['font_size'], layout['line_width']
    plt.rcParams.update({
        'text.usetex': layout['usetex'],
        'text.latex.preamble': layout['preamble'],
        'font.family': layout['family'],
        f'font.{layout["family"]}': layout['fonts'],
        'mathtext.fontset': layout['mathtext'],
        'axes.formatter.use_mathtext': True,
        'font.size': size,
        'axes.titlesize': size,
        'axes.labelsize': size,
        'xtick.labelsize': size,
        'ytick.labelsize': size,
        'legend.fontsize': size,
        'lines.linewidth': width,
        'axes.linewidth': width,
        'grid.linewidth': width,
        'xtick.major.width': width,
        'ytick.major.width': width,
        'xtick.minor.width': width,
        'ytick.minor.width': width,
        'lines.markersize': 3.5,
    })
    plt.rcParams.update(layout.get('rc', {}))
    return layout


# Fixed markers by method class; other classes use FALLBACK_MARKERS.
# SIAM prints in black and white, so the marker tells the classes
# apart there and LINE_STYLES tells the orders apart.
MARKERS = {
    'Symplectic': 'o',
    'ItohAbe': 's',
    'EnergyVolumeSplit': 'D',
}

# Okabe-Ito colorblind-safe palette.
PALETTE = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
]

FALLBACK_MARKERS = ['v', 'P', 'X', '*', '<', '>']

# Line style by order; other orders are drawn solid.
LINE_STYLES = {1: '--', 2: '-', 4: '-.'}


def resolve(name):
    '''
    Find a data file given a path, a filename, or a bare stem.

    Looks in the working directory, then DATA_DIR, adding .json if
    the name lacks it.

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
    given = Path(name)
    names = ([given] if given.suffix == '.json'
             else [given, given.with_name(given.name + '.json')])
    for base in (Path('.'), DATA_DIR):
        for candidate in names:
            full = candidate if candidate.is_absolute() else base / candidate
            if full.is_file():
                return full
    raise SystemExit(f'{name}: no such data file in . or {DATA_DIR}/')


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
        raise SystemExit(f'no .json files found in {DATA_DIR}/')
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
    for key in ('experiment', 'parameters', 'reference', 'energy_time',
                'families'):
        if key not in record:
            raise ValueError(f'{filename}: missing required key {key!r}')
    for family in record['families']:
        if 'dt' not in family:
            raise ValueError(f'{filename}: family {family.get("label")!r} '
                             f'has no step sizes')
    return record


def entries(record):
    '''
    Yield every method in the record, family by family.

    Parameters
    ----------
    record : dict
        Record read from a data file.

    Yields
    ------
    name : str
        Display name of the method.
    entry : dict
        Its stored results.
    dt : np.ndarray
        The step sizes of its family.
    '''
    for family in record['families']:
        dt = np.asarray(family['dt'], dtype=float)
        for name in family['method_order']:
            yield name, family['methods'][name], dt


def assign_styles(record):
    '''
    Line style, marker and color for every method.

    Methods of the same class share a color and a marker, and differ
    by line style, which follows the order through LINE_STYLES. The
    marker keeps the classes apart in black and white.

    Parameters
    ----------
    record : dict
        Record read from a data file.

    Returns
    -------
    dict
        Display name -> dict of matplotlib line properties.
    '''
    colors, markers, styles = {}, {}, {}
    for name, entry, _ in entries(record):
        cls = entry['class']
        colors.setdefault(cls, PALETTE[len(colors) % len(PALETTE)])
        markers.setdefault(cls, MARKERS.get(
            cls, FALLBACK_MARKERS[len(markers) % len(FALLBACK_MARKERS)]))
        styles[name] = {
            'color': colors[cls],
            'linestyle': LINE_STYLES.get(entry['order'], '-'),
            'marker': markers[cls],
        }
    return styles


def place_label(ax, text, x, y, renderer, side='below'):
    '''
    Label a guide line where the label overlaps no line.

    Tries points along the guide from its middle outwards, on the
    preferred side first, and keeps the first placement inside the
    axes that touches no line or marker; failing that, the first one
    tried. Call once the layout is final, since the test is made in
    display coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes holding the guide.
    text : str
        Label text.
    x, y : np.ndarray
        Points of the guide, on log-log axes.
    renderer : matplotlib.backend_bases.RendererBase
        Renderer used to measure the label.
    side : {'below', 'above'}, optional
        Side of the guide tried first. Default 'below'.

    Returns
    -------
    matplotlib.text.Annotation
        The placed label.
    '''
    lx, ly = np.log(x), np.log(y)
    order = np.argsort(lx)
    lines = ax.get_lines()
    paths = [line.get_transform().transform_path(line.get_path())
             for line in lines]
    points = [line.get_transform().transform(line.get_xydata())
              for line in lines if line.get_marker() not in (None, 'None')]
    pad = renderer.points_to_pixels(plt.rcParams['lines.markersize'])
    gap = 4
    frame = ax.get_window_extent(renderer).padded(
        -renderer.points_to_pixels(gap))
    label = ax.annotate(text, xy=(x[0], y[0]), xytext=(0, 0),
                        textcoords='offset points')
    # Offset in points, then horizontal and vertical alignment.
    placements = {
        'below': [((gap, -gap), 'left', 'top'), ((0, -gap), 'center', 'top')],
        'above': [((-gap, gap), 'right', 'bottom'),
                  ((0, gap), 'center', 'bottom')],
    }
    other = 'above' if side == 'below' else 'below'
    candidates = []
    for sides in (placements[side], placements[other]):
        for fraction in (0.5, 0.3, 0.7, 0.1, 0.9):
            at = lx.min() + fraction * (lx.max() - lx.min())
            xy = (np.exp(at), np.exp(np.interp(at, lx[order], ly[order])))
            candidates += [(xy, placement) for placement in sides]

    def put(xy, placement):
        offset, ha, va = placement
        label.xy = xy
        label.set_position(offset)
        label.set_ha(ha)
        label.set_va(va)

    for xy, placement in candidates:
        put(xy, placement)
        box = label.get_window_extent(renderer)
        inside = (frame.x0 <= box.x0 and box.x1 <= frame.x1
                  and frame.y0 <= box.y0 and box.y1 <= frame.y1)
        clear = (not any(path.intersects_bbox(box, filled=False)
                         for path in paths)
                 and not any(box.padded(pad).contains(px, py)
                             for pts in points for px, py in pts))
        if inside and clear:
            return label
    put(*candidates[0])
    return label


def differences(states, dt, order, reference=None):
    '''
    Position errors and observed orders for one method.

    Errors align with dt[:-1] under either estimator: a successive
    difference belongs to the coarser step of its pair, and the finest
    run is dropped from the exact errors to match.

    Parameters
    ----------
    states : array_like
        Final states, shape (len(dt), 2n), one per step size.
    dt : array_like
        Step sizes, each half the one before.
    order : int
        Nominal order, used to rescale differences into errors.
    reference : array_like, optional
        Exact final state, shape (2n,).

    Returns
    -------
    err : np.ndarray
        Position errors, shape (len(dt) - 1,).
    orders : np.ndarray
        Observed orders, shape (len(dt) - 2,).
    '''
    states = np.asarray(states, dtype=float)
    dt = np.asarray(dt, dtype=float)
    half = states.shape[1] // 2
    q = states[:, :half]
    if reference is not None:
        err = np.linalg.norm(q[:-1] - np.asarray(reference)[:half], axis=1)
    else:
        scale = 1.0 / (1.0 - 2.0 ** -order)
        err = scale * np.linalg.norm(np.diff(q, axis=0), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        orders = (np.log(err[:-1] / err[1:])
                  / np.log(dt[:-2] / dt[1:-1]))
    return err, orders


def analyze(record):
    '''
    Errors and observed orders for every method in a record.

    Parameters
    ----------
    record : dict
        Record read from a data file.

    Returns
    -------
    dict
        Display name -> dict of step sizes 'h', position errors
        'err', observed 'orders', nominal 'order' and energy error
        'energy', in the order of the families.
    '''
    ref = record['reference']['state']
    out = {}
    for name, entry, dt in entries(record):
        err, orders = differences(entry['final_state'], dt,
                                  entry['order'], ref)
        out[name] = {'h': dt[:-1], 'err': err, 'orders': orders,
                     'order': entry['order'],
                     'energy': np.asarray(entry['energy_error'])}
    return out


def step_label(h):
    '''
    Label a step size, as a power of two where it is one.

    Parameters
    ----------
    h : float
        Step size.

    Returns
    -------
    str
        Label such as '2^-9', or two significant figures otherwise.
    '''
    k = np.log2(h)
    return f'2^{round(k):d}' if abs(k - round(k)) < 1e-9 else f'{h:.2e}'


def print_tables(record, panels):
    '''
    Print the estimated errors and observed orders for every family.

    Parameters
    ----------
    record : dict
        Record read from a data file; supplies the heading.
    panels : dict
        Per-method results, as returned by analyze.
    '''
    kind = record['reference']['kind']
    print(f'\n{record["experiment"]}  (reference: {kind})')
    for family in record['families']:
        names = family['method_order']
        width = max(len(name) for name in names)
        h = panels[names[0]]['h']
        header = (f'{"Method":>{width}} |'
                  + ''.join(f'{step_label(x):>9}' for x in h))
        print('#' * len(header))
        print(f'# {family["label"]} methods')
        print('#' * len(header))
        print(header)
        for name in names:
            p = panels[name]
            print(f'{name:>{width}} |'
                  + ''.join(f'{e:>9.2e}' for e in p['err']))
            # Each order sits under the finer step of its pair.
            print(f'{"order":>{width}} |' + ' ' * 9
                  + ''.join(f'{o:>9.2f}' for o in p['orders']))
        print()
    width = max(len(name) for name in panels)
    print('Maximum energy error over the long run')
    for name, p in panels.items():
        print(f'{name:>{width}} | {p["energy"].max():>8.2e}')


def plot(record, panels, filename, layout='default'):
    '''
    Energy error against time, and position error against step size.

    The panels are stacked and untitled, with one shared legend below.

    Parameters
    ----------
    record : dict
        Record read from a data file.
    panels : dict
        Per-method results, as returned by analyze.
    filename : str or Path
        Output figure path.
    layout : str, optional
        Key of LAYOUTS, giving the figure width; lettering is already
        set by use_layout.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, already saved to filename.
    '''
    styles = assign_styles(record)
    t = np.asarray(record['energy_time'])
    page = LAYOUTS[layout]
    fig, axes = plt.subplots(
        2, 1, figsize=(page['width'], FIGURE_RATIO * page['width']))
    ax_e, ax_c = axes

    for name in panels:
        # No markers: these traces are too dense to carry them.
        style = dict(styles[name], marker='None')
        ax_e.semilogy(t, np.maximum(panels[name]['energy'], FLOOR),
                      label=name, **style)
    h_run = record['parameters']['energy_h']
    k = np.log2(h_run)
    h_text = f'2^{{{k:.0f}}}' if k == round(k) else f'{h_run:g}'
    # The solution exists at the step times and nowhere between, so
    # the axis is t_n = n h, which also ties it to the z_n above.
    ax_e.set_xlabel(f'$t_n = n h$, $h = {h_text}$')
    ax_e.set_ylabel(r'$|H(z_n) - H(z_0)| / |H(z_0)|$')
    ax_e.set_ylim(bottom=FLOOR)

    by_order = {}
    for name, p in panels.items():
        ax_c.loglog(p['h'], p['err'], label=name, **styles[name])
        by_order.setdefault(p['order'], []).append(p)

    lowest = min(by_order)
    guides = []
    for order, results in sorted(by_order.items()):
        # Each guide spans the steps of its own order, as the families
        # are measured over different ranges.
        h = results[0]['h']
        fine = h[len(h) // 2 - 1:]
        above = order == lowest
        errs = [p['err'][-1] for p in results]
        e0 = 2.0 * max(errs) if above else 0.5 * min(errs)
        guide = e0 * (fine / fine[-1]) ** order
        ax_c.loglog(fine, guide, color='0.5', linestyle=':')
        guides.append((rf'$O\left(h^{{{order}}}\right)$' if order > 1
                       else r'$O\left(h\right)$', fine, guide,
                       'above' if above else 'below'))
    ax_c.set_xscale('log', base=2)
    # Extra headroom for the guide labels.
    ax_c.margins(y=0.15)
    ax_c.set_xlabel(f'$h$, $T = {record["parameters"]["T"]:.4g}$')
    ax_c.set_ylabel(r'$\|q_N - q(T)\|_2$' if record['reference']['state']
                    else r'Estimated $\|q_N - q(T)\|_2$')

    # One legend under the stacked panels, where it covers no data.
    handles, labels = ax_c.get_legend_handles_labels()
    # Three columns, a point smaller than the text, to fit the
    # narrowest page.
    ncol = 3
    legend_height = 0.2 * -(-len(labels) // ncol)
    fig.tight_layout(rect=(0, legend_height / fig.get_figheight(), 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=ncol,
               frameon=False, fontsize=page['font_size'] - 1,
               handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
    # Labeled once the layout is final, clear of the lines.
    renderer = fig.canvas.get_renderer()
    for text, steps, guide, side in guides:
        place_label(ax_c, text, steps, guide, renderer, side)
    fig.savefig(filename)
    print(f'Saved {filename}')
    return fig


def main():
    '''
    Plot the data files named on the command line, or all of them.

    Raises
    ------
    SystemExit
        If -o is given with several data files, or if any file is
        skipped.
    '''
    parser = argparse.ArgumentParser(
        description=' '.join(__doc__.split('\n\n')[0].split()))
    parser.add_argument('results', nargs='*',
                        help=f'data files, by path or bare name; '
                             f'omit to plot every .json in '
                             f'{DATA_DIR}/')
    parser.add_argument('-o', '--output', default=None,
                        help=f'output figure (default: {PLOT_DIR}/<name>'
                             f', with the suffix the layout calls for)')
    # At most one page; the default needs no LaTeX.
    page = parser.add_mutually_exclusive_group()
    page.add_argument('--journal', action='store_const', dest='layout',
                      const='journal',
                      help='draw at the text width and in the typeface '
                           'of the journal; needs LaTeX')
    page.add_argument('--thesis', action='store_const', dest='layout',
                      const='thesis',
                      help='draw at the text width of an A4 thesis, in '
                           'Computer Modern; needs LaTeX')
    parser.set_defaults(layout='default')
    args = parser.parse_args()

    layout = args.layout
    suffix = use_layout(layout)['suffix']

    files = ([resolve(name) for name in args.results] if args.results
             else find_all())
    if args.output and len(files) > 1:
        raise SystemExit('-o takes a single data file; with several, '
                         'each figure is named after its own input')

    if not args.output:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in files:
        output = args.output or str(
            PLOT_DIR / (path.stem + suffix))
        try:
            record = load(path)
        except (ValueError, KeyError) as exc:
            # Skip a bad file rather than abandon the batch.
            print(f'{path}: skipped ({exc})')
            failed += 1
            continue
        print(f'\n{path}')
        panels = analyze(record)
        print_tables(record, panels)
        plt.close(plot(record, panels, output, layout))
    if failed:
        raise SystemExit(f'{failed} file(s) skipped')


if __name__ == '__main__':
    main()
