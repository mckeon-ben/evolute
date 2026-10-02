'''
Reads JSON data files, turning the stored final states into error
estimates, observed orders and convergence figures, and the stored
energy histories into energy-error plots.

Run with no arguments to plot every data file in the data folder, or
name one or more files. The figure is drawn for one of three pages:
--journal for the journal's text width and typeface, --thesis for an
A4 thesis in Computer Modern, and, with neither, for matplotlib's own
renderer, which needs no LaTeX installation. The three differ only in
width, lettering and renderer; the arrangement is the same in all.

File contract
-------------
Required: 'schema', 'experiment', 'reference', 'energy_time' and
'families'. 'reference' carries 'kind' and 'state', the exact final
state or None. Each family carries 'label', its own 'dt',
'method_order' and 'methods', the last mapping each display name to
'class', 'order', 'time' -- the seconds that method took -- and
'final_state', one final state per entry of that family's dt, with
'energy_error' recorded at the times in 'energy_time'. Steps are per
family because each order enters its asymptotic regime at a step of
its own and leaves it again at the round-off floor, which a
fourth-order method reaches four times faster than a second-order one,
so no single shared list serves them all.
'parameters' must carry 'T', the final time of the convergence study,
and 'energy_h', the step of the long run.

Anything else in the file is ignored.

From final states to errors
---------------------------
With an exact reference the error is the position error against it.
Otherwise, with q(dt) the positions of the final state stored
for each method at step size dt, the successive-difference norm

    delta(dt) = || q(dt) - q(dt/2) ||

is the error up to a known constant. Writing e(dt) for the error,
delta(dt) = e(dt) - e(dt/2) = e(dt) (1 - 2**-p), so

    e(dt) ~ delta(dt) / (1 - 2**-p),

the error at the coarser step of each pair, leaving the finest run as
the yardstick of the ladder rather than a point of its own. The finest
run is dropped from the exact errors too, so that an example reports
the same step sizes, and the same number of points, whichever of the
two estimators it uses.

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

SCHEMA = 2

# Where data files live when they are not given by an explicit path.
# A bare name on the command line is resolved against the working
# directory first and then here, and passing no name at all plots every
# data file in this folder.
DATA_DIR = Path(__file__).resolve().parent / 'data'

# Where figures are written, kept apart from the data folder so one
# directory holds only inputs and the other only outputs.
PLOT_DIR = Path(__file__).resolve().parent / 'plots'

# Smallest value shown on the energy axis. Steps where the energy error
# is exactly zero in floating point are drawn at this floor.
FLOOR = 1e-17

# A figure is drawn for one of three pages, named on the command line.
# The arrangement is the same in all three -- the panels, their labels
# and the legend beneath them do not move -- so a figure differs
# between them only in the width it is drawn at, its lettering, and
# the renderer that sets the text.
#
#   --journal   the text width and typeface of the journal
#   --thesis    the text width of an A4 thesis, set in Computer
#               Modern, the body typeface of a LaTeX thesis
#   neither     matplotlib's own renderer and fonts, which needs no
#               LaTeX installation, so that anyone can redraw these
#               figures from the data files
#
# Each width is the width the figure is used at, so nothing is scaled
# afterwards and no lettering shrinks: the SIAM class sets its text at
# 5.125 in, an A4 page with 25 mm margins leaves 160 mm, and
# matplotlib's own figure is 6.4 in wide. SIAM prints in black and
# white, so methods are told apart by marker and dash as well as by
# color, in every layout.
JOURNAL_WIDTH = 5.125
THESIS_WIDTH = 160 / 25.4
DEFAULT_WIDTH = 6.4

# LaTeX sets Computer Modern unless told otherwise, so neither page
# that goes through it asks for a font package: the SIAM class sets
# its text in Computer Modern as well.
CM_PREAMBLE = ''

# What matplotlib would set the Computer Modern pages in were their
# usetex turned off below: cmr10, with DejaVu Serif behind it for any
# glyph cmr10 lacks, such as the umlaut in Stormer.
CM_FONTS = ['cmr10', 'DejaVu Serif']

# matplotlib's own, so that the figure drawn without a flag is the one
# its documentation would lead a reader to expect.
DEFAULT_FONTS = ['DejaVu Sans']

# Each layout ends the figure's name its own way, so that drawing the
# same data for two pages leaves two files rather than one.
LAYOUTS = {
    'journal': {'width': JOURNAL_WIDTH, 'font_size': 8, 'line_width': 1.0,
                'suffix': '-journal.pdf', 'usetex': True,
                'preamble': CM_PREAMBLE, 'fonts': CM_FONTS,
                'mathtext': 'cm'},
    'thesis': {'width': THESIS_WIDTH, 'font_size': 10, 'line_width': 1.0,
               'suffix': '-thesis.pdf', 'usetex': True,
               'preamble': CM_PREAMBLE, 'fonts': CM_FONTS,
               'mathtext': 'cm'},
    'default': {'width': DEFAULT_WIDTH, 'font_size': 10, 'line_width': 1.0,
                'suffix': '.pdf', 'usetex': False, 'preamble': '',
                'fonts': DEFAULT_FONTS, 'mathtext': 'dejavusans'},
}

# The figure is this many times as tall as it is wide, so its panels
# keep their shape whichever page it is drawn for.
FIGURE_RATIO = 0.94

# What every layout shares. The fonts and sizes are not here, since
# those are what the layouts differ in; use_layout sets them.
plt.rcParams.update({
    'axes.grid': True,
    # Opaque rather than translucent, which EPS cannot store.
    'grid.color': '0.9',
    'figure.dpi': 150,
    # Embed TrueType rather than the Type 3 fonts matplotlib writes by
    # default.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Marker by method class, so the classes stay distinguishable in black
# and white. Classes not listed fall back to the markers below.
MARKERS = {
    'Symplectic': 'o',
    'ItohAbe': 's',
    'EnergyVolumeSplit': 'D',
}

# Okabe-Ito, the standard colorblind-safe qualitative palette.
PALETTE = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
]

FALLBACK_MARKERS = ['v', 'P', 'X', '*', '<', '>']

# Line style by order, so the order reads off the figure in black and
# white as well as the class does from the marker. Orders not listed
# are drawn solid.
LINE_STYLES = {1: '--', 2: '-', 4: '-.'}


def use_layout(name):
    '''
    Set the lettering, line widths and renderer of one layout.

    Under usetex the font and mathtext settings are ignored, since the
    preamble decides both; they stay in place as the fallback for the
    layout that does without LaTeX.

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
        'font.family': layout['fonts'],
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
    # Anything a layout needs beyond the settings every layout sets.
    plt.rcParams.update(layout.get('rc', {}))
    return layout


def resolve(name):
    '''
    Find a data file given a path, a filename, or a bare stem.

    Tries the working directory before DATA_DIR so an explicit path
    always wins, and appends the .json suffix only when the name does
    not already carry one -- appending unconditionally would mangle a
    name that has dots in it for other reasons.

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

    Tries the preferred side of the guide at points along it, from its
    middle outwards, before the other side. Below the guide the label
    goes to the right of the point, then directly under it; above, to
    the left, then directly over it. Offsetting it sideways first keeps
    it off a rising guide however steep. The first placement that lies
    inside the axes, clear of the frame by the same gap as the guide,
    and touches no line or marker drawn on them is kept; if none does,
    the label takes the first placement tried. Call once the layout is
    final, since the test is made in display coordinates.

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
        Side of the guide tried first; the one away from the data, so
        the label cannot be read as labeling a curve. Default 'below'.

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
    # The label keeps the same gap from the frame as from the guide.
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

    Without a reference state, successive step sizes are compared and
    the difference rescaled into an error estimate, which belongs to the
    coarser step of each pair; the finest run is then the yardstick and
    carries no error of its own. With a reference state the error is the
    L2 position error, and the finest run is dropped from it so that
    both estimators report the same step sizes, dt[:-1], and so the same
    number of points whichever one an example uses.

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
        Display name -> dict with the step sizes 'h' the errors
        correspond to, the position errors 'err', the observed
        'orders', the nominal 'order' and the energy error
        'energy'. Methods keep the order of the families they belong
        to.
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

    The step counts are powers of two, so the labels match the ticks of
    the step-size axis and stay short.

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
            # Each observed order belongs to a pair of step sizes, and
            # is written under the finer of the two.
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

    The panels are stacked, untitled, with one legend below them,
    whichever layout is in use: a caption carries what a title would
    say, and the figure then reads the same on every page.

    Parameters
    ----------
    record : dict
        Record read from a data file.
    panels : dict
        Per-method results, as returned by analyze.
    filename : str or Path
        Output figure path.
    layout : str, optional
        Key of LAYOUTS, which fixes the width the figure is drawn at.
        The lettering and renderer are already in place from
        use_layout.

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
        # The histories are told apart by color, line style and level;
        # markers on traces this dense only add clutter, so they are
        # left to the convergence panel, whose points are the data.
        style = dict(styles[name], marker='None')
        ax_e.semilogy(t, np.maximum(panels[name]['energy'], FLOOR),
                      label=name, **style)
    # The long-run step is a power of two, so it is written as one, as
    # on the step-size axis.
    h_run = record['parameters']['energy_h']
    k = np.log2(h_run)
    h_text = f'2^{{{k:.0f}}}' if k == round(k) else f'{h_run:g}'
    # The numerical solution exists at the step times and nowhere
    # between them, so the axis is t_n = n h rather than a continuous
    # t, which also ties it to the z_n of the ordinate. The step the
    # long run was made at is named here rather than in a title, since
    # it is the unit of this axis and the panels carry no titles.
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
        # A guide spans the steps of its own order alone: the families
        # are measured over different ranges, so one shared span would
        # leave a guide floating away from the curves it describes.
        h = results[0]['h']
        fine = h[len(h) // 2 - 1:]
        above = order == lowest
        errs = [p['err'][-1] for p in results]
        e0 = 2.0 * max(errs) if above else 0.5 * min(errs)
        guide = e0 * (fine / fine[-1]) ** order
        ax_c.loglog(fine, guide, color='0.5', linestyle=':')
        # Each guide carries the steps it was drawn over, since they
        # differ between orders and the labels are placed later.
        guides.append((rf'$O\left(h^{{{order}}}\right)$' if order > 1
                       else r'$O\left(h\right)$', fine, guide,
                       'above' if above else 'below'))
    ax_c.set_xscale('log', base=2)
    # Three times matplotlib's default headroom, so the guide labels have
    # room above the upper guide and below the lower one.
    ax_c.margins(y=0.15)
    # As on the other panel, the final time the errors were measured at
    # is named on the axis rather than in a title.
    ax_c.set_xlabel(f'$h$, $T = {record["parameters"]["T"]:.4g}$')
    ax_c.set_ylabel(r'$\|q_N - q(T)\|_2$' if record['reference']['state']
                    else r'Estimated $\|q_N - q(T)\|_2$')

    # One legend under the stacked panels, where it covers no data.
    handles, labels = ax_c.get_legend_handles_labels()
    # Three columns, with enough height for the rows they take. The
    # legend is set a point below the body text, with its handles and
    # columns drawn in, which keeps three columns of these names
    # inside even the narrowest of the pages above.
    ncol = 3
    legend_height = 0.2 * -(-len(labels) // ncol)
    fig.tight_layout(rect=(0, legend_height / fig.get_figheight(), 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=ncol,
               frameon=False, fontsize=page['font_size'] - 1,
               handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
    # The guides are labeled once the layout is final, where no line or
    # marker is in the way.
    renderer = fig.canvas.get_renderer()
    for text, steps, guide, side in guides:
        place_label(ax_c, text, steps, guide, renderer, side)
    # Saved at the width it was drawn at, so the figure arrives on the
    # page at the size its lettering was chosen for.
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
    # One page at a time, and matplotlib's own when neither is asked
    # for, which is the layout that needs no LaTeX installation.
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
            # Report and carry on rather than abandoning the batch: a
            # results folder may hold unrelated JSON, and one bad file
            # should not cost the figures for the good ones.
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
