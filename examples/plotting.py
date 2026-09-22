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

Optional: 'momentum_error', present for every method, adds a momentum
panel, labelled with 'momentum_symbol' (default 'L'). Anything else in
the file is ignored.

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

# Print layout for the SIAM Journal on Numerical Analysis, used with
# --print. The figure is drawn at the text width of the SIAM class,
# 5.125 in, so no text or line is scaled down: lines of at least 1 pt,
# all text in black, and vector PDF. SIAM prints in black and white, so
# methods are told apart by marker and dash as well as colour. The text
# is in Computer Modern, the typeface of the SIAM class, rather than the
# Helvetica of the screen figures.
PRINT_WIDTH = 5.125
PRINT_FONT_SIZE = 8
PRINT_LINE_WIDTH = 1.0
PRINT_SUFFIX = '.pdf'

# LaTeX's own default is Computer Modern, so the print preamble needs no
# font packages.
PRINT_LATEX_PREAMBLE = ''

# Fallback for when LaTeX is off, from fonts matplotlib ships: Computer
# Modern as cmr10, with tick labels set as maths to keep their minus
# signs, and DejaVu Serif for any glyph cmr10 lacks, such as the umlaut
# in Stormer.
PRINT_FONT_STACK = ['cmr10', 'DejaVu Serif']

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
\DeclareSymbolFont{sansletters}{OT1}{phv}{m}{n}
\DeclareMathSymbol{-}{\mathbin}{sansletters}{"2D}
'''

FONT_STACK = ['Helvetica', 'Arial', 'TeX Gyre Heros', 'Nimbus Sans',
              'Liberation Sans', 'FreeSans', 'DejaVu Sans']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': FONT_STACK,
    'mathtext.fontset': 'dejavusans',
    'axes.grid': True,
    # Opaque rather than translucent, which EPS cannot store.
    'grid.color': '0.9',
    'figure.dpi': 150,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.0,
    'legend.fontsize': 9,
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
    'Gonzalez': '^',
    'EnergyVolumeSplit': 'D',
}

# Okabe-Ito, the standard colourblind-safe qualitative palette.
PALETTE = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
]

FALLBACK_MARKERS = ['v', 'P', 'X', '*', '<', '>']


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


def use_print_layout():
    '''
    Switch to the journal's print sizes for text and lines.

    All text is set in Computer Modern at PRINT_FONT_SIZE and every line,
    including the axes, ticks and grid, at PRINT_LINE_WIDTH. Call after
    use_latex, whose preamble this replaces.
    '''
    size, width = PRINT_FONT_SIZE, PRINT_LINE_WIDTH
    if plt.rcParams['text.usetex']:
        plt.rcParams['text.latex.preamble'] = PRINT_LATEX_PREAMBLE
    plt.rcParams.update({
        'font.family': PRINT_FONT_STACK,
        'mathtext.fontset': 'cm',
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
    for key in ('experiment', 'dt', 'reference', 'energy_time',
                'method_order', 'methods'):
        if key not in record:
            raise ValueError(f'{filename}: missing required key {key!r}')
    return record


def assign_styles(record):
    '''
    Line style, marker and colour for every method.

    Methods of the same class share a colour and a marker, so a
    first-order method and its symmetric composition are drawn alike
    except that first-order methods are dashed and second-order ones
    solid. The marker keeps the classes apart in black and white.

    Parameters
    ----------
    record : dict
        Record read from a data file.

    Returns
    -------
    dict
        Display name -> dict of matplotlib line properties.
    '''
    colours, markers, styles = {}, {}, {}
    for name in record['method_order']:
        cls = record['methods'][name]['class']
        colours.setdefault(cls, PALETTE[len(colours) % len(PALETTE)])
        markers.setdefault(cls, MARKERS.get(
            cls, FALLBACK_MARKERS[len(markers) % len(FALLBACK_MARKERS)]))
        styles[name] = {
            'color': colours[cls],
            'linestyle': '--' if record['methods'][name]['order'] == 1
            else '-',
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
        the label cannot be read as labelling a curve. Default 'below'.

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


def plot(record, panels, filename, layout='screen'):
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
    layout : {'screen', 'print'}, optional
        'screen' (default) puts the panels side by side under a title;
        'print' stacks them at the journal width, untitled, with the
        legend below, since the caption belongs to the paper.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, already saved to filename.
    '''
    styles = assign_styles(record)
    t = np.asarray(record['energy_time'])
    momentum = has_momentum(panels)
    n_panels = 3 if momentum else 2
    if layout == 'print':
        fig, axes = plt.subplots(n_panels, 1,
                                 figsize=(PRINT_WIDTH, 2.1 * n_panels + 0.6))
    else:
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(16 if momentum else 11, 4.4))
    ax_e, ax_c = axes[0], axes[-1]
    ax_m = axes[1] if momentum else None

    for name in record['method_order']:
        # The histories are told apart by colour, line style and level;
        # markers on traces this dense only add clutter, so they are
        # left to the convergence panel, whose points are the data.
        style = dict(styles[name], marker='None')
        ax_e.semilogy(t, np.maximum(panels[name]['energy'], FLOOR),
                      label=name, **style)
        if momentum:
            ax_m.semilogy(t, np.maximum(panels[name]['momentum'], FLOOR),
                          label=name, **style)
    # The long-run step is taken from the convergence study, so a power
    # of two is written as one, as on the step-size axis.
    h_run = record['parameters']['energy_h']
    k = np.log2(h_run)
    h_text = f'2^{{{k:.0f}}}' if k == round(k) else f'{h_run:g}'
    ax_e.set_xlabel('$t$')
    ax_e.set_ylabel(r'$|H(z_n) - H(z_0)| / |H(z_0)|$')
    ax_e.set_title(f'energy error, $h = {h_text}$')
    ax_e.set_ylim(bottom=FLOOR)
    if momentum:
        symbol = record.get('momentum_symbol', 'L')
        ax_m.set_xlabel('$t$')
        ax_m.set_ylabel(rf'$|{symbol}(z_n) - {symbol}(z_0)|$')
        ax_m.set_title(f'momentum error, $h = {h_text}$')
        ax_m.set_ylim(bottom=FLOOR)

    finest = {}
    for name in record['method_order']:
        p = panels[name]
        ax_c.loglog(p['h'], p['err'], label=name, **styles[name])
        order = record['methods'][name]['order']
        finest.setdefault(order, []).append(p['err'][-1])

    h = panels[record['method_order'][0]]['h']
    fine = h[len(h) // 2 - 1:]
    lowest = min(finest)
    guides = []
    for order, errs in sorted(finest.items()):
        above = order == lowest
        e0 = 2.0 * max(errs) if above else 0.5 * min(errs)
        guide = e0 * (fine / fine[-1]) ** order
        ax_c.loglog(fine, guide, color='0.5', linestyle=':')
        guides.append((rf'$O\left(h^{{{order}}}\right)$' if order > 1
                       else r'$O\left(h\right)$', guide,
                       'above' if above else 'below'))
    ax_c.set_xscale('log', base=2)
    # Three times matplotlib's default headroom, so the guide labels have
    # room above the upper guide and below the lower one.
    ax_c.margins(y=0.15)
    ax_c.set_xlabel('$h$')
    ax_c.set_ylabel(r'$\|q_N - q(T)\|_2$' if record['reference']['state']
                    else r'Estimated $\|q_N - q(T)\|_2$')
    ax_c.set_title(f'position error at $T = {record["parameters"]["T"]:.4g}$')

    if layout == 'print':
        # Untitled; the caption describes the panels.
        for ax in axes:
            ax.set_title('')
        # One legend under the stacked panels, where it covers no data.
        handles, labels = ax_c.get_legend_handles_labels()
        legend_height = 0.6
        fig.tight_layout(rect=(0, legend_height / fig.get_figheight(), 1, 1))
        fig.legend(handles, labels, loc='lower center', ncol=2,
                   frameon=False)
    else:
        ax_c.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(record['experiment'])
        fig.tight_layout()
    # The guides are labelled once the layout is final, where no line or
    # marker is in the way.
    renderer = fig.canvas.get_renderer()
    for text, guide, side in guides:
        place_label(ax_c, text, fine, guide, renderer, side)
    # Print figures keep their exact width; screen ones are cropped.
    fig.savefig(filename,
                bbox_inches=None if layout == 'print' else 'tight')
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
                             f'{FIGURE_SUFFIX}, or {PRINT_SUFFIX} with '
                             f'--print)')
    parser.add_argument('--print', action='store_true', dest='print_layout',
                        help='draw at the print size of the journal')
    args = parser.parse_args()

    use_latex()
    layout, suffix = 'screen', FIGURE_SUFFIX
    if args.print_layout:
        use_print_layout()
        layout, suffix = 'print', PRINT_SUFFIX

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
        panels = analyse(record)
        print_tables(record, panels)
        plt.close(plot(record, panels, output, layout))
    if failed:
        raise SystemExit(f'{failed} file(s) skipped')


if __name__ == '__main__':
    main()
