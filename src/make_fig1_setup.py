"""
Reproduce Fig. 1 of the paper: (a) the four overlapping subdomains of the
reconstruction network and (b) the sparse measurement points.

The points are drawn exactly as in training: of a nominal n = 1000 points, a
fraction 0.3 is requested on the fluid-structure interface (only 53 interface
nodes exist, so all 53 are taken) and the remaining 700 are drawn from the
interior fluid nodes with the given seed, giving the 753 points of the paper.

Needs the reference mesh in data/ (see README.md). Without it, the script draws
the same layout with stand-in points, marked LAYOUT PREVIEW.

Usage:  python src/make_fig1_setup.py [--case fsi3] [--seed 42] [--data-dir data] [--out figures/fig1_setup]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
    'legend.fontsize': 7.8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

CX, CY, R = 0.2, 0.2, 0.05
BEAM = (0.2489, 0.6, 0.19, 0.21)          # x0, x1, y0, y1
XMAX, YMAX = 2.5, 0.41
# Subdomain colours are chosen to appear in no other figure: Figs. 2 to 4 use
# blue, green and vermillion for the bases, black, grey and purple for the
# cases, and blue shades for the regions.
SD = [  # label, x0, x1, y0, y1, colour, linestyle
    (r'$\Omega_1$ upstream',   0.0,    0.5745, 0.0,  0.41, '#6B7A0F', '-'),   # olive
    (r'$\Omega_2$ upper wake', 0.2745, 2.5,    0.05, 0.41, '#D7191C', '-'),   # red
    (r'$\Omega_3$ lower wake', 0.2745, 2.5,    0.0,  0.35, '#E6AB02', '-'),   # gold
    (r'$\Omega_4$ near body',  0.05,   0.85,   0.0,  0.41, '#CE1F8E', '--'),  # magenta
]
BODY = '0.78'


def real_inputs(case, seed, n, frac, data_dir='data'):
    r = np.load(os.path.join(data_dir, f'turek_hron_{case}_reference_dense.npz'))
    c, t, m = r['coordinates'], r['topology'], r['domain_markers']
    solid = np.zeros(c.shape[0], bool)
    fluid = np.zeros(c.shape[0], bool)
    for e in range(t.shape[0]):
        k = t[e]
        (solid if m[e] == 2 else fluid)[k] = True
    fsi = np.where(solid & fluid)[0]
    fl = np.where(fluid & ~solid)[0]
    rng = np.random.RandomState(seed)             # same draw order as training
    n_fsi = int(frac * n)
    pi = rng.choice(fl, min(n - n_fsi, len(fl)), replace=False)
    pf = rng.choice(fsi, min(n_fsi, len(fsi)), replace=False)
    return c[pi], c[pf], False


def preview_inputs():
    rng = np.random.RandomState(0)
    pts = []
    while len(pts) < 700:                         # denser near the body, like the mesh
        x, y = rng.uniform(0, XMAX), rng.uniform(0, YMAX)
        d = np.hypot(x - CX, y - CY)
        if d < R or (BEAM[0] <= x <= BEAM[1] and BEAM[2] <= y <= BEAM[3]):
            continue
        if rng.rand() < min(1.0, 0.08 + 0.9 * np.exp(-max(d - R, 0) / 0.12)):
            pts.append((x, y))
    xs = np.linspace(BEAM[0], BEAM[1], 25)
    iface = [(x, BEAM[3]) for x in xs] + [(x, BEAM[2]) for x in xs] + \
            [(BEAM[1], y) for y in np.linspace(BEAM[2], BEAM[3], 3)]
    return np.array(pts), np.array(iface), True


def body(ax):
    ax.add_patch(Circle((CX, CY), R, color=BODY, zorder=4, lw=0))
    ax.add_patch(Rectangle((BEAM[0], BEAM[2]), BEAM[1] - BEAM[0], BEAM[3] - BEAM[2],
                           color=BODY, zorder=4, lw=0))
    ax.add_patch(Rectangle((0, 0), XMAX, YMAX, fill=False, ec='k', lw=0.9, zorder=5))


def frame(ax):
    ax.set_xlim(-0.02, XMAX + 0.02)
    ax.set_ylim(-0.02, YMAX + 0.02)
    ax.set_aspect('equal')
    ax.set_yticks([0, 0.2, 0.4])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def plot(p_int, p_if, preview, out):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 3.35))
    fig.subplots_adjust(hspace=0.62)

    # (a) decomposition
    body(a1)
    for i, (lab, x0, x1, y0, y1, col, ls) in enumerate(SD):
        ins = 0.006 * (i + 1)                     # separate coincident edges
        a1.add_patch(Rectangle((x0 + ins, y0 + ins), (x1 - x0) - 2 * ins,
                               (y1 - y0) - 2 * ins, fill=False, ec=col,
                               lw=1.5, ls=ls, zorder=3))
    frame(a1)
    a1.set_title('(a)  Domain decomposition into four overlapping subdomains',
                 loc='left', fontsize=9)
    a1.set_ylabel('y (m)')
    handles = [Line2D([], [], color=c, ls=ls, lw=1.5, label=l)
               for l, _, _, _, _, c, ls in SD]
    a1.legend(handles=handles, ncol=4, frameon=False, loc='lower center',
              bbox_to_anchor=(0.5, -0.52), handlelength=2.2, columnspacing=1.6)

    # (b) sampling
    body(a2)
    a2.scatter(p_int[:, 0], p_int[:, 1], s=2.2, color='0.40', lw=0, zorder=2,
               label=f'fluid interior ({len(p_int)})')
    a2.scatter(p_if[:, 0], p_if[:, 1], s=10, color='k', ec='white', lw=0.3,
               zorder=6, label=f'fluid-structure interface ({len(p_if)})')
    frame(a2)
    a2.set_title(f'(b)  Sparse measurement locations, {len(p_int) + len(p_if)} points',
                 loc='left', fontsize=9)
    a2.set_xlabel('x (m)')
    a2.set_ylabel('y (m)')
    a2.legend(frameon=False, loc='upper right', markerscale=1.8,
              bbox_to_anchor=(1.0, 1.02))

    if preview:
        fig.text(0.5, 0.5, 'LAYOUT PREVIEW, stand-in points', ha='center',
                 va='center', fontsize=22, color='red', alpha=0.18, rotation=12)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300)
    print(f'wrote {out}.pdf / .png'
          + ('   [preview: reference data not found]' if preview else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='fsi3')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--frac', type=float, default=0.3)
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out', default='figures/fig1_setup')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    if os.path.exists(os.path.join(a.data_dir, f'turek_hron_{a.case}_reference_dense.npz')):
        p_int, p_if, prev = real_inputs(a.case, a.seed, a.n, a.frac, a.data_dir)
    else:
        p_int, p_if, prev = preview_inputs()
    plot(p_int, p_if, prev, a.out)


if __name__ == '__main__':
    main()
