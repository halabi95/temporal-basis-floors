"""
Projection floors of harmonic-balance (HB), proper-orthogonal-decomposition
(POD) and dynamic-mode-decomposition (DMD) temporal bases, and the phase-slip
model of a misplaced fundamental, for the paper

  H. El Halabi and Y. Takahashi, "How Accurately Must the Frequency Be Known?
  Temporal Basis Selection for Reduced-Order Reconstruction of Periodic
  Fluid-Structure Interaction", Turbomachinery (ISROMAC19 special issue).

Definitions (equation numbers refer to the paper)
  floor(X, Psi)   Eq. (6)  ||X - Psi Psi^+ X||_F / ||X||_F, in percent
  opt_floor(X, K) Eq. (7)  floor of [1, U_1..U_{K-1}], POD of X itself
  hb(t, f, NH)    Eq. (3)  constant plus NH harmonic pairs of f
  dmd_freqs       Eq. (5)  exact DMD of the time-centred joint (u, v, p) state,
                           on-circle 0.9 < |lambda| < 1.1, ranked by amplitude
  refine          f0*: argmin_f floor(surface pressure, HB at N_H = 3)
  predicted       Eq. (9)  energy-weighted phase-slip model of the floor

Outputs results/floors.json with the data behind Figs. 2, 3 and 4.

Usage:  python src/floors.py --data-dir data --out results/floors.json
The reference .npz files are described in README.md.
"""

import argparse
import json
import os
import numpy as np

DATA_DIR = 'data'
OUT = 'results/floors.json'
KS = [3, 5, 7, 9, 11, 13]
ERRS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00]   # percent


# ------------------------------------------------------------------ primitives
def floor(X, Psi):
    A, *_ = np.linalg.lstsq(Psi, X, rcond=None)
    return float(np.linalg.norm(Psi @ A - X) / np.linalg.norm(X) * 100)


def hb(t, f, NH):
    cols = [np.ones_like(t)]
    for j in range(1, NH + 1):
        cols += [np.cos(2 * np.pi * j * f * t), np.sin(2 * np.pi * j * f * t)]
    return np.column_stack(cols)


def opt_floor(X, K):
    Xc = X - X.mean(0, keepdims=True)
    U, _, _ = np.linalg.svd(Xc, full_matrices=False)
    return floor(X, np.column_stack([np.ones(X.shape[0]), U[:, :K - 1]]))


def dmd_freqs(state, dt, npairs):
    """state: (N_t, M), time-centred. Returns up to npairs frequencies (Hz)."""
    Y = state.T
    Y1, Y2 = Y[:, :-1], Y[:, 1:]
    U, S, Vt = np.linalg.svd(Y1, full_matrices=False)
    r = min(int((S > 1e-10 * S[0]).sum()), 2 * npairs + 10)
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    At = Ur.T @ Y2 @ Vr @ np.diag(1 / Sr)
    lam, W = np.linalg.eig(At)
    Phi = Y2 @ Vr @ np.diag(1 / Sr) @ W
    b = np.linalg.lstsq(Phi, Y[:, 0], rcond=None)[0]
    om = np.angle(lam) / dt
    amp = np.abs(b) * np.linalg.norm(Phi, axis=0)
    sel = np.where((np.abs(lam) > 0.9) & (np.abs(lam) < 1.1) & (om > 1e-6))[0]
    sel = sel[np.argsort(amp[sel])[::-1]][:npairs]
    return sorted(om[sel] / (2 * np.pi))


def dmd_basis(t, freqs):
    cols = [np.ones_like(t)]
    for f in freqs:
        cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    return np.column_stack(cols)


def fft_peak(sig, dt):
    s = sig - sig.mean()
    mag = np.abs(np.fft.rfft(s))
    fr = np.fft.rfftfreq(len(s), d=dt)
    return float(fr[int(np.argmax(mag[1:]) + 1)])


def refine(X, t, f_guess, NH=3, span=0.05, rounds=4, n=401):
    lo, hi, best = f_guess * (1 - span), f_guess * (1 + span), f_guess
    for _ in range(rounds):
        g = np.linspace(lo, hi, n)
        vals = [floor(X, hb(t, x, NH)) for x in g]
        best = float(g[int(np.argmin(vals))])
        half = (hi - lo) / (n - 1) * 4
        lo, hi = best - half, best + half
    return best


def energy_fractions(X, t, f, NHfit=8):
    """Mean, per-harmonic and residual energy fractions of X at fundamental f."""
    Psi = hb(t, f, NHfit)
    A, *_ = np.linalg.lstsq(Psi, X, rcond=None)
    tot = np.linalg.norm(X) ** 2
    w = {0: float(np.linalg.norm(np.outer(Psi[:, 0], A[0])) ** 2 / tot)}
    for k in range(1, NHfit + 1):
        part = Psi[:, 2 * k - 1:2 * k + 1] @ A[2 * k - 1:2 * k + 1]
        w[k] = float(np.linalg.norm(part) ** 2 / tot)
    w['rest'] = float(np.linalg.norm(Psi @ A - X) ** 2 / tot)
    return w


def predicted(w, N, d, NH=3):
    s = lambda x: 1.0 if x == 0 else np.sin(x) / x
    e2 = sum(w[k] * (1 - s(np.pi * k * N * d) ** 2) for k in range(1, NH + 1))
    e2 += sum(w[k] for k in w if isinstance(k, int) and k > NH) + w['rest']
    return float(np.sqrt(max(e2, 0.0)) * 100)


# ------------------------------------------------------------------ loaders
def load_fsi(case):
    r = np.load(os.path.join(DATA_DIR, f'turek_hron_{case}_reference_dense.npz'))
    c, topo, mk = r['coordinates'], r['topology'], r['domain_markers']
    solid = np.zeros(c.shape[0], bool)
    fluid = np.zeros(c.shape[0], bool)
    for e in range(topo.shape[0]):
        (solid if mk[e] == 2 else fluid)[topo[e]] = True
    fsi = np.where(solid & fluid)[0]
    fl = np.where(fluid & ~solid)[0]
    U = {'fsi2': 1.0, 'fsi3': 2.0}[case]
    times = r['times']
    per = np.where(times >= 0.5 * times[-1])[0]
    t = times[per]
    vel = r['velocity'][per] / U
    pres = r['pressure'][per] / (1000.0 * U * U)
    tip = fsi[np.argmin(np.linalg.norm(c[fsi] - np.array([0.6, 0.2]), axis=1))]
    fc = c[fsi]
    x0, x1, y0, y1 = fc[:, 0].min(), fc[:, 0].max(), fc[:, 1].min(), fc[:, 1].max()
    near = ((c[:, 0] > x0 - .05) & (c[:, 0] < x1 + .05) &
            (c[:, 1] > y0 - .05) & (c[:, 1] < y1 + .05))
    isfl = np.zeros(c.shape[0], bool)
    isfl[fl] = True
    regions = {'interface': fsi, 'near-body': np.where(near & isfl)[0],
               'wake': np.where((c[:, 0] > x1) & isfl)[0], 'all fluid': fl}
    state = np.hstack([vel[:, :, 0], vel[:, :, 1], pres])          # all nodes
    state = state - state.mean(0, keepdims=True)
    return dict(t=t, dt=float(np.diff(times).mean()), field_p=pres[:, fl],
                surf_p=pres[:, fsi], surf_u=vel[:, fsi, 0], surf_v=vel[:, fsi, 1],
                regions={k: pres[:, v] for k, v in regions.items()},
                nreg={k: int(len(v)) for k, v in regions.items()},
                state=state, peak_sig=r['displacement'][per, tip, 1])


def load_cyl():
    fn = os.path.join(DATA_DIR, 'cylinder_re100_vortex_reference.npz')
    if not os.path.exists(fn):
        return None
    d = np.load(fn)
    t = d['times'].astype(float)
    ok = (np.isfinite(d['U']).all(0) & np.isfinite(d['V']).all(0)
          & np.isfinite(d['P']).all(0))
    u, v, p = d['U'][:, ok], d['V'][:, ok], d['P'][:, ok]
    ps = d['P_surf'][:, np.isfinite(d['P_surf']).all(0)]
    th = d['theta_surf'][np.isfinite(d['P_surf']).all(0)]
    lift = -(ps * np.sin(th)).sum(1)
    state = np.hstack([u, v, p])
    state = state - state.mean(0, keepdims=True)
    return dict(t=t - t[0], dt=float(np.diff(t).mean()), field_p=p, surf_p=ps,
                state=state, peak_sig=lift)


# ------------------------------------------------------------------ analysis
def analyse(name, d, regions=True):
    t, dt = d['t'], d['dt']
    f_pk = fft_peak(d['peak_sig'], dt)
    f_st = refine(d['surf_p'], t, f_pk)
    N = (t[-1] - t[0]) * f_st
    binw = 1.0 / (t[-1] - t[0])
    out = dict(f_peak=f_pk, f_refined=f_st, N=N, bin=binw,
               eps_peak=abs(f_pk - f_st) / f_st * 100,
               bins_off=abs(f_pk - f_st) / binw, worst=100 / (2 * N))
    print(f'\n=== {name.upper()} ===  f_pk {f_pk:.5f}  f0* {f_st:.5f} Hz  '
          f'N {N:.2f}  peak error {out["eps_peak"]:.3f} % '
          f'({out["bins_off"]:.3f} bins)  worst 1/(2N) {out["worst"]:.3f} %')

    # fig 2 / fig 3(b): pressure on all fluid points
    P = d['field_p']
    f2 = {'K': KS, 'HB_ref': [], 'HB_peak': [], 'DMD': [], 'DMD_freqs': [], 'POD_opt': []}
    for K in KS:
        NH = (K - 1) // 2
        fr = dmd_freqs(d['state'], dt, NH)
        f2['HB_ref'].append(floor(P, hb(t, f_st, NH)))
        f2['HB_peak'].append(floor(P, hb(t, f_pk, NH)))
        f2['DMD'].append(floor(P, dmd_basis(t, fr)) if len(fr) == NH else None)
        f2['DMD_freqs'].append(fr)
        f2['POD_opt'].append(opt_floor(P, K))
    out['fig2'] = f2
    print('  K   HB f0*  HB peak     DMD   POD*  (pressure, all fluid)')
    for i, K in enumerate(KS):
        dm = f2['DMD'][i]
        print(f'  {K:2d} {f2["HB_ref"][i]:7.3f} {f2["HB_peak"][i]:8.3f} '
              f'{"   n/a" if dm is None else f"{dm:7.3f}"} {f2["POD_opt"][i]:6.3f}'
              + ('' if dm is not None else f'   (DMD found {len(f2["DMD_freqs"][i])} pairs)'))

    # fig 3(a): surface pressure at N_H = 3 against frequency error
    S = d['surf_p']
    out['fig3a'] = dict(err=ERRS, floor=[floor(S, hb(t, f_st * (1 + e / 100), 3)) for e in ERRS],
                        peak_floor=floor(S, hb(t, f_pk, 3)))

    # fig 4: optimal pressure floors by region
    if regions:
        out['fig4'] = {k: [opt_floor(X, K) for K in KS] for k, X in d['regions'].items()
                       if X.shape[1] > 0 and np.linalg.norm(X) > 0}
        out['nreg'] = d['nreg']
        print('  optimal pressure floor by region, K = ' + ', '.join(map(str, KS)))
        for k, v in out['fig4'].items():
            print(f'    {k:10s} ' + ' '.join(f'{x:7.3f}' for x in v))

    # mechanism: energy-weighted single-tone prediction against measurement
    w = energy_fractions(S, t, f_st)
    out['energy'] = {str(k): v for k, v in w.items()}
    out['mech'] = [dict(err=e, measured=out['fig3a']['floor'][i],
                        predicted=predicted(w, N, e / 100)) for i, e in enumerate(ERRS)]
    print(f'  surface-pressure energy: mean {w[0]*100:.1f} %, '
          + ', '.join(f'h{k} {w[k]*100:.2f} %' for k in range(1, 5))
          + f', rest {w["rest"]*100:.3f} %')
    print('  err %   measured  predicted')
    for m in out['mech']:
        print(f'  {m["err"]:5.2f}   {m["measured"]:8.3f}  {m["predicted"]:9.3f}')
    return out


def main():
    res = {}
    for case in ('fsi2', 'fsi3'):
        d = load_fsi(case)
        res[case] = analyse(case, d)
        if case == 'fsi3':
            # the trained-run check: HB at the tip-refined 5.3885 Hz vs f0*
            f_tip = 5.3885
            t = d['t']
            fst = res[case]['f_refined']
            Nn = (t[-1] - t[0]) * fst
            dd = abs(fst - f_tip) / fst
            tone = predicted({0: 0.0, 1: 1.0, 'rest': 0.0}, Nn, dd, NH=1)
            res[case]['tip_check'] = dict(
                f_tip=f_tip, delta=dd * 100, single_tone_prediction=tone,
                v_floor_tip=floor(d['surf_v'], hb(t, f_tip, 6)),
                v_floor_ref=floor(d['surf_v'], hb(t, fst, 6)))
            tc = res[case]['tip_check']
            print(f'  FSI3 interface v, NH=6: at 5.3885 Hz {tc["v_floor_tip"]:.3f} %, '
                  f'at f0* {tc["v_floor_ref"]:.3f} %;  single-tone prediction '
                  f'{tone:.3f} % for delta {tc["delta"]:.4f} %')
    cyl = load_cyl()
    if cyl is not None:
        res['cylinder'] = analyse('cylinder', cyl, regions=False)
    else:
        print('\n(cylinder reference not found in this directory; skipped)')
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data-dir', default=DATA_DIR, help='folder with the reference .npz files')
    ap.add_argument('--out', default=OUT, help='output JSON')
    a = ap.parse_args()
    DATA_DIR, OUT = a.data_dir, a.out
    main()
