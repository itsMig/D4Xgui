"""Faster D47crunch helpers for pooled+drift standardization in the GUI.

D47crunch's pooled ``standardize`` residual is a pure-Python loop over every
replicate on every lmfit evaluation, and ``plot_single_session(fig=None)`` still
builds a 50×50 standardization-error mesh via nested Python calls (plus
matplotlib side effects). Both keep drift + pooled semantics; only the
numerics/IO path is sped up.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

import D47crunch as D47c


def fast_pooled_standardize(
    data: Any,
    *,
    consolidate: bool = True,
    consolidate_tables: bool = False,
    consolidate_plots: bool = False,
    constraints: Optional[dict] = None,
) -> Any:
    """Pooled standardization with vectorized residuals (drift params preserved).

    Drop-in replacement for ``data.standardize(method='pooled', ...)`` that
    keeps scrambling/slope/wg drift flags and returns the same lmfit result
    object attached as ``data.standardization``.
    """
    if constraints is None:
        constraints = {}

    mz = data._4x
    data.standardization_method = "pooled"
    data.assign_timestamps()

    for r in data:
        r[f"wD{mz}raw"] = 1.0

    params = D47c.Parameters()
    session_list: List[str] = list(data.sessions.keys())
    session_pf = [D47c.pf(s) for s in session_list]

    for session, s_pf in zip(session_list, session_pf):
        sess = data.sessions[session]
        params.add(f"a_{s_pf}", value=0.9)
        params.add(f"b_{s_pf}", value=0.0)
        params.add(f"c_{s_pf}", value=-0.9)
        params.add(f"a2_{s_pf}", value=0.0)
        params.add(f"b2_{s_pf}", value=0.0)
        params.add(f"c2_{s_pf}", value=0.0)
        if not sess["scrambling_drift"]:
            params[f"a2_{s_pf}"].expr = "0"
        if not sess["slope_drift"]:
            params[f"b2_{s_pf}"].expr = "0"
        if not sess["wg_drift"]:
            params[f"c2_{s_pf}"].expr = "0"

    unknown_list: List[str] = list(data.unknowns)
    unknown_pf = [D47c.pf(s) for s in unknown_list]
    for u_pf in unknown_pf:
        params.add(f"D{mz}_{u_pf}", value=0.5)

    for k, expr in constraints.items():
        params[k].expr = expr

    # --- Precompute replicate arrays (avoids Python loop in residuals) ---
    sess_index = {s_pf: i for i, s_pf in enumerate(session_pf)}
    unk_index = {u_pf: i for i, u_pf in enumerate(unknown_pf)}

    n = len(data)
    sess_idx = np.empty(n, dtype=np.intp)
    is_anchor = np.empty(n, dtype=bool)
    D_nom = np.zeros(n, dtype=float)
    unk_idx = np.full(n, -1, dtype=np.intp)
    D_raw = np.empty(n, dtype=float)
    d_val = np.empty(n, dtype=float)
    t_val = np.empty(n, dtype=float)
    w_val = np.empty(n, dtype=float)

    nomin = data.Nominal_D4x
    for i, r in enumerate(data):
        s_pf = D47c.pf(r["Session"])
        samp = r["Sample"]
        sess_idx[i] = sess_index[s_pf]
        anchor = samp in nomin
        is_anchor[i] = anchor
        if anchor:
            D_nom[i] = float(nomin[samp])
        else:
            unk_idx[i] = unk_index[D47c.pf(samp)]
        D_raw[i] = float(r[f"D{mz}raw"])
        d_val[i] = float(r[f"d{mz}"])
        t_val[i] = float(r["t"])
        w_val[i] = float(r[f"wD{mz}raw"])

    n_sess = len(session_list)
    n_unk = len(unknown_list)
    a_names = [f"a_{s}" for s in session_pf]
    b_names = [f"b_{s}" for s in session_pf]
    c_names = [f"c_{s}" for s in session_pf]
    a2_names = [f"a2_{s}" for s in session_pf]
    b2_names = [f"b2_{s}" for s in session_pf]
    c2_names = [f"c2_{s}" for s in session_pf]
    D_names = [f"D{mz}_{u}" for u in unknown_pf]

    def residuals(p):
        pv = p.valuesdict()
        a = np.fromiter((pv[n] for n in a_names), dtype=float, count=n_sess)
        b = np.fromiter((pv[n] for n in b_names), dtype=float, count=n_sess)
        c = np.fromiter((pv[n] for n in c_names), dtype=float, count=n_sess)
        a2 = np.fromiter((pv[n] for n in a2_names), dtype=float, count=n_sess)
        b2 = np.fromiter((pv[n] for n in b2_names), dtype=float, count=n_sess)
        c2 = np.fromiter((pv[n] for n in c2_names), dtype=float, count=n_sess)

        D_true = D_nom.copy()
        if n_unk:
            D_unk = np.fromiter((pv[n] for n in D_names), dtype=float, count=n_unk)
            mask = ~is_anchor
            D_true[mask] = D_unk[unk_idx[mask]]

        a_r, b_r, c_r = a[sess_idx], b[sess_idx], c[sess_idx]
        a2_r, b2_r, c2_r = a2[sess_idx], b2[sess_idx], c2[sess_idx]
        model = (
            a_r * D_true
            + b_r * d_val
            + c_r
            + t_val * (a2_r * D_true + b2_r * d_val + c2_r)
        )
        return (D_raw - model) / w_val

    result = D47c.Minimizer(residuals, params).least_squares()
    data.Nf = result.nfree
    data.t95 = D47c.tstudent.ppf(1 - 0.05 / 2, data.Nf)
    new_names, new_covar, _new_se = D47c._fullcovar(result)[:3]
    result.var_names = new_names
    result.covar = new_covar

    pv = result.params.valuesdict()
    for r in data:
        s = D47c.pf(r["Session"])
        a, b, c = pv[f"a_{s}"], pv[f"b_{s}"], pv[f"c_{s}"]
        a2, b2, c2 = pv[f"a2_{s}"], pv[f"b2_{s}"], pv[f"c2_{s}"]
        r[f"D{mz}"] = (
            r[f"D{mz}raw"]
            - c
            - b * r[f"d{mz}"]
            - c2 * r["t"]
            - b2 * r["t"] * r[f"d{mz}"]
        ) / (a + a2 * r["t"])

    data.standardization = result

    for session in data.sessions:
        data.sessions[session]["Np"] = 3
        for k in ("scrambling", "slope", "wg"):
            if data.sessions[session][f"{k}_drift"]:
                data.sessions[session]["Np"] += 1

    if consolidate:
        data.consolidate(tables=consolidate_tables, plots=consolidate_plots)
    return result


def _vectorized_std_error_grid(
    data: Any,
    session: str,
    d_grid: np.ndarray,
    D_grid: np.ndarray,
    t: float = 0.0,
) -> np.ndarray:
    """Return SI[i,j] = standardization_error(session, d=d_grid[j], D=D_grid[i])."""
    s = data.sessions[session]
    a, b, c = s["a"], s["b"], s["c"]
    a2, b2, c2 = s["a2"], s["b2"], s["c2"]
    CM = np.asarray(s["CM"], dtype=float)

    # mesh: rows follow D (yi), cols follow d (xi) — matches original meshgrid
    d = d_grid[np.newaxis, :]  # (1, nx)
    D = D_grid[:, np.newaxis]  # (ny, 1)
    # Broadcast to full grid; original uses x,y = D4x, d4x
    x, y = np.broadcast_arrays(D, d)
    denom = a + a2 * t
    dxda = -x / denom
    dxdb = -y / denom
    dxdc = np.full_like(x, -1.0 / denom)
    dxda2 = -x * a2 / denom
    dxdb2 = -y * t / denom
    dxdc2 = np.full_like(x, -t / denom)

    # V shape (6, ny, nx); sx = sqrt(V^T CM V) per grid point
    V = np.stack([dxda, dxdb, dxdc, dxda2, dxdb2, dxdc2], axis=0)
    Vt = np.moveaxis(V, 0, -1)  # (ny, nx, 6)
    CMv = np.einsum("ij,...j->...i", CM, Vt)
    sx2 = np.einsum("...i,...i->...", Vt, CMv)
    return np.sqrt(np.maximum(sx2, 0.0))


def _contour_levels(SI: np.ndarray, *, max_levels: int = 200) -> tuple[np.ndarray, float]:
    """Pick contour levels without allocating a huge ``np.arange``.

    Ill-conditioned session covariance (common for noisy Δ49 fits) can make
    standardization-error grids span 1e6–Inf. The original D47crunch logic used
    ``np.arange(SImin, SImax, 0.5)``, which then tries to allocate gigabytes and
    gets the Streamlit worker SIGKILL'd by the OS with no Python traceback.
    """
    finite = SI[np.isfinite(SI)]
    if finite.size == 0:
        return np.array([0.0]), 0.5

    si_min = float(np.min(finite))
    si_max = float(np.max(finite))
    rng = si_max - si_min

    if rng <= 0.01:
        cinterval = 0.001
    elif rng <= 0.03:
        cinterval = 0.004
    elif rng <= 0.1:
        cinterval = 0.01
    elif rng <= 0.3:
        cinterval = 0.03
    elif rng <= 1.0:
        cinterval = 0.1
    elif rng <= 50.0:
        cinterval = 0.5
    else:
        # Pathological range: fixed-count linspace instead of arange.
        cval = np.linspace(si_min, si_max, num=min(max_levels, 50))
        cinterval = float(cval[1] - cval[0]) if len(cval) > 1 else 0.5
        return cval, cinterval

    start = np.ceil(si_min / 0.001) * 0.001
    stop = np.ceil(si_max / 0.001 + 1) * 0.001
    n = int(np.ceil((stop - start) / cinterval)) if cinterval > 0 else 0
    if n <= 0 or not np.isfinite(n):
        return np.array([si_min if np.isfinite(si_min) else 0.0]), cinterval
    if n > max_levels:
        cval = np.linspace(start, stop, num=max_levels)
        cinterval = float(cval[1] - cval[0]) if len(cval) > 1 else cinterval
        return cval, cinterval
    return np.arange(start, stop, cinterval), cinterval


def fast_session_plot_data(
    data: Any,
    session: str,
    *,
    n_grid: int = 40,
) -> Dict[str, Any]:
    """Build the same dict as ``plot_single_session(..., fig=None)`` without matplotlib.

    Contour mesh uses vectorized standardization-error evaluation.
    Contour levels are capped so ill-conditioned Δ49 covariances cannot OOM.
    """
    mz = data._4x
    sdata = data.sessions[session]["data"]
    anchors = [
        a for a in data.anchors
        if any(r["Sample"] == a for r in sdata)
    ]
    unknowns = [
        u for u in data.unknowns
        if any(r["Sample"] == u for r in sdata)
    ]
    anchors_d = [r[f"d{mz}"] for r in sdata if r["Sample"] in data.anchors]
    anchors_D = [r[f"D{mz}"] for r in sdata if r["Sample"] in data.anchors]
    unknowns_d = [r[f"d{mz}"] for r in sdata if r["Sample"] in data.unknowns]
    unknowns_D = [r[f"D{mz}"] for r in sdata if r["Sample"] in data.unknowns]

    if anchors:
        anchor_avg = (
            np.array([
                np.array([
                    np.min([r[f"d{mz}"] for r in sdata if r["Sample"] == sample]) - 1,
                    np.max([r[f"d{mz}"] for r in sdata if r["Sample"] == sample]) + 1,
                ])
                for sample in anchors
            ]).T,
            np.array([
                np.array([0, 0]) + data.Nominal_D4x[sample]
                for sample in anchors
            ]).T,
        )
    else:
        anchor_avg = (np.zeros((2, 0)), np.zeros((2, 0)))

    if unknowns:
        unknown_avg = (
            np.array([
                np.array([
                    np.min([r[f"d{mz}"] for r in sdata if r["Sample"] == sample]) - 1,
                    np.max([r[f"d{mz}"] for r in sdata if r["Sample"] == sample]) + 1,
                ])
                for sample in unknowns
            ]).T,
            np.array([
                np.array([0, 0]) + data.unknowns[sample][f"D{mz}"]
                for sample in unknowns
            ]).T,
        )
    else:
        unknown_avg = (np.zeros((2, 0)), np.zeros((2, 0)))

    xs = anchors_d + unknowns_d
    ys = anchors_D + unknowns_D
    if not xs:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    x1, x2 = float(np.min(xs)), float(np.max(xs))
    y1, y2 = float(np.min(ys)), float(np.max(ys))
    # Guard against NaN/Inf extents from bad D49 replicates
    if not all(np.isfinite(v) for v in (x1, x2, y1, y2)):
        x1, x2, y1, y2 = 0.0, 1.0, 0.0, 1.0
    w, h = (x2 - x1) or 1.0, (y2 - y1) or 1.0
    x1 -= w / 20
    x2 += w / 20
    y1 -= h / 20
    y2 += h / 20

    xi = np.linspace(x1, x2, n_grid)
    yi = np.linspace(y1, y2, n_grid)
    XI, YI = np.meshgrid(xi, yi)
    try:
        SI = _vectorized_std_error_grid(data, session, xi, yi)
    except Exception:
        SI = np.zeros_like(XI)

    # Keep mesh finite for Plotly. Ill-conditioned Δ49 CM often yields Inf;
    # clamp those cells to the finite max so contours stay usable.
    finite = SI[np.isfinite(SI)]
    cap = float(np.max(finite)) if finite.size else 0.0
    SI = np.nan_to_num(SI, nan=cap, posinf=cap, neginf=0.0)
    cval, cinterval = _contour_levels(SI)
    contour = (XI, YI, SI, cval, cinterval)

    return {
        "anchors": anchors,
        "unknowns": unknowns,
        "anchors_d": anchors_d,
        "anchors_D": anchors_D,
        "unknowns_d": unknowns_d,
        "unknowns_D": unknowns_D,
        "anchor_avg": anchor_avg,
        "unknown_avg": unknown_avg,
        "contour": contour,
    }
