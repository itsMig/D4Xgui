#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotly geometry bridge for D95eq (no Streamlit imports)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import uncertainties as uc
from correldata import uarray


def ellipse_params_to_xy(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_deg: float,
    n: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parametric ellipse points for Plotly line traces."""
    t = np.linspace(0, 2 * np.pi, n)
    a = width / 2.0
    b = height / 2.0
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    x_local = a * np.cos(t)
    y_local = b * np.sin(t)
    x = cx + x_local * cos_a - y_local * sin_a
    y = cy + x_local * sin_a + y_local * cos_a
    return x, y


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(np.asarray(value).ravel()[0])
    return float(value)


def add_conf_ellipses(
    fig: go.Figure,
    ellipses: Sequence[Sequence[Any]],
    color: Union[str, Dict[str, str], None] = None,
    name: str = "95% confidence",
    showlegend: bool = False,
    legendgroup: Optional[str] = None,
    line_width: float = 1.0,
    opacity: float = 1.0,
    sample_names: Optional[Sequence[str]] = None,
    x_axis: str = "D47",
    y_axis: str = "D48",
) -> None:
    """Add one closed line trace per confidence ellipse."""
    for i, params in enumerate(ellipses):
        cx, cy, width, height, angle = params
        x_d47, y_d48 = ellipse_params_to_xy(
            _scalar(cx),
            _scalar(cy),
            _scalar(width),
            _scalar(height),
            _scalar(angle),
        )
        x, y = map_d47d48_to_axes(x_d47, y_d48, x_axis, y_axis)
        trace_color = color
        if isinstance(color, dict) and sample_names is not None:
            trace_color = color.get(sample_names[i], "grey")
        trace_legendgroup = legendgroup or name
        trace_showlegend = showlegend and i == 0
        if sample_names is not None:
            trace_legendgroup = sample_names[i]
            trace_showlegend = False
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=trace_color, width=line_width),
                opacity=opacity,
                name=name if i == 0 else None,
                legendgroup=trace_legendgroup,
                showlegend=trace_showlegend,
                hoverinfo="skip",
            )
        )


def add_confidence_band(
    fig: go.Figure,
    vertices: np.ndarray,
    x_axis: str,
    y_axis: str,
    name: str = "D95eq 95% confidence band",
    fillcolor: str = "rgba(0, 100, 0, 0.12)",
    line_color: str = "rgba(0, 100, 0, 0.35)",
    legendgroup: str = "d95eq_equilibrium",
) -> None:
    """Add filled polygon from ``plot_D95_confidence_band(plot=False)``."""
    d47 = vertices[:, 0]
    d48 = vertices[:, 1]
    x_vals, y_vals = map_d47d48_to_axes(d47, d48, x_axis, y_axis)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            fill="toself",
            fillcolor=fillcolor,
            line=dict(color=line_color, width=0.5),
            name=name,
            legendgroup=legendgroup,
            showlegend=True,
            hoverinfo="skip",
        )
    )


def near_calibration_marker_mask(
    temps: Union[np.ndarray, Sequence[float]],
    calibration_temps_c: Sequence[int],
    tolerance: float = 0.5,
) -> np.ndarray:
    """True where dense-grid °C is within ``tolerance`` of a calibration marker."""
    temps_arr = np.asarray(temps, dtype=float)
    cal = np.asarray(calibration_temps_c, dtype=float)
    return np.min(np.abs(temps_arr[:, None] - cal[None, :]), axis=1) <= tolerance


def add_equilibrium_curve(
    fig: go.Figure,
    data: Dict[str, Any],
    x_axis: str,
    y_axis: str,
    d47_key: str = "D47e",
    d48_key: str = "D48e",
    t_key: str = "Te",
    name: str = "D95eq carbonate equilibrium",
    line_color: str = "darkgreen",
    legendgroup: str = "d95eq_equilibrium",
    show_hover: bool = True,
    calibration_temps_c: Optional[Sequence[int]] = None,
) -> None:
    """Line trace from ``Engine.plot_D95_equilibrium`` data dict."""
    d47 = uarray_nominal(data[d47_key])
    d48 = uarray_nominal(data[d48_key])
    x_vals, y_vals = map_d47d48_to_axes(d47, d48, x_axis, y_axis)
    trace_kwargs: Dict[str, Any] = dict(
        x=x_vals,
        y=y_vals,
        mode="lines",
        name=name,
        line=dict(color=line_color, width=1.5, dash="dot"),
        legendgroup=legendgroup,
        showlegend=True,
    )
    if show_hover and t_key in data:
        temps = uarray_nominal(data[t_key])
        hover_labels = d95eq_calibration_hover_labels(temps, d47, d48)
        if calibration_temps_c is not None:
            near_marker = near_calibration_marker_mask(temps, calibration_temps_c)
            hover_templates = [
                "%{text}<extra></extra>" if not is_near else "<extra></extra>"
                for is_near in near_marker
            ]
            trace_kwargs["hovertemplate"] = hover_templates
        else:
            trace_kwargs["hovertemplate"] = "%{text}<extra></extra>"
        trace_kwargs["text"] = hover_labels
    else:
        trace_kwargs["hoverinfo"] = "skip"
    fig.add_trace(go.Scatter(**trace_kwargs))


def add_equilibrium_temperature_markers(
    fig: go.Figure,
    data: Dict[str, Any],
    x_axis: str,
    y_axis: str,
    legendgroup: str = "d95eq_equilibrium",
    marker_color: str = "darkgreen",
    temp_markers_c: Optional[Sequence[int]] = None,
) -> None:
    """Marker points and °C labels from ``plot_D95_equilibrium`` (Tm, D47m, D48m)."""
    if not {"Tm", "D47m", "D48m"}.issubset(data):
        return

    d47 = uarray_nominal(data["D47m"])
    d48 = uarray_nominal(data["D48m"])
    if temp_markers_c is not None:
        temps = np.asarray(temp_markers_c, dtype=int)
    else:
        temps = np.asarray(data["Tm"], dtype=float)
    x_vals, y_vals = map_d47d48_to_axes(d47, d48, x_axis, y_axis)
    labels = calibration_temp_labels(temps)
    hover_labels = d95eq_calibration_hover_labels(temps, d47, d48)

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            legendgroup=legendgroup,
            name="",
            marker=dict(color=marker_color),
            showlegend=False,
            text=hover_labels,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="text",
            legendgroup=legendgroup,
            name="",
            showlegend=False,
            text=labels,
            textposition="bottom right",
        )
    )


def resolve_se_columns(error_mode: str, mz: str) -> Tuple[str, float]:
    """
    Map ``error_dualClumped`` pattern to a column name and divisor for 1σ SE.

    Returns (column_name, divisor) where 1σ SE = df[column] / divisor.
    """
    col = error_mode.format(mz=mz)
    if "2SE" in col and "longterm" not in col.lower():
        return col, 2.0
    if "longterm" in col.lower():
        return col, 2.0
    return col, 1.0


def replicate_se_columns() -> Tuple[str, str]:
    """Default 1σ SE columns for replicate-level ellipses."""
    return "SE_D47", "SE_D48"


def df_to_d95eq_uarrays(
    df: pd.DataFrame,
    d47_col: str = "D47",
    d48_col: str = "D48",
    se_d47: str = "SE_D47",
    se_d48: str = "SE_D48",
    se_div_d47: float = 1.0,
    se_div_d48: float = 1.0,
    rho: float = 0.0,
    sample_rho: Optional[Dict[str, float]] = None,
    sample_col: str = "Sample",
) -> Tuple[uarray, uarray, pd.Index]:
    """
    Build ``correldata.uarray`` pairs from summary/replicate rows.

    Per-row correlation between Δ47 and Δ48 is resolved in this order:

    1. ``sample_rho[row[sample_col]]`` if ``sample_rho`` is provided and the
       sample key is present (used to inject sample-based replicate
       correlations).
    2. Otherwise the scalar ``rho`` fallback.

    The 2×2 covariance passed to ``uc.correlated_values`` is
    ``[[σ47², ρ·σ47·σ48], [ρ·σ47·σ48, σ48²]]``. ρ close to 0 short-circuits
    to two independent ufloats (axis-aligned ellipse).
    """
    required = [d47_col, d48_col, se_d47, se_d48]
    if sample_rho is not None and sample_col in df.columns:
        required = [sample_col, *required]
    valid = df[required].dropna(subset=[d47_col, d48_col, se_d47, se_d48])

    ufloats_d47: List[Any] = []
    ufloats_d48: List[Any] = []
    for _, row in valid.iterrows():
        if sample_rho is not None and sample_col in row:
            r = float(sample_rho.get(str(row[sample_col]), rho))
        else:
            r = float(rho)
        s47 = row[se_d47] / se_div_d47
        s48 = row[se_d48] / se_div_d48
        if abs(r) < 1e-12:
            ufloats_d47.append(uc.ufloat(row[d47_col], s47))
            ufloats_d48.append(uc.ufloat(row[d48_col], s48))
            continue
        r = max(-0.999, min(0.999, r))
        cov = [
            [s47 ** 2, r * s47 * s48],
            [r * s47 * s48, s48 ** 2],
        ]
        u47, u48 = uc.correlated_values([row[d47_col], row[d48_col]], cov)
        ufloats_d47.append(u47)
        ufloats_d48.append(u48)
    return uarray(ufloats_d47), uarray(ufloats_d48), valid.index


def map_d47d48_to_axes(
    d47: Union[np.ndarray, Sequence[float]],
    d48: Union[np.ndarray, Sequence[float]],
    x_axis: str,
    y_axis: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Swap coordinates when the plot has D48 on x and D47 on y."""
    d47_arr = np.asarray(d47, dtype=float)
    d48_arr = np.asarray(d48, dtype=float)
    if x_axis == "D47" and y_axis == "D48":
        return d47_arr, d48_arr
    if x_axis == "D48" and y_axis == "D47":
        return d48_arr, d47_arr
    raise ValueError(f"D95eq axes must be D47 and D48, got {x_axis} vs {y_axis}")


def map_d47d48_point_to_axes(
    d47: float,
    d48: float,
    x_axis: str,
    y_axis: str,
) -> Tuple[float, float]:
    x_vals, y_vals = map_d47d48_to_axes([d47], [d48], x_axis, y_axis)
    return float(x_vals[0]), float(y_vals[0])


def uarray_nominal(values: Any) -> np.ndarray:
    """Extract nominal values from uarray / ufloat sequences."""
    if hasattr(values, "n"):
        return np.asarray(values.n, dtype=float)
    return np.array([v.n if hasattr(v, "n") else float(v) for v in values], dtype=float)


def calibration_style(is_dark: bool) -> Dict[str, str]:
    """Theme-dependent colors for D95eq equilibrium curve and 95% band."""
    if is_dark:
        return {
            "line_color": "white",
            "marker_color": "white",
            "band_fillcolor": "rgba(120, 230, 120, 0.28)",
            "band_line_color": "rgba(140, 255, 140, 0.6)",
        }
    return {
        "line_color": "darkgreen",
        "marker_color": "darkgreen",
        "band_fillcolor": "rgba(0, 100, 0, 0.12)",
        "band_line_color": "rgba(0, 100, 0, 0.35)",
    }


def calibration_temp_label(t: Union[int, float]) -> str:
    """Format a calibration-line temperature (integer °C, no decimals)."""
    return f"{int(t)}°C"


def calibration_temp_labels(temps: Union[np.ndarray, Sequence[float]]) -> List[str]:
    """Format calibration-line temperature values as label strings."""
    return [calibration_temp_label(t) for t in np.asarray(temps)]


def format_temp_label(t: float) -> str:
    """Format a single temperature value."""
    return calibration_temp_label(t) if abs(t - round(t)) < 1e-6 else f"{t:.1f}°C"


def format_temp_labels(temps: Union[np.ndarray, Sequence[float]]) -> List[str]:
    """Format temperature values as hover/label strings."""
    return [format_temp_label(t) for t in np.asarray(temps, dtype=float)]


def d95eq_calibration_hover_labels(
    temps: Union[np.ndarray, Sequence[float]],
    d47: Union[np.ndarray, Sequence[float]],
    d48: Union[np.ndarray, Sequence[float]],
) -> List[str]:
    """Hover text with temperature and Δ47/Δ48 for D95eq calibration points."""
    temps_arr = np.asarray(temps, dtype=float)
    d47_arr = np.asarray(d47, dtype=float)
    d48_arr = np.asarray(d48, dtype=float)
    return [
        f"T = {format_temp_label(t)}<br>Δ47 = {d47_arr[i]:.6f} ‰<br>Δ48 = {d48_arr[i]:.6f} ‰"
        for i, t in enumerate(temps_arr)
    ]


def axis_calibration_hover_labels(
    temps: Union[np.ndarray, Sequence[float]],
    x_vals: Union[np.ndarray, Sequence[float]],
    y_vals: Union[np.ndarray, Sequence[float]],
    x_axis: str,
    y_axis: str,
) -> List[str]:
    """Hover text with temperature and axis Δ values for generic calibration curves."""
    temps_arr = np.asarray(temps, dtype=float)
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    x_n = x_axis.replace("D", "")
    y_n = y_axis.replace("D", "")
    return [
        f"T = {format_temp_label(t)}<br>Δ{x_n} = {x_arr[i]:.3f} ‰<br>Δ{y_n} = {y_arr[i]:.3f} ‰"
        for i, t in enumerate(temps_arr)
    ]


def uarray_std(values: Any) -> np.ndarray:
    if hasattr(values, "s"):
        return np.asarray(values.s, dtype=float)
    return np.array([v.s if hasattr(v, "s") else 0.0 for v in values], dtype=float)


def sample_color_map(fig: go.Figure) -> Dict[str, str]:
    """Map scatter trace names to marker colors."""
    colors: Dict[str, str] = {}
    for trace in fig.data:
        if trace.mode and "markers" in trace.mode and trace.name:
            marker_color = trace.marker.color if trace.marker else None
            if isinstance(marker_color, str):
                colors[trace.name] = marker_color
    return colors


def compute_conf_ellipses(
    d47_u: uarray,
    d48_u: uarray,
    p: float = 0.95,
) -> Tuple[Any, ...]:
    """Call ``D95eq.conf_ellipse`` with ``plot=False`` (X=D47, Y=D48)."""
    from D95eq import conf_ellipse

    return conf_ellipse(d47_u, d48_u, p=p, plot=False)


# Fiebig et al. (2024) Hill×affine fit covariances (scaling, offset order).
_FIEBIG2024_D47_SCALING_OFFSET_PCOV = (
    (7.06720078e-05, -1.21156975e-05),
    (-1.21156975e-05, 4.10228295e-06),
)
_FIEBIG2024_D48_SCALING_OFFSET_PCOV = (
    (1.46002771e-03, -6.90467053e-05),
    (-6.90467053e-05, 7.39697438e-06),
)


def hill_affine_to_d95eq_uarray(
    hill_coeffs: Tuple[float, ...],
    scaling: float,
    offset: float,
    scaling_offset_pcov: Tuple[Tuple[float, float], Tuple[float, float]],
) -> uarray:
    """
    Map Hill(2014) × (scaling, offset) to D95eq 5-term 1/T polynomial ``uarray``.

    ``scaling_offset_pcov`` is the 2×2 covariance of (scaling, offset) from the
    original affine regression (Fiebig2024 CDES90 fit).
    """
    pcov = np.asarray(scaling_offset_pcov, dtype=float)
    offset_u, scaling_u = uc.correlated_values(
        [offset, scaling],
        [[pcov[1, 1], pcov[0, 1]], [pcov[0, 1], pcov[0, 0]]],
    )
    a1, a2, a3, a4 = hill_coeffs
    return uarray([offset_u, scaling_u * a1, scaling_u * a2, scaling_u * a3, scaling_u * a4])


def fiebig2024_engine_coefs() -> Tuple[uarray, uarray]:
    """
    Build D95eq ``Engine`` calibration coefficients from ``TemperatureCalculator`` Fiebig2024.

    Uses published rounded Hill×affine parameters (1.038, 0.1848 / 0.1214) with
    uncertainties from the original Fiebig2024 CDES90 affine regressions.
    """
    from tools.calc_temperature import TemperatureCalculator as TC

    d47_coefs = hill_affine_to_d95eq_uarray(
        TC.POLY_63_COEFFS,
        TC.FIEBIG2024_D47_SCALING,
        TC.FIEBIG2024_D47_OFFSET,
        _FIEBIG2024_D47_SCALING_OFFSET_PCOV,
    )
    d48_coefs = hill_affine_to_d95eq_uarray(
        TC.POLY_64_COEFFS,
        TC.FIEBIG2024_D48_SCALING,
        TC.FIEBIG2024_D48_OFFSET,
        _FIEBIG2024_D48_SCALING_OFFSET_PCOV,
    )
    return d47_coefs, d48_coefs


def coefs_have_calibration_uncertainties(coefs: uarray) -> bool:
    """True when coefficient standard errors are available for confidence bands."""
    try:
        return bool(np.any(np.asarray(coefs.s) > 0))
    except AttributeError:
        return False


def plot_kinetic_slope_to_d95eq(
    plot_slope: float,
    plot_slope_se: float,
    x_axis: str,
    y_axis: str,
) -> Tuple[float, float]:
    """
    Convert plot-native dy/dx to D95eq's ``∂Δ48/∂Δ47``.

    D95eq always uses ``kinetic_slope = ∂Δ48/∂Δ47``. When the plot has
    Δ48 on x and Δ47 on y (dual-clumped default), the visual slope is
    ``∂Δ47/∂Δ48 = 1 / (∂Δ48/∂Δ47)``.
    """
    if x_axis == "D47" and y_axis == "D48":
        return plot_slope, plot_slope_se
    if x_axis == "D48" and y_axis == "D47":
        if plot_slope == 0:
            raise ValueError("Disequilibrium slope cannot be zero.")
        d95eq_slope = 1.0 / plot_slope
        d95eq_slope_se = abs(plot_slope_se / (plot_slope ** 2))
        return d95eq_slope, d95eq_slope_se
    raise ValueError(f"D95eq axes must be D47 and D48, got {x_axis} vs {y_axis}")


def d95eq_kinetic_slope_label(x_axis: str, y_axis: str) -> str:
    """Sidebar label for the slope control in current plot coordinates (dy/dx)."""
    if {x_axis, y_axis} != {"D47", "D48"}:
        return "Disequilibrium slope"
    return f"Disequilibrium slope (∂Δ{y_axis.replace('D', '')}/∂Δ{x_axis.replace('D', '')})"


def d95eq_kinetic_slope_help(x_axis: str, y_axis: str) -> str:
    """Help text clarifying plot-native vs D95eq internal convention."""
    if x_axis == "D48" and y_axis == "D47":
        return (
            "Slope in current plot coordinates (∂Δ on y-axis / ∂Δ on x-axis). "
            "Internally converted to D95eq's ∂Δ48/∂Δ47."
        )
    return "Kinetic fractionation slope ∂Δ48/∂Δ47 (D95eq convention)."


def subset_uarray(values: uarray, mask: np.ndarray) -> uarray:
    """Select elements from a uarray by boolean mask."""
    return uarray([values[i] for i in range(values.size) if mask[i]])


def sample_pearson_map(
    reps_df: pd.DataFrame,
    sample_col: str = "Sample",
    d47_col: str = "D47",
    d48_col: str = "D48",
    min_n: int = 3,
    sig_level: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-sample Pearson correlation ``r(Δ47, Δ48)`` on replicate rows.

    Returns a mapping ``sample -> {"r", "p", "n", "significant", "used_rho"}``.
    ``used_rho`` equals ``r`` when the two-sided t-test on Pearson r is
    significant at ``sig_level`` **and** ``n >= min_n``, else ``0.0``.
    """
    from scipy import stats as _stats

    valid = reps_df.dropna(subset=[d47_col, d48_col])
    out: Dict[str, Dict[str, Any]] = {}
    for sample, sub in valid.groupby(sample_col):
        n = len(sub)
        entry: Dict[str, Any] = {
            "r": None,
            "p": None,
            "n": int(n),
            "significant": False,
            "used_rho": 0.0,
        }
        if n >= min_n:
            x = sub[d47_col].to_numpy(dtype=float)
            y = sub[d48_col].to_numpy(dtype=float)
            if x.std() > 0 and y.std() > 0:
                r, p = _stats.pearsonr(x, y)
                entry["r"] = float(r)
                entry["p"] = float(p)
                entry["significant"] = bool(p < sig_level)
                if entry["significant"]:
                    entry["used_rho"] = float(r)
        out[str(sample)] = entry
    return out


def teq_asymmetric_from_pdf(
    engine: Any,
    d47_u: uarray,
    p_levels: Tuple[float, float, float] = (0.16, 0.50, 0.84),
    tinc: float = 0.2,
) -> List[Tuple[float, float, float]]:
    """
    Return ``[(t_lo, t_med, t_hi), ...]`` for each ufloat in ``d47_u`` using
    ``Engine.Teq_pdf`` (analytical change-of-variables PDF, D95eq).

    ``p_levels`` are cumulative probability targets, e.g. ``(0.16, 0.5, 0.84)``
    for 1σ or ``(0.025, 0.5, 0.975)`` for 2σ. Because T(Δ47) is monotonic but
    non-linear, ``t_hi - t_med`` is generally not equal to ``t_med - t_lo``.
    """
    results: List[Tuple[float, float, float]] = []
    n = d47_u.size if hasattr(d47_u, "size") else len(d47_u)
    for i in range(n):
        d47 = d47_u[i]
        ti, pdf = engine.Teq_pdf(d47, Tinc=tinc)
        cdf = np.cumsum(pdf) * (ti[1] - ti[0])
        cdf /= cdf[-1]
        t_lo, t_med, t_hi = np.interp(p_levels, cdf, ti)
        results.append((float(t_lo), float(t_med), float(t_hi)))
    return results


def format_asym_temperature(
    t_lo: float,
    t_med: float,
    t_hi: float,
    unit: str = "°C",
    precision: int = 1,
) -> str:
    """Render an asymmetric temperature as ``t_med (+plus/-minus) unit``."""
    plus = t_hi - t_med
    minus = t_med - t_lo
    return f"{t_med:.{precision}f} (+{plus:.{precision}f}/-{minus:.{precision}f}) {unit}"
