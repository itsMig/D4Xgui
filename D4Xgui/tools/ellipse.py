"""Pure-math helpers for 2-D uncertainty ellipses.

Uses only NumPy (eigendecomposition via numpy.linalg.eigh).
No UI / Streamlit imports.
"""

from typing import Tuple

import numpy as np


def ellipse_params_from_uncertainty(
    sx: float,
    sy: float,
    cov_xy: float | None = None,
    rho: float | None = None,
    n_sigma: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute semi-axes and orientation of a 2-D uncertainty ellipse.

    Parameters
    ----------
    sx, sy : float
        Marginal standard errors along x and y.
    cov_xy : float or None
        Off-diagonal covariance.  Takes priority over *rho*.
    rho : float or None
        Correlation coefficient; used as ``cov_xy = rho * sx * sy``
        when *cov_xy* is not supplied.
    n_sigma : float
        Scaling factor (1 for 1-SE, 2 for 2-SE, etc.).

    Returns
    -------
    semi_major, semi_minor, angle_rad : float
        Semi-axis lengths (already scaled by *n_sigma*) and rotation angle
        (radians, counter-clockwise from x-axis to the major axis).
    """
    if cov_xy is None:
        cov_xy = rho * sx * sy if rho is not None else 0.0

    cov = np.array([[sx**2, cov_xy], [cov_xy, sy**2]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns eigenvalues in ascending order; major axis = last
    semi_minor = n_sigma * np.sqrt(np.clip(eigenvalues[0], 0.0, None))
    semi_major = n_sigma * np.sqrt(np.clip(eigenvalues[1], 0.0, None))

    angle_rad = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    return float(semi_major), float(semi_minor), float(angle_rad)


def ellipse_boundary_points(
    cx: float,
    cy: float,
    semi_major: float,
    semi_minor: float,
    angle_rad: float,
    n_points: int = 72,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) arrays tracing a closed ellipse for ``go.Scatter``.

    Parameters
    ----------
    cx, cy : float
        Centre coordinates of the ellipse.
    semi_major, semi_minor : float
        Semi-axis lengths.
    angle_rad : float
        Counter-clockwise rotation of the major axis from the x-axis.
    n_points : int
        Number of boundary vertices (*excluding* the closing duplicate).
        The returned arrays have length ``n_points + 1`` so the curve is
        closed (first point == last point).

    Returns
    -------
    xs, ys : ndarray of shape (n_points + 1,)
    """
    theta = np.linspace(0, 2 * np.pi, n_points + 1)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    xs = cx + semi_major * np.cos(theta) * cos_a - semi_minor * np.sin(theta) * sin_a
    ys = cy + semi_major * np.cos(theta) * sin_a + semi_minor * np.sin(theta) * cos_a
    return xs, ys
