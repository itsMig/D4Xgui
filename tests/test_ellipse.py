"""Tests for D4Xgui.tools.ellipse helpers."""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "D4Xgui"))

from tools.ellipse import ellipse_boundary_points, ellipse_params_from_uncertainty


class TestEllipseParams:
    """Tests for ellipse_params_from_uncertainty."""

    def test_zero_covariance_axis_aligned(self):
        """Zero covariance → angle is 0 or ±π/2 (axes align with x/y)."""
        sx, sy = 0.02, 0.01
        major, minor, angle = ellipse_params_from_uncertainty(sx, sy, cov_xy=0.0)
        assert major == pytest.approx(sx, abs=1e-12)
        assert minor == pytest.approx(sy, abs=1e-12)
        assert angle == pytest.approx(0.0, abs=1e-12)

    def test_positive_covariance_angle(self):
        """Positive covariance → major axis tilted into first quadrant (angle mod π in (0, π/4))."""
        major, minor, angle = ellipse_params_from_uncertainty(
            0.02, 0.01, cov_xy=0.0001,
        )
        angle_norm = angle % math.pi
        assert 0 < angle_norm < math.pi / 4

    def test_negative_covariance_angle(self):
        """Negative covariance → major axis tilted into fourth quadrant (angle mod π in (3π/4, π))."""
        major, minor, angle = ellipse_params_from_uncertainty(
            0.02, 0.01, cov_xy=-0.0001,
        )
        angle_norm = angle % math.pi
        assert 3 * math.pi / 4 < angle_norm < math.pi

    def test_equal_se_zero_cov_gives_circle(self):
        """Equal sx/sy and zero covariance → circle (semi_major == semi_minor)."""
        major, minor, angle = ellipse_params_from_uncertainty(0.01, 0.01, cov_xy=0.0)
        assert major == pytest.approx(minor, abs=1e-12)

    def test_missing_cov_and_rho_defaults_to_zero(self):
        """Neither cov_xy nor rho supplied → same as zero covariance."""
        r1 = ellipse_params_from_uncertainty(0.02, 0.01)
        r2 = ellipse_params_from_uncertainty(0.02, 0.01, cov_xy=0.0)
        for a, b in zip(r1, r2):
            assert a == pytest.approx(b, abs=1e-12)

    def test_rho_equivalent_to_cov_xy(self):
        """Supplying rho produces the same result as the equivalent cov_xy."""
        sx, sy, rho = 0.02, 0.01, 0.5
        r1 = ellipse_params_from_uncertainty(sx, sy, cov_xy=rho * sx * sy)
        r2 = ellipse_params_from_uncertainty(sx, sy, rho=rho)
        for a, b in zip(r1, r2):
            assert a == pytest.approx(b, abs=1e-12)

    def test_n_sigma_doubles_semiaxes(self):
        """n_sigma=2 doubles semi-axes relative to n_sigma=1."""
        sx, sy = 0.02, 0.01
        maj1, min1, ang1 = ellipse_params_from_uncertainty(sx, sy, n_sigma=1)
        maj2, min2, ang2 = ellipse_params_from_uncertainty(sx, sy, n_sigma=2)
        assert maj2 == pytest.approx(2 * maj1, abs=1e-12)
        assert min2 == pytest.approx(2 * min1, abs=1e-12)
        assert ang2 == pytest.approx(ang1, abs=1e-12)


class TestBoundaryPoints:
    """Tests for ellipse_boundary_points."""

    def test_closed_curve(self):
        """First and last boundary point must coincide."""
        xs, ys = ellipse_boundary_points(0, 0, 1, 0.5, 0.0, n_points=36)
        assert xs[0] == pytest.approx(xs[-1], abs=1e-12)
        assert ys[0] == pytest.approx(ys[-1], abs=1e-12)

    def test_correct_point_count(self):
        """Returned arrays have length n_points + 1."""
        n = 72
        xs, ys = ellipse_boundary_points(0, 0, 1, 0.5, 0.0, n_points=n)
        assert len(xs) == n + 1
        assert len(ys) == n + 1

    def test_centre_offset(self):
        """Centre offset is reflected in the mean of the boundary."""
        cx, cy = 3.0, -1.5
        xs, ys = ellipse_boundary_points(cx, cy, 1, 1, 0.0, n_points=360)
        assert np.mean(xs[:-1]) == pytest.approx(cx, abs=1e-6)
        assert np.mean(ys[:-1]) == pytest.approx(cy, abs=1e-6)
