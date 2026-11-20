"""Tests for the Pentacam-preview detection overlay geometry.

When a Pentacam image is loaded in the Add-Patient "Pentacam Preview" the
main-window detector is run on it (still-image path) and the detected pupil /
limbus are overlaid.  Those ellipses carry a tilt (``angle_deg``), so the
overlay is drawn as a sampled polygon -- ``_preview_ellipse_points`` -- rather
than a plain Tk oval.  These tests cover that pure geometry helper.

Synthetic data only; no Tk / no model required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pupil_tracking.interface.gui_app import PupilTrackingGUI


def _ellipse(cx, cy, semi_major, semi_minor, angle_deg=0.0):
    return SimpleNamespace(
        center_x=cx,
        center_y=cy,
        semi_major=semi_major,
        semi_minor=semi_minor,
        angle_deg=angle_deg,
        radius=(semi_major + semi_minor) / 2.0,
    )


def _bbox(points):
    xs = points[0::2]
    ys = points[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def test_polygon_has_two_coords_per_segment():
    pts = PupilTrackingGUI._preview_ellipse_points(
        _ellipse(50, 60, 10, 5), 1.0, 1.0, segments=36
    )
    assert len(pts) == 72


def test_axis_aligned_ellipse_applies_scale_and_offset():
    # centre (100, 200) * 0.5 + 2 = (52, 102); semi-axes (20, 10)
    pts = PupilTrackingGUI._preview_ellipse_points(
        _ellipse(100, 200, 40, 20), 0.5, 0.5, offset=2.0
    )
    x0, y0, x1, y1 = _bbox(pts)
    assert x0 == pytest.approx(32.0, abs=0.5)
    assert x1 == pytest.approx(72.0, abs=0.5)
    assert y0 == pytest.approx(92.0, abs=0.5)
    assert y1 == pytest.approx(112.0, abs=0.5)


def test_quarter_turn_swaps_the_axes():
    upright = _bbox(
        PupilTrackingGUI._preview_ellipse_points(
            _ellipse(0, 0, 40, 10, angle_deg=0.0), 1.0, 1.0
        )
    )
    turned = _bbox(
        PupilTrackingGUI._preview_ellipse_points(
            _ellipse(0, 0, 40, 10, angle_deg=90.0), 1.0, 1.0
        )
    )
    assert (upright[2] - upright[0]) == pytest.approx(80.0, abs=0.5)
    assert (upright[3] - upright[1]) == pytest.approx(20.0, abs=0.5)
    assert (turned[2] - turned[0]) == pytest.approx(20.0, abs=0.5)
    assert (turned[3] - turned[1]) == pytest.approx(80.0, abs=0.5)


def test_missing_angle_defaults_to_zero():
    no_angle = SimpleNamespace(
        center_x=0, center_y=0, semi_major=30, semi_minor=10
    )
    pts = PupilTrackingGUI._preview_ellipse_points(no_angle, 1.0, 1.0)
    x0, y0, x1, y1 = _bbox(pts)
    assert (x1 - x0) == pytest.approx(60.0, abs=0.5)
    assert (y1 - y0) == pytest.approx(20.0, abs=0.5)


def test_independent_x_y_scale_for_non_square_thumbnails():
    # A 1200x838 image shown as a 556x196 thumbnail scales x and y equally in
    # practice, but the helper must honour whatever scale it is handed.
    pts = PupilTrackingGUI._preview_ellipse_points(
        _ellipse(0, 0, 100, 100), 0.5, 0.2, offset=0.0
    )
    x0, y0, x1, y1 = _bbox(pts)
    assert (x1 - x0) == pytest.approx(100.0, abs=0.5)
    assert (y1 - y0) == pytest.approx(40.0, abs=0.5)
