"""Tests for Module 2 — fast vectorised iris polar unwrapping.

These tests are deterministic and synthetic: no real clinical imagery is used.
The key property exercised here is *rotation equivariance* — rotating the
source image about the iris centre must cyclically shift the unwrapped strip
by the corresponding number of columns.  That property is the foundation the
cyclotorsion stage relies on, so it is asserted explicitly.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion.polar_unwrap import (
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    UnwrapMaps,
    angle_to_column,
    apply_unwrap_maps,
    build_unwrap_maps,
    column_to_angle,
    roll_strip,
    unwrap_from_ellipses,
    unwrap_iris,
)
from pupil_tracking.iris.types import IrisROI
from pupil_tracking.utils.types import EllipseParams


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CX = CY = 100.0
PUPIL_R = 30.0
LIMBUS_R = 80.0
IMG_SIZE = 200


def _circular_roi(
    pupil_r: float = PUPIL_R,
    limbus_r: float = LIMBUS_R,
    inner_inset: float = 0.0,
    outer_inset: float = 0.0,
) -> IrisROI:
    """Build a valid annular IrisROI with circular pupil/limbus boundaries."""
    return IrisROI(
        valid=True,
        reason="ok",
        center_x=CX,
        center_y=CY,
        pupil_semi_major=pupil_r,
        pupil_semi_minor=pupil_r,
        pupil_angle_deg=0.0,
        limbus_semi_major=limbus_r,
        limbus_semi_minor=limbus_r,
        limbus_angle_deg=0.0,
        inner_inset_frac=inner_inset,
        outer_inset_frac=outer_inset,
        pupil_radius_px=pupil_r,
        limbus_radius_px=limbus_r,
    )


def _spoke_image(angle_deg: float = 0.0, width: int = 5) -> np.ndarray:
    """A dark image with one bright radial spoke through the iris annulus."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    a = math.radians(angle_deg)
    p0 = (int(round(CX + (PUPIL_R - 2) * math.cos(a))),
          int(round(CY + (PUPIL_R - 2) * math.sin(a))))
    p1 = (int(round(CX + (LIMBUS_R - 5) * math.cos(a))),
          int(round(CY + (LIMBUS_R - 5) * math.sin(a))))
    cv2.line(img, p0, p1, 255, width)
    return img


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    m = cv2.getRotationMatrix2D((CX, CY), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)


def _peak_column(strip: np.ndarray) -> int:
    """Column with the highest mean intensity (the spoke's angular position)."""
    profile = strip.reshape(-1, strip.shape[1]).mean(axis=0)
    return int(np.argmax(profile))


def _circular_distance(a: int, b: int, width: int) -> int:
    d = (a - b) % width
    return min(d, width - d)


# ---------------------------------------------------------------------------
# Basic geometry / shape
# ---------------------------------------------------------------------------

def test_default_strip_size_matches_cnn_encoder():
    assert (DEFAULT_STRIP_HEIGHT, DEFAULT_STRIP_WIDTH) == (64, 512)


def test_unwrap_gray_shape_and_dtype():
    strip = unwrap_iris(_spoke_image(), _circular_roi(), (64, 512))
    assert strip is not None
    assert strip.shape == (64, 512)
    assert strip.dtype == np.uint8


def test_unwrap_preserves_channels():
    gray = _spoke_image()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    strip = unwrap_iris(bgr, _circular_roi(), (48, 256))
    assert strip is not None
    assert strip.shape == (48, 256, 3)


def test_build_maps_returns_float32_maps():
    maps = build_unwrap_maps(_circular_roi(), (32, 128))
    assert isinstance(maps, UnwrapMaps)
    assert maps.map_x.shape == (32, 128)
    assert maps.map_y.shape == (32, 128)
    assert maps.map_x.dtype == np.float32
    assert maps.height == 32 and maps.width == 128


def test_apply_maps_matches_unwrap_iris():
    img = _spoke_image()
    roi = _circular_roi()
    maps = build_unwrap_maps(roi, (40, 200), angle_offset_deg=17.0)
    a = apply_unwrap_maps(img, maps)
    b = unwrap_iris(img, roi, (40, 200), angle_offset_deg=17.0)
    assert np.array_equal(a, b)


def test_invalid_roi_returns_none():
    invalid = IrisROI(valid=False)
    assert unwrap_iris(_spoke_image(), invalid) is None
    assert build_unwrap_maps(invalid) is None
    assert unwrap_iris(_spoke_image(), None) is None


def test_degenerate_output_size_returns_none():
    assert build_unwrap_maps(_circular_roi(), (0, 100)) is None
    assert build_unwrap_maps(_circular_roi(), (100, 0)) is None


# ---------------------------------------------------------------------------
# Angular helpers
# ---------------------------------------------------------------------------

def test_column_angle_roundtrip():
    for width in (360, 512):
        for col in (0, 1, width // 2, width - 1):
            angle = column_to_angle(col, width)
            assert 0.0 <= angle < 360.0
            assert _circular_distance(angle_to_column(angle, width), col, width) <= 1


def test_angle_to_column_wraps():
    assert angle_to_column(0.0, 360) == 0
    assert angle_to_column(360.0, 360) == 0
    assert angle_to_column(90.0, 360) == 90
    assert angle_to_column(-90.0, 360) == 270


def test_roll_strip_is_cyclic():
    strip = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    rolled = roll_strip(strip, 2)
    assert np.array_equal(rolled[:, 2:], strip[:, :-2])
    assert np.array_equal(rolled[:, :2], strip[:, -2:])
    assert np.array_equal(roll_strip(strip, 0), strip)


# ---------------------------------------------------------------------------
# Geometry correctness
# ---------------------------------------------------------------------------

def test_spoke_at_zero_angle_sits_at_column_zero():
    strip = unwrap_iris(_spoke_image(angle_deg=0.0), _circular_roi(), (64, 512))
    peak = _peak_column(strip)
    # Column 0 and column 511 are adjacent around the seam.
    assert _circular_distance(peak, 0, 512) <= 1


def test_spoke_at_ninety_degrees_sits_at_quarter_width():
    strip = unwrap_iris(_spoke_image(angle_deg=90.0), _circular_roi(), (64, 512))
    peak = _peak_column(strip)
    assert _circular_distance(peak, 512 // 4, 512) <= 1


def test_inner_rows_map_to_pupil_side_and_outer_rows_to_limbus_side():
    # An annulus that is bright from the pupil boundary out to the mid radius
    # and dark from the mid radius out to the limbus must produce a strip whose
    # pupil-side (inner) rows are bright and whose limbus-side (outer) rows are
    # dark.  This pins down the radial orientation of the strip.
    mid = int((PUPIL_R + LIMBUS_R) / 2)
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(img, (int(CX), int(CY)), mid, 255, -1)
    cv2.circle(img, (int(CX), int(CY)), int(PUPIL_R), 0, -1)

    strip = unwrap_iris(img, _circular_roi(), (64, 512))
    top = float(strip[:24].mean())      # near the pupil boundary
    bottom = float(strip[40:].mean())   # near the limbus boundary
    assert top > bottom
    assert top > 200.0
    assert bottom < 50.0


# ---------------------------------------------------------------------------
# Cyclotorsion-critical property: rotation == cyclic column shift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [15.0, 30.0, 45.0, 90.0])
def test_rotation_shifts_strip_columns(angle_deg):
    width = 512
    base = _spoke_image(angle_deg=0.0)
    rotated = _rotate(base, angle_deg)
    roi = _circular_roi()

    strip_base = unwrap_iris(base, roi, (64, width))
    strip_rot = unwrap_iris(rotated, roi, (64, width))

    shift = _circular_distance(_peak_column(strip_rot), _peak_column(strip_base), width)
    expected = angle_deg / 360.0 * width
    assert abs(shift - expected) <= 2.0


def test_rotated_strip_is_cyclic_shift_of_original():
    width = 256
    base = _spoke_image(angle_deg=0.0)
    rotated = _rotate(base, 40.0)
    roi = _circular_roi()

    strip_base = unwrap_iris(base, roi, (64, width)).astype(np.float32)
    strip_rot = unwrap_iris(rotated, roi, (64, width)).astype(np.float32)

    # Find the integer column shift that best aligns the two strips.
    profile_base = strip_base.mean(axis=0)
    best_shift, best_err = 0, None
    for s in range(width):
        err = float(np.sum((profile_base - np.roll(strip_rot.mean(axis=0), s)) ** 2))
        if best_err is None or err < best_err:
            best_err, best_shift = err, s

    circ = _circular_distance(best_shift, 0, width)
    expected = 40.0 / 360.0 * width
    assert abs(circ - expected) <= 2.0


def test_angle_offset_shifts_strip():
    width = 512
    img = _spoke_image(angle_deg=0.0)
    roi = _circular_roi()

    base = unwrap_iris(img, roi, (64, width), angle_offset_deg=0.0)
    offset = unwrap_iris(img, roi, (64, width), angle_offset_deg=90.0)

    peak_base = _peak_column(base)
    peak_offset = _peak_column(offset)
    # A +90 deg offset moves the angle-0 content to the 3/4 column.
    assert _circular_distance(peak_offset, (peak_base + 3 * width // 4) % width, width) <= 1


def test_angle_offset_equals_roll():
    width = 256
    img = _spoke_image(angle_deg=0.0)
    roi = _circular_roi()

    base = unwrap_iris(img, roi, (64, width), angle_offset_deg=0.0)
    offset = unwrap_iris(img, roi, (64, width), angle_offset_deg=90.0)

    assert np.array_equal(offset, np.roll(base, -width // 4, axis=1))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def test_unwrap_from_ellipses():
    img = _spoke_image(angle_deg=0.0)
    pupil = EllipseParams(center_x=CX, center_y=CY, semi_major=PUPIL_R,
                          semi_minor=PUPIL_R, angle_deg=0.0)
    limbus = EllipseParams(center_x=CX, center_y=CY, semi_major=LIMBUS_R,
                           semi_minor=LIMBUS_R, angle_deg=0.0)
    strip = unwrap_from_ellipses(
        img, pupil, limbus, (48, 256),
        inner_inset_frac=0.0, outer_inset_frac=0.0,
    )
    assert strip is not None
    assert strip.shape == (48, 256)
    assert _circular_distance(_peak_column(strip), 0, 256) <= 1


def test_unwrap_from_ellipses_invalid_geometry_returns_none():
    img = _spoke_image()
    # Pupil larger than limbus -> implausible geometry -> invalid ROI.
    pupil = EllipseParams(center_x=CX, center_y=CY, semi_major=90.0,
                          semi_minor=90.0, angle_deg=0.0)
    limbus = EllipseParams(center_x=CX, center_y=CY, semi_major=80.0,
                           semi_minor=80.0, angle_deg=0.0)
    assert unwrap_from_ellipses(img, pupil, limbus) is None
    assert unwrap_from_ellipses(img, None, limbus) is None
