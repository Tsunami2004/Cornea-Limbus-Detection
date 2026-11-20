"""Unit tests for the cyclotorsion Pentacam loader + iris ROI helpers.

Covers Module 1 of the cyclotorsion feature:
- image load / auto-flip / preview resize
- normalised iris ROI serialisation, parsing and pixel conversion

Synthetic data only.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion.pentacam_loader import (
    PENTACAM_HEIGHT,
    PENTACAM_WIDTH,
    flip_pentacam,
    format_roi_norm,
    load_pentacam_image,
    parse_roi_norm,
    resize_for_preview,
    roi_norm_to_px,
)


# ── Dimensions ──────────────────────────────────────────────────────

def test_pentacam_dimensions_are_1200x838():
    assert PENTACAM_WIDTH == 1200
    assert PENTACAM_HEIGHT == 838


# ── Load / flip / resize ────────────────────────────────────────────

def test_load_and_flip_synthetic_image(tmp_path):
    img = np.zeros((PENTACAM_HEIGHT, PENTACAM_WIDTH, 3), dtype=np.uint8)
    img[0:5, 100:1100] = (0, 0, 255)  # red strip on the top edge
    img[100:800, 0:5] = (0, 255, 0)   # green strip on the left edge
    path = tmp_path / "eye.png"
    assert cv2.imwrite(str(path), img)

    loaded = load_pentacam_image(str(path))
    assert loaded is not None
    assert loaded.shape == (PENTACAM_HEIGHT, PENTACAM_WIDTH, 3)
    # Before flipping: red on top, green on the left.
    assert loaded[2, PENTACAM_WIDTH // 2].tolist() == [0, 0, 255]
    assert loaded[PENTACAM_HEIGHT // 2, 2].tolist() == [0, 255, 0]

    flipped = flip_pentacam(loaded)
    # A 180° flip moves the top strip to the bottom and the left strip to
    # the right (vertical upside-down kept + right-to-left mirror).
    assert flipped[PENTACAM_HEIGHT - 3, PENTACAM_WIDTH // 2].tolist() == [0, 0, 255]
    assert flipped[PENTACAM_HEIGHT // 2, PENTACAM_WIDTH - 3].tolist() == [0, 255, 0]


def test_load_missing_file_returns_none():
    assert load_pentacam_image("definitely_missing_file.png") is None


def test_resize_for_preview_keeps_aspect_ratio():
    img = np.zeros((PENTACAM_HEIGHT, PENTACAM_WIDTH, 3), dtype=np.uint8)
    thumb = resize_for_preview(img, max_width=560, max_height=196)
    assert thumb.shape[1] <= 560
    assert thumb.shape[0] <= 196
    # Aspect ratio preserved to within rounding.
    assert abs((thumb.shape[1] / thumb.shape[0]) - (PENTACAM_WIDTH / PENTACAM_HEIGHT)) < 0.02


def test_resize_for_preview_does_not_upscale():
    small = np.zeros((100, 100, 3), dtype=np.uint8)
    out = resize_for_preview(small, max_width=560, max_height=320)
    assert out.shape[:2] == (100, 100)


# ── ROI serialisation ───────────────────────────────────────────────

def test_roi_round_trip():
    roi = (0.1, 0.2, 0.6, 0.75)
    text = format_roi_norm(roi)
    assert text == "0.100000,0.200000,0.600000,0.750000"
    assert parse_roi_norm(text) == pytest.approx(roi)


def test_parse_roi_rejects_malformed():
    assert parse_roi_norm("") is None
    assert parse_roi_norm("nope") is None
    assert parse_roi_norm("0.1,0.2,0.3") is None
    assert parse_roi_norm("0.1,0.2,0.3,0.4,0.5") is None


def test_parse_roi_rejects_degenerate():
    assert parse_roi_norm("0.6,0.2,0.1,0.75") is None  # x1 < x0
    assert parse_roi_norm("0.1,0.75,0.6,0.2") is None  # y1 < y0


def test_parse_roi_clamps_out_of_range():
    parsed = parse_roi_norm("1.5,-0.2,2.0,0.9")
    assert parsed == pytest.approx((1.0, 0.0, 1.0, 0.9))


def test_roi_norm_to_px_matches_pentacam_size():
    px = roi_norm_to_px((0.1, 0.2, 0.6, 0.75))
    assert px == (120, 168, 720, 628)
