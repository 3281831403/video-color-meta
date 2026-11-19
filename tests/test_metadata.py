"""Red tests for the color metadata kernels.

The tests are intentionally read-only: fix ``color/metadata.py`` instead of
editing this file.
"""

import math

import pytest

from color.metadata import (
    ColorError,
    compatible,
    convert_batch,
    convert_pixel,
    gamut_map,
    normalize,
    range_scale,
    tone_map,
    yuv_to_rgb_matrix,
)


def test_bt709_aliases_are_normalized_and_compatible():
    a = normalize(
        {
            "primaries": "bt.709",
            "transfer": "bt.709",
            "matrix": "1",
            "range": "limited",
        }
    )
    b = normalize(
        {
            "primaries": "1",
            "transfer": "bt709",
            "matrix": "bt709",
            "range": "full",
        }
    )

    assert a == {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "limited",
    }
    assert b == {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "full",
    }
    assert compatible(a, b) is True


def test_transfer_mismatch_makes_hdr_incompatible():
    hdr = {
        "primaries": "bt2020",
        "transfer": "pq",
        "matrix": "bt2020_ncl",
        "range": "limited",
    }
    sdr = {
        "primaries": "bt2020",
        "transfer": "srgb",
        "matrix": "bt2020_ncl",
        "range": "limited",
    }

    assert compatible(hdr, sdr) is False


def test_bt709_matrix_does_not_use_bt601_coefficients():
    matrix = yuv_to_rgb_matrix(
        {
            "primaries": "bt709",
            "transfer": "bt709",
            "matrix": "bt709",
            "range": "limited",
        }
    )

    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    assert matrix[0][0] == pytest.approx(1.0, abs=1e-6)
    assert matrix[0][2] == pytest.approx(1.5748, abs=1e-6)
    assert matrix[2][0] == pytest.approx(1.8556, abs=1e-6)
    assert abs(matrix[0][2] - 1.402) > 1e-3


def test_range_scale_distinguishes_limited_and_full():
    limited = range_scale(
        {
            "primaries": "bt709",
            "transfer": "bt709",
            "matrix": "bt709",
            "range": "limited",
        }
    )
    full = range_scale(
        {
            "primaries": "bt709",
            "transfer": "bt709",
            "matrix": "bt709",
            "range": "full",
        }
    )

    assert limited == (16, 235)
    assert full == (0, 255)


def test_tone_map_highlights_do_not_wrap():
    values = [0.0, 10.0, 100.0, 1000.0, 10000.0]

    for method in ("reinhard", "hable", "aces"):
        mapped = tone_map(values, 100.0, 1000.0, method)

        assert len(mapped) == len(values)
        assert all(math.isfinite(value) for value in mapped)
        assert all(0.0 <= value <= 100.0 + 1e-9 for value in mapped)
        assert all(
            mapped[index] <= mapped[index + 1] + 1e-9
            for index in range(len(mapped) - 1)
        )


def test_gamut_map_clips_to_target_gamut():
    mapped = gamut_map([0.5, 0.5, 0.0], "bt2020", "bt709")

    assert isinstance(mapped, list)
    assert len(mapped) == 3
    assert mapped[0] == pytest.approx(0.536425, abs=1e-6)
    assert mapped[1] == pytest.approx(0.504175, abs=1e-6)
    assert mapped[2] == pytest.approx(0.0, abs=1e-6)


def test_convert_pixel_decodes_range_before_matrix():
    limited = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "limited",
    }
    full = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "full",
    }

    black = convert_pixel(16, 128, 128, limited, full)
    white = convert_pixel(235, 128, 128, limited, full)

    assert black == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert white == pytest.approx((255.0, 255.0, 255.0), abs=1e-6)


def test_convert_batch_raises_instead_of_skipping_bad_metadata():
    valid = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "limited",
    }
    unknown = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bogus_matrix",
        "range": "limited",
    }

    with pytest.raises(ColorError):
        convert_batch([(16, 128, 128), (128, 128, 128)], unknown, valid)

    with pytest.raises(ColorError):
        convert_batch(
            [(16, 128, 128), (300, 128, 128)],
            valid,
            valid,
        )


def test_unknown_transfer_raises_color_error():
    with pytest.raises(ColorError):
        normalize(
            {
                "primaries": "bt709",
                "transfer": "unknown_transfer",
                "matrix": "bt709",
                "range": "limited",
            }
        )


def test_convert_pixel_quantizes_to_8_bit():
    limited = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "limited",
    }
    full = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "full",
    }

    pixel = convert_pixel(100, 140, 150, limited, full)

    assert len(pixel) == 3
    for component in pixel:
        assert 0.0 <= component <= 255.0
        assert abs(component - round(component)) <= 1e-9


def test_same_metadata_conversion_is_identity():
    limited = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "limited",
    }
    for pixel in ((16, 128, 128), (128, 128, 128), (235, 128, 128)):
        assert convert_pixel(*pixel, limited, limited) == pytest.approx(
            pixel,
            abs=1e-6,
        )

    full = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "full",
    }
    for pixel in ((0, 128, 128), (128, 128, 128), (255, 128, 128)):
        assert convert_pixel(*pixel, full, full) == pytest.approx(
            pixel,
            abs=1e-6,
        )


def test_tone_map_never_exceeds_target_nits():
    values = [0.0, 1.0, 18.0, 100.0, 1000.0, 10000.0]

    for method in ("reinhard", "hable", "aces"):
        mapped = tone_map(values, 100.0, 1000.0, method)

        assert len(mapped) == len(values)
        assert all(0.0 <= value <= 100.0 + 1e-9 for value in mapped)
        assert max(mapped) <= 100.0 + 1e-9


def test_empty_batch_returns_empty_frames():
    source = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "limited",
    }
    target = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "full",
    }

    result = convert_batch([], source, target)

    assert isinstance(result, dict)
    assert result.get("frames") == []
    assert result.get("count") == 0


def test_unknown_primaries_raises_color_error():
    with pytest.raises(ColorError):
        normalize(
            {
                "primaries": "unknown_primaries",
                "transfer": "bt709",
                "matrix": "bt709",
                "range": "limited",
            }
        )


def test_batch_success_preserves_order():
    source = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "limited",
    }
    target = {
        "primaries": "bt709",
        "transfer": "bt709",
        "matrix": "bt709",
        "range": "full",
    }
    frames = [
        (16, 128, 128),
        (64, 128, 128),
        (128, 128, 128),
        (192, 128, 128),
        (235, 128, 128),
    ]

    result = convert_batch(frames, source, target)
    converted = result["frames"]
    luma = [frame[0] for frame in converted]

    assert len(converted) == len(frames)
    assert luma == sorted(luma)
    assert len(set(luma)) == len(luma)
    assert converted[0][0] == pytest.approx(0.0, abs=1e-6)
    assert converted[-1][0] == pytest.approx(255.0, abs=1e-6)
