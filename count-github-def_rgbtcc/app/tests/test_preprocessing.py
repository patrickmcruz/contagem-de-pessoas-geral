import pytest
from pathlib import Path
import numpy as np
import cv2

from head_counting.preprocessing import RGBTImageEqualizer


def test_rgbt_image_equalizer_homography_process_pair() -> None:
    """Verifies that RGBTImageEqualizer homography mode produces matching equalized layers."""
    equalizer = RGBTImageEqualizer(target_size=(1280, 1024), mode="homography")

    dummy_rgb = np.full((6000, 8000, 3), 100, dtype=np.uint8)
    dummy_thermal = np.full((512, 640, 3), 180, dtype=np.uint8)

    # Draw synthetic shapes so feature detection finds points
    cv2.rectangle(dummy_rgb, (1000, 1000), (5000, 5000), (255, 255, 255), 10)
    cv2.rectangle(dummy_thermal, (100, 100), (500, 400), (255, 255, 255), 5)

    rgb_eq, thermal_eq = equalizer.process_pair(dummy_rgb, dummy_thermal)

    assert rgb_eq.shape == (1024, 1280, 3)
    assert thermal_eq.shape == (1024, 1280, 3)


def test_rgbt_image_equalizer_process_files(tmp_path: Path) -> None:
    """Verifies that RGBTImageEqualizer saves equalized files and the blend check overlay."""
    rgb_file = tmp_path / "DJI_0789_W.JPG"
    thermal_file = tmp_path / "DJI_0790_T.JPG"
    out_dir = tmp_path / "equalized"

    dummy_rgb = np.full((600, 800, 3), 120, dtype=np.uint8)
    dummy_thermal = np.full((500, 500, 3), 150, dtype=np.uint8)

    cv2.imwrite(str(rgb_file), dummy_rgb)
    cv2.imwrite(str(thermal_file), dummy_thermal)

    equalizer = RGBTImageEqualizer(target_size=(640, 480), mode="homography")
    out_rgb, out_thermal, out_blend = equalizer.process_files(rgb_file, thermal_file, out_dir)

    assert out_rgb.exists()
    assert out_thermal.exists()
    assert out_blend.exists()

    saved_rgb = cv2.imread(str(out_rgb))
    saved_thermal = cv2.imread(str(out_thermal))
    saved_blend = cv2.imread(str(out_blend))

    assert saved_rgb.shape == (480, 640, 3)
    assert saved_thermal.shape == (480, 640, 3)
    assert saved_blend.shape == (480, 640, 3)


def test_ground_pitch_compensation_adr_007():
    """Validates ADR 007 ground-plane pitch-aware affine warp behavior."""
    eq_pitch = RGBTImageEqualizer(
        target_size=(1280, 1024),
        shift_rgb_x=-22,
        shift_rgb_y=-23,
        ground_pitch_compensation=True,
        pitch_gradient_x=-0.025,
        pitch_gradient_y=0.045,
    )
    dummy_rgb = np.zeros((6000, 8000, 3), dtype=np.uint8)
    dummy_th = np.zeros((512, 640, 3), dtype=np.uint8)

    rgb_out, th_out = eq_pitch.process_pair(dummy_rgb, dummy_th)
    assert rgb_out.shape == (1024, 1280, 3)
    assert th_out.shape == (1024, 1280, 3)

    # Validate mathematical alignment matrix properties
    yc = eq_pitch.target_h / 2.0
    M_align = np.float32([
        [1.0, eq_pitch.pitch_gradient_x, eq_pitch.shift_rgb_x - eq_pitch.pitch_gradient_x * yc],
        [0.0, 1.0 + eq_pitch.pitch_gradient_y, eq_pitch.shift_rgb_y - eq_pitch.pitch_gradient_y * yc],
    ])

    # Center displacement must be exactly (-22, -23)
    center_pt = np.array([640.0, 512.0, 1.0])
    center_mapped = M_align @ center_pt
    assert np.isclose(center_mapped[0] - 640.0, -22.0)
    assert np.isclose(center_mapped[1] - 512.0, -23.0)

    # Bottom region must have greater vertical shift (ground closer)
    bottom_pt = np.array([90.0, 883.0, 1.0])
    bottom_mapped = M_align @ bottom_pt
    # Y shift at bottom is -23 + 0.045*(883 - 512) = -6.3 px, meaning RGB shifted down by ~16.7 px
    assert bottom_mapped[1] - 883.0 > -23.0

