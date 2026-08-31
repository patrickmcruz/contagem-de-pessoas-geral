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
