import pytest
from pathlib import Path
import numpy as np
import cv2

from head_counting.preprocessing import RGBTImageEqualizer


def test_rgbt_image_equalizer_process_pair() -> None:
    """Verifies that RGBTImageEqualizer resizes different aspect/resolution images to exact matching shapes."""
    equalizer = RGBTImageEqualizer(target_size=(1280, 1024), keep_aspect_ratio=True, thermal_clahe=True)

    dummy_rgb = np.full((6000, 8000, 3), 100, dtype=np.uint8)
    dummy_thermal = np.full((512, 640, 3), 180, dtype=np.uint8)

    rgb_eq, thermal_eq = equalizer.process_pair(dummy_rgb, dummy_thermal)

    assert rgb_eq.shape == (1024, 1280, 3)
    assert thermal_eq.shape == (1024, 1280, 3)
    assert rgb_eq.dtype == np.uint8
    assert thermal_eq.dtype == np.uint8


def test_rgbt_image_equalizer_process_files(tmp_path: Path) -> None:
    """Verifies that RGBTImageEqualizer loads image files and saves equalized output files to disk."""
    rgb_file = tmp_path / "DJI_0789_W.JPG"
    thermal_file = tmp_path / "DJI_0790_T.JPG"
    out_dir = tmp_path / "equalized"

    dummy_rgb = np.full((600, 800, 3), 120, dtype=np.uint8)
    dummy_thermal = np.full((500, 500, 3), 150, dtype=np.uint8)

    cv2.imwrite(str(rgb_file), dummy_rgb)
    cv2.imwrite(str(thermal_file), dummy_thermal)

    equalizer = RGBTImageEqualizer(target_size=(640, 480))
    out_rgb, out_thermal = equalizer.process_files(rgb_file, thermal_file, out_dir)

    assert out_rgb.exists()
    assert out_thermal.exists()

    saved_rgb = cv2.imread(str(out_rgb))
    saved_thermal = cv2.imread(str(out_thermal))

    assert saved_rgb.shape == (480, 640, 3)
    assert saved_thermal.shape == (480, 640, 3)
