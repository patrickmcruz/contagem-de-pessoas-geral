import pytest
from pathlib import Path
import numpy as np
import cv2

from head_counting.video import DualStreamVideoReader
from head_counting.config import PipelineConfig
from head_counting.pipeline import CountingPipeline


def test_dual_stream_image_reader(tmp_path: Path) -> None:
    """Verifies that DualStreamVideoReader handles single RGB and Thermal image files."""
    rgb_img_path = tmp_path / "test_W.jpg"
    thermal_img_path = tmp_path / "test_T.jpg"

    dummy_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
    dummy_thermal = np.full((100, 100, 3), 200, dtype=np.uint8)

    cv2.imwrite(str(rgb_img_path), dummy_rgb)
    cv2.imwrite(str(thermal_img_path), dummy_thermal)

    reader = DualStreamVideoReader(rgb_img_path, thermal_img_path)
    assert reader.metadata["frame_count"] == 1
    assert reader.metadata["width"] == 100
    assert reader.metadata["height"] == 100

    frames = []
    with reader:
        for f_rgb, f_thermal, idx in reader.iter_frames():
            frames.append((f_rgb, f_thermal, idx))

    assert len(frames) == 1
    assert frames[0][2] == 0
    assert frames[0][0].shape == (100, 100, 3)
    assert frames[0][1].shape == (100, 100, 3)


def test_dji_images_exist() -> None:
    """Verifies that the actual DJI test images exist in app/input/images."""
    images_dir = Path("/home/patrickcruz/Git/projects/contagem-de-pessoas/count-github-def_rgbtcc/app/input/images")
    rgb_image = images_dir / "DJI_0789_W.JPG"
    thermal_image = images_dir / "DJI_0790_T.JPG"

    assert rgb_image.exists()
    assert thermal_image.exists()
