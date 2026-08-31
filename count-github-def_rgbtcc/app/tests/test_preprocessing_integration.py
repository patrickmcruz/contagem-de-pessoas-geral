import pytest
from pathlib import Path
import numpy as np
import cv2

from head_counting.config import PipelineConfig, PreprocessingConfig
from head_counting.video import DualStreamVideoReader
from head_counting.pipeline import CountingPipeline


def test_dual_stream_video_reader_with_preprocessing(tmp_path: Path) -> None:
    """Verifies DualStreamVideoReader automatically applies homography alignment when enabled in config."""
    rgb_img_path = tmp_path / "test_W.jpg"
    thermal_img_path = tmp_path / "test_T.jpg"

    dummy_rgb = np.full((600, 800, 3), 100, dtype=np.uint8)
    dummy_thermal = np.full((500, 500, 3), 180, dtype=np.uint8)

    cv2.imwrite(str(rgb_img_path), dummy_rgb)
    cv2.imwrite(str(thermal_img_path), dummy_thermal)

    prep_cfg = PreprocessingConfig(
        enabled=True,
        mode="homography",
        target_resolution=[640, 480],
        thermal_clahe=True,
    )

    reader = DualStreamVideoReader(
        video_rgb_path=rgb_img_path,
        video_thermal_path=thermal_img_path,
        preprocessing_config=prep_cfg,
    )

    assert reader.metadata["width"] == 640
    assert reader.metadata["height"] == 480

    frames = []
    with reader:
        for f_rgb, f_thermal, idx in reader.iter_frames():
            frames.append((f_rgb, f_thermal, idx))

    assert len(frames) == 1
    assert frames[0][0].shape == (480, 640, 3)
    assert frames[0][1].shape == (480, 640, 3)


def test_pipeline_rgbt_e2e_with_preprocessing(tmp_path: Path) -> None:
    """Verifies CountingPipeline runs end-to-end with homography pre-processing enabled."""
    rgb_img_path = tmp_path / "test_W.jpg"
    thermal_img_path = tmp_path / "test_T.jpg"
    yaml_path = tmp_path / "config_prep.yaml"

    dummy_rgb = np.full((600, 800, 3), 120, dtype=np.uint8)
    dummy_thermal = np.full((500, 500, 3), 160, dtype=np.uint8)

    cv2.imwrite(str(rgb_img_path), dummy_rgb)
    cv2.imwrite(str(thermal_img_path), dummy_thermal)

    yaml_content = f"""
app:
  name: "test-preprocessing-pipeline"
paths:
  video_rgb: "{rgb_img_path}"
  video_thermal: "{thermal_img_path}"
  output_dir: "{tmp_path / 'out'}"
preprocessing:
  enabled: true
  mode: "homography"
  target_resolution: [640, 480]
  thermal_clahe: true
runtime:
  require_cuda: false
  batch_size: 1
inference:
  task: "density_count"
mlflow:
  enabled: false
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    config = PipelineConfig.from_yaml(yaml_path)
    pipeline = CountingPipeline(config=config)
    results = pipeline.run()

    assert results is not None
    assert results["processed_frames"] == 1
    assert (tmp_path / "out" / "annotated_heatmap.jpg").exists()
