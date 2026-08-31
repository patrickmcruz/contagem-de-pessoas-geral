import pytest
from pathlib import Path
import numpy as np
import cv2

from head_counting.config import PipelineConfig
from head_counting.medallion import MedallionPipelineRunner


def test_medallion_pipeline_runner_silver_and_gold(tmp_path: Path) -> None:
    """Verifies MedallionPipelineRunner transforms Bronze raw images to Silver aligned images, then Gold analytics."""
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    yaml_path = tmp_path / "data_medallion.yaml"

    bronze_dir.mkdir(parents=True, exist_ok=True)

    rgb_raw_path = bronze_dir / "DJI_0789_W.JPG"
    thermal_raw_path = bronze_dir / "DJI_0790_T.JPG"

    dummy_rgb = np.full((600, 800, 3), 120, dtype=np.uint8)
    dummy_thermal = np.full((500, 500, 3), 160, dtype=np.uint8)

    cv2.imwrite(str(rgb_raw_path), dummy_rgb)
    cv2.imwrite(str(thermal_raw_path), dummy_thermal)

    yaml_content = f"""
app:
  name: "test-medallion-pipeline"
paths:
  video_rgb: "{rgb_raw_path}"
  video_thermal: "{thermal_raw_path}"
  bronze_dir: "{bronze_dir}"
  silver_dir: "{silver_dir}"
  gold_dir: "{gold_dir}"
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
    runner = MedallionPipelineRunner(config=config, config_path=yaml_path)

    # 1. Test Silver layer transformation
    silver_rgb, silver_thermal = runner.run_silver()

    assert silver_rgb.exists()
    assert silver_thermal.exists()
    assert (silver_dir / "images").exists()
    assert (silver_dir / "layer_blend_checks").exists()

    img_s_rgb = cv2.imread(str(silver_rgb))
    assert img_s_rgb.shape == (480, 640, 3)

    # 2. Test Gold layer prediction and analytics
    summary = runner.run_gold(silver_rgb, silver_thermal)

    assert summary is not None
    assert summary["processed_frames"] == 1
    assert (gold_dir / "heatmaps").exists()
    assert (gold_dir / "telemetry" / "frame_counts.csv").exists()
    assert (gold_dir / "telemetry" / "summary.json").exists()


def test_medallion_pipeline_runner_run_all(tmp_path: Path) -> None:
    """Verifies runner.run_all() executes Bronze -> Silver -> Gold in a single invocation."""
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    yaml_path = tmp_path / "data_medallion_all.yaml"

    bronze_dir.mkdir(parents=True, exist_ok=True)

    rgb_raw_path = bronze_dir / "sample_W.jpg"
    thermal_raw_path = bronze_dir / "sample_T.jpg"

    dummy_rgb = np.full((600, 800, 3), 100, dtype=np.uint8)
    dummy_thermal = np.full((500, 500, 3), 150, dtype=np.uint8)

    cv2.imwrite(str(rgb_raw_path), dummy_rgb)
    cv2.imwrite(str(thermal_raw_path), dummy_thermal)

    yaml_content = f"""
app:
  name: "test-medallion-all"
paths:
  video_rgb: "{rgb_raw_path}"
  video_thermal: "{thermal_raw_path}"
  bronze_dir: "{bronze_dir}"
  silver_dir: "{silver_dir}"
  gold_dir: "{gold_dir}"
preprocessing:
  enabled: true
  mode: "homography"
  target_resolution: [640, 480]
runtime:
  require_cuda: false
  batch_size: 1
mlflow:
  enabled: false
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    config = PipelineConfig.from_yaml(yaml_path)
    runner = MedallionPipelineRunner(config=config, config_path=yaml_path)
    summary = runner.run_all()

    assert summary["processed_frames"] == 1
    assert (gold_dir / "telemetry" / "summary.json").exists()
