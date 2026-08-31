import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import yaml

from head_counting import PipelineConfig, CountingPipeline


@pytest.fixture
def rgbt_temp_workspace():
    """Sets up a temporary directory with dummy RGB and Thermal video files and YAML config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        rgb_path = temp_path / "dummy_rgb.mp4"
        thermal_path = temp_path / "dummy_thermal.mp4"
        rgb_path.touch()
        thermal_path.touch()

        config_data = {
            "app": {
                "name": "test-rgbt-pipeline-e2e"
            },
            "paths": {
                "video_rgb": str(rgb_path),
                "video_thermal": str(thermal_path),
                "weights": "model.pth",
                "output_dir": str(temp_path / "output")
            },
            "runtime": {
                "require_cuda": False,
                "batch_size": 2
            },
            "inference": {
                "task": "density_count",
                "imgsz": 640
            },
            "output": {
                "save_annotated_video": False,
                "save_density_heatmap": True,
                "heatmap_colormap": "JET",
                "heatmap_alpha": 0.5,
                "save_frame_counts": True,
                "save_summary": True,
                "save_snapshot_every_n_frames": 0
            }
        }

        config_file = temp_path / "data_rgbt.yaml"
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        yield config_file


@patch("head_counting.pipeline.DEFModelHandler")
@patch("head_counting.pipeline.DualStreamVideoReader")
def test_rgbt_pipeline_execution_flow(mock_reader_class, mock_handler_class, rgbt_temp_workspace) -> None:
    """Verifies that CountingPipeline executes dual-stream RGBT workflow and writes outputs."""
    # 1. Mock DualStreamVideoReader
    mock_reader = MagicMock()
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    mock_reader.metadata = {
        "width": 100,
        "height": 100,
        "fps": 30.0,
        "frame_count": 2,
        "rgb": {"width": 100, "height": 100},
        "thermal": {"width": 100, "height": 100}
    }
    
    # Return 2 frames, then context manager exit
    mock_reader.iter_frames.return_value = [
        (dummy_frame, dummy_frame, 0),
        (dummy_frame, dummy_frame, 1)
    ]
    mock_reader_class.return_value.__enter__.return_value = mock_reader

    # 2. Mock DEFModelHandler
    mock_handler = MagicMock()
    mock_handler.weights_ref = "model.pth"
    mock_handler.predict_args = {"device": "cpu", "imgsz": 640}
    dummy_density = np.full((100, 100), 0.05, dtype=np.float32)
    mock_handler.predict_batch.return_value = [
        {"orig_img": dummy_frame, "density_map": dummy_density, "count": 12.5},
        {"orig_img": dummy_frame, "density_map": dummy_density, "count": 18.0}
    ]
    mock_handler_class.return_value = mock_handler

    # 3. Load config and run pipeline
    config = PipelineConfig.from_yaml(rgbt_temp_workspace)
    pipeline = CountingPipeline(config)
    summary = pipeline.run()

    # 4. Assert correctness
    assert summary["processed_frames"] == 2
    assert summary["counts"]["min_people_in_frame"] == 12
    assert summary["counts"]["max_people_in_frame"] == 18
    assert summary["counts"]["mean_people_per_frame"] == 15.25

    # Check generated files
    out_dir = Path(config.paths.output_dir)
    assert (out_dir / "frame_counts.csv").exists()
    assert (out_dir / "summary.json").exists()
