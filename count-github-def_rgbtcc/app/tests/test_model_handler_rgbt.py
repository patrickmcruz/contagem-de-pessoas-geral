import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from head_counting import PipelineConfig
from head_counting.model import DEFModelHandler


@pytest.fixture
def sample_config():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yaml_file = temp_path / "test_config.yaml"

        config_data = {
            "paths": {
                "video_rgb": "rgb.mp4",
                "video_thermal": "thermal.mp4",
                "weights": "model.pth",
                "weights_search_dirs": [str(temp_path)]
            },
            "runtime": {
                "require_cuda": False,
                "device": 0,
                "batch_size": 2,
                "auto_reduce_batch_on_oom": True
            },
            "inference": {
                "task": "density_count",
                "imgsz": 640,
                "weight_format": "auto"
            }
        }

        import yaml
        with yaml_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = PipelineConfig.from_yaml(yaml_file)
        yield config, temp_path


def test_def_model_handler_weight_resolution_priority(sample_config) -> None:
    """Verifies that weight resolution prioritizes TensorRT (.trt) > SafeTensors > PyTorch (.pth)."""
    config, temp_path = sample_config

    # Create dummy weight files
    pth_file = temp_path / "model.pth"
    trt_file = temp_path / "model_fp16.trt"
    pth_file.touch()

    handler = DEFModelHandler(config)

    # Without TensorRT file, should resolve to model.pth
    assert handler.weights_ref == str(pth_file.resolve())

    # Create TensorRT file and verify priority boost
    trt_file.touch()
    handler_trt = DEFModelHandler(config)
    assert handler_trt.weights_ref == str(trt_file.resolve())


@patch("head_counting.model.RGBTCCInference")
def test_def_model_handler_predict_batch_density(mock_inference_class, sample_config) -> None:
    """Verifies that DEFModelHandler executes predict_batch and returns density map and count dictionary."""
    config, _ = sample_config

    mock_model = MagicMock()
    mock_density = np.full((640, 640), 0.05, dtype=np.float32)
    mock_model.predict.return_value = {"density_map": mock_density, "count": 20.5}
    mock_inference_class.return_value = mock_model

    handler = DEFModelHandler(config)
    handler.load_model()

    dummy_rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_thermal = np.zeros((640, 640, 3), dtype=np.uint8)

    results = handler.predict_batch([dummy_rgb, dummy_rgb], [dummy_thermal, dummy_thermal])

    assert len(results) == 2
    assert results[0]["count"] == 20.5
    assert results[0]["density_map"].shape == (640, 640)
    assert np.array_equal(results[0]["orig_img"], dummy_rgb)


def test_def_model_handler_empty_batch(sample_config) -> None:
    """Verifies that predict_batch returns empty list when given empty input."""
    config, _ = sample_config
    handler = DEFModelHandler(config)
    results = handler.predict_batch([])
    assert results == []
