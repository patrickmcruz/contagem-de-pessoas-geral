import tempfile
from pathlib import Path
import yaml
import pytest

from head_counting import PipelineConfig


def test_rgbt_config_parsing_and_resolution() -> None:
    """Verifies that PipelineConfig parses dual-stream RGB-T configuration fields and resolves paths correctly."""
    config_data = {
        "app": {
            "name": "test-rgbt-app",
            "seed": 123
        },
        "paths": {
            "video_rgb": "input/videos/rgb.mp4",
            "video_thermal": "input/videos/thermal.mp4",
            "weights": "weights/model.pth",
            "output_dir": "custom_rgbt_out"
        },
        "runtime": {
            "batch_size": 4,
            "device": 0
        },
        "inference": {
            "task": "density_count",
            "weight_format": "safetensors",
            "imgsz": 640,
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_std": [0.229, 0.224, 0.225],
            "thermal_mean": [0.5, 0.5, 0.5],
            "thermal_std": [0.5, 0.5, 0.5]
        },
        "output": {
            "save_density_heatmap": True,
            "heatmap_colormap": "JET",
            "heatmap_alpha": 0.5
        }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        yaml_file = temp_dir_path / "test_rgbt_config.yaml"
        
        with yaml_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = PipelineConfig.from_yaml(yaml_file)

        # Basic assertions
        assert config.app.name == "test-rgbt-app"
        assert config.app.seed == 123
        assert config.inference.task == "density_count"
        assert config.inference.weight_format == "safetensors"
        assert config.inference.imgsz == 640
        assert config.counting.count_source == "density_map"

        # Heatmap assertions
        assert config.output.save_density_heatmap is True
        assert config.output.heatmap_colormap == "JET"
        assert config.output.heatmap_alpha == 0.5

        # Normalization assertions
        assert config.inference.rgb_mean == [0.485, 0.456, 0.406]
        assert config.inference.thermal_mean == [0.5, 0.5, 0.5]

        # Path resolution assertions
        assert Path(config.paths.video_rgb).is_absolute()
        assert Path(config.paths.video_rgb) == (temp_dir_path / "input/videos/rgb.mp4").resolve()

        assert Path(config.paths.video_thermal).is_absolute()
        assert Path(config.paths.video_thermal) == (temp_dir_path / "input/videos/thermal.mp4").resolve()

        assert Path(config.paths.weights).is_absolute()
        assert Path(config.paths.weights) == (temp_dir_path / "weights/model.pth").resolve()


def test_rgbt_config_fallback_single_video() -> None:
    """Verifies that legacy single video field falls back correctly to video_rgb."""
    config_data = {
        "paths": {
            "video": "input/videos/legacy.mp4"
        }
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        yaml_file = temp_dir_path / "legacy_config.yaml"
        
        with yaml_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = PipelineConfig.from_yaml(yaml_file)
        assert Path(config.paths.video_rgb) == (temp_dir_path / "input/videos/legacy.mp4").resolve()
        assert Path(config.paths.video) == (temp_dir_path / "input/videos/legacy.mp4").resolve()


def test_rgbt_day_night_yaml_parsing() -> None:
    """Verifies that default day and night YAML config files load properly."""
    app_dir = Path(__file__).parent.parent
    day_yaml = app_dir / "data_rgbt_day.yaml"
    night_yaml = app_dir / "data_rgbt_night.yaml"

    if day_yaml.exists():
        config_day = PipelineConfig.from_yaml(day_yaml)
        assert config_day.inference.task == "density_count"
        assert config_day.output.heatmap_colormap == "JET"

    if night_yaml.exists():
        config_night = PipelineConfig.from_yaml(night_yaml)
        assert config_night.inference.task == "density_count"
        assert config_night.output.heatmap_colormap == "INFERNO"
