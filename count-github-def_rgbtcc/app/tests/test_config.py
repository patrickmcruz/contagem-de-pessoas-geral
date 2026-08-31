import tempfile
from pathlib import Path
import yaml
import pytest

from head_counting import PipelineConfig


def test_config_parsing_and_resolution() -> None:
    """Verifies that PipelineConfig parses raw YAML configurations correctly and resolves relative paths."""
    config_data = {
        "app": {
            "name": "test-app",
            "seed": 99
        },
        "paths": {
            "video": "relative/path/to/video.mp4",
            "weights": "relative/path/to/weights.pt",
            "weights_search_dirs": [
                "some/search/dir",
                "/absolute/search/dir"
            ],
            "output_dir": "custom_output"
        },
        "runtime": {
            "batch_size": 8,
            "device": 1
        },
        "inference": {
            "conf": 0.25,
            "imgsz": 640
        }
    }

    # Create a temporary YAML config file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        yaml_file = temp_dir_path / "test_config.yaml"
        
        with yaml_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Load config
        config = PipelineConfig.from_yaml(yaml_file)

        # Assert values
        assert config.app.name == "test-app"
        assert config.app.seed == 99
        assert config.runtime.batch_size == 8
        assert config.runtime.device == 1
        assert config.inference.conf == 0.25
        assert config.inference.imgsz == 640

        # Assert resolved absolute paths
        assert Path(config.paths.video).is_absolute()
        assert Path(config.paths.video) == (temp_dir_path / "relative/path/to/video.mp4").resolve()
        
        assert Path(config.paths.weights).is_absolute()
        assert Path(config.paths.weights) == (temp_dir_path / "relative/path/to/weights.pt").resolve()

        # Weights search dirs resolution
        assert len(config.paths.weights_search_dirs) == 2
        assert Path(config.paths.weights_search_dirs[0]) == (temp_dir_path / "some/search/dir").resolve()
        # The absolute path should remain absolute
        assert Path(config.paths.weights_search_dirs[1]) == Path("/absolute/search/dir").resolve()

        # Check default derived paths
        expected_csv = (temp_dir_path / "custom_output" / "frame_counts.csv").resolve()
        assert Path(config.paths.frame_counts_csv) == expected_csv


def test_missing_config_file_raises_error() -> None:
    """Verifies that an error is raised when trying to load a non-existent configuration file."""
    with pytest.raises(FileNotFoundError):
        PipelineConfig.from_yaml("non_existent_file.yaml")
