import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import cv2

from head_counting import PipelineConfig
from head_counting.video import DualStreamVideoReader, DualStreamVideoWriterWrapper


@patch("head_counting.video.cv2.VideoCapture")
def test_dual_stream_video_reader_metadata_and_sync(mock_capture_class) -> None:
    """Verifies that DualStreamVideoReader opens both RGB and Thermal feeds and extracts metadata."""
    # Mock RGB capture
    mock_cap_rgb = MagicMock()
    mock_cap_rgb.isOpened.return_value = True
    mock_cap_rgb.get.side_effect = [640, 480, 30.0, 100]  # w, h, fps, count
    
    # Mock Thermal capture
    mock_cap_thermal = MagicMock()
    mock_cap_thermal.isOpened.return_value = True
    mock_cap_thermal.get.side_effect = [640, 480, 30.0, 100]  # w, h, fps, count

    mock_capture_class.side_effect = [mock_cap_rgb, mock_cap_thermal]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        rgb_file = temp_path / "video_rgb.mp4"
        thermal_file = temp_path / "video_thermal.mp4"
        rgb_file.touch()
        thermal_file.touch()

        reader = DualStreamVideoReader(video_rgb_path=rgb_file, video_thermal_path=thermal_file)
        
        assert reader.metadata["rgb"]["width"] == 640
        assert reader.metadata["thermal"]["width"] == 640
        assert reader.metadata["fps"] == 30.0
        assert reader.metadata["frame_count"] == 100


@patch("head_counting.video.cv2.VideoCapture")
def test_dual_stream_video_reader_iter_frames(mock_capture_class) -> None:
    """Verifies that DualStreamVideoReader yields synchronized (frame_rgb, frame_thermal, frame_idx)."""
    dummy_rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
    dummy_thermal = np.full((100, 100, 3), 100, dtype=np.uint8)

    mock_cap_rgb = MagicMock()
    mock_cap_rgb.isOpened.side_effect = [True, True, True, True, False]
    mock_cap_rgb.get.side_effect = [100, 100, 30.0, 2]
    mock_cap_rgb.read.side_effect = [(True, dummy_rgb), (True, dummy_rgb), (False, None)]

    mock_cap_thermal = MagicMock()
    mock_cap_thermal.isOpened.side_effect = [True, True, True, True, False]
    mock_cap_thermal.get.side_effect = [100, 100, 30.0, 2]
    mock_cap_thermal.read.side_effect = [(True, dummy_thermal), (True, dummy_thermal), (False, None)]

    mock_capture_class.side_effect = [mock_cap_rgb, mock_cap_thermal, mock_cap_rgb, mock_cap_thermal]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        rgb_file = temp_path / "video_rgb.mp4"
        thermal_file = temp_path / "video_thermal.mp4"
        rgb_file.touch()
        thermal_file.touch()

        reader = DualStreamVideoReader(video_rgb_path=rgb_file, video_thermal_path=thermal_file)
        
        frames = []
        with reader:
            for frame_rgb, frame_thermal, idx in reader.iter_frames():
                frames.append((frame_rgb, frame_thermal, idx))

        assert len(frames) == 2
        assert frames[0][2] == 0
        assert frames[1][2] == 1
        assert np.array_equal(frames[0][0], dummy_rgb)
        assert np.array_equal(frames[0][1], dummy_thermal)


@patch("head_counting.video.cv2.VideoWriter")
def test_dual_stream_video_writer_heatmap_rendering(mock_writer_class) -> None:
    """Verifies that DualStreamVideoWriterWrapper converts density maps into colored heatmaps and writes video frames."""
    mock_writer = MagicMock()
    mock_writer_class.return_value = mock_writer

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yaml_file = temp_path / "test_config.yaml"

        config_data = {
            "paths": {
                "video_rgb": "rgb.mp4",
                "output_dir": str(temp_path / "out")
            },
            "output": {
                "save_annotated_video": True,
                "save_density_heatmap": True,
                "heatmap_colormap": "JET",
                "heatmap_alpha": 0.5,
                "save_snapshot_every_n_frames": 0
            }
        }

        import yaml
        with yaml_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        config = PipelineConfig.from_yaml(yaml_file)
        video_meta = {"width": 100, "height": 100, "fps": 30.0}

        writer = DualStreamVideoWriterWrapper(config, video_meta)
        
        dummy_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_density = np.random.rand(100, 100).astype(np.float32)

        result_dict = {
            "orig_img": dummy_rgb,
            "density_map": dummy_density,
            "count": 14.5
        }

        with writer:
            writer.write_result(result_dict, count=14.5, frame_idx=0, timestamp_sec=0.0)

        mock_writer.write.assert_called_once()
