import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import yaml

from head_counting import PipelineConfig, CountingPipeline


@pytest.fixture
def temp_workspace():
    """Sets up a temporary directory with a dummy video and config file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create dummy video path (will be mocked anyway)
        video_path = temp_path / "dummy_video.mp4"
        video_path.touch()
        
        config_data = {
            "app": {
                "name": "test-pipeline-run"
            },
            "paths": {
                "video": str(video_path),
                "weights": "dummy_weights.pt",
                "output_dir": str(temp_path / "output")
            },
            "runtime": {
                "require_cuda": False,  # Run on CPU for testing
                "batch_size": 2
            },
            "inference": {
                "task": "detect",
                "imgsz": 640,
                "conf": 0.25
            },
            "output": {
                "save_annotated_video": False, # Avoid creating actual video writer
                "save_frame_counts": True,
                "save_summary": True,
                "save_snapshot_every_n_frames": 0 # Disable snapshots
            }
        }
        
        config_file = temp_path / "data.yaml"
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f)
            
        yield config_file


@patch("head_counting.model.YOLO")
@patch("head_counting.video.cv2.VideoCapture")
def test_pipeline_execution_flow(mock_capture_class, mock_yolo_class, temp_workspace) -> None:
    """Verifies that the CountingPipeline completes successfully, calling the model and writing outputs."""
    # 1. Mock Video Capture behaviour
    mock_cap = MagicMock()
    mock_cap.isOpened.side_effect = [True, True, True, False]  # Open, then read 2 frames, then end
    
    # Return 2 dummy frames, then stop
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, dummy_frame),
        (True, dummy_frame),
        (False, None)
    ]
    
    # Metadata calls
    mock_cap.get.side_effect = [
        100,  # width
        100,  # height
        30.0, # fps
        2     # frame_count
    ]
    mock_capture_class.return_value = mock_cap

    # 2. Mock YOLO Predict behavior
    mock_yolo = MagicMock()
    mock_yolo.task = "detect"
    
    # Mock result box objects
    mock_box = MagicMock()
    mock_box.xyxy = np.array([[10, 10, 20, 20]])
    
    mock_result_1 = MagicMock()
    mock_result_1.boxes = [mock_box]  # 1 detection
    mock_result_1.orig_img = dummy_frame
    
    mock_result_2 = MagicMock()
    mock_result_2.boxes = [mock_box, mock_box]  # 2 detections
    mock_result_2.orig_img = dummy_frame
    
    mock_yolo.predict.return_value = [mock_result_1, mock_result_2]
    mock_yolo_class.return_value = mock_yolo

    # 3. Load config and run pipeline
    config = PipelineConfig.from_yaml(temp_workspace)
    pipeline = CountingPipeline(config)
    
    # Execute
    summary = pipeline.run()

    # 4. Assert correctness
    assert summary["processed_frames"] == 2
    assert summary["counts"]["min_people_in_frame"] == 1
    assert summary["counts"]["max_people_in_frame"] == 2
    assert summary["counts"]["mean_people_per_frame"] == 1.5

    # Check outputs generated
    out_dir = Path(config.paths.output_dir)
    assert (out_dir / "frame_counts.csv").exists()
    assert (out_dir / "summary.json").exists()
