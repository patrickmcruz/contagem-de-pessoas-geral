"""
Head Counting Package
~~~~~~~~~~~~~~~~~~~~

A modular, high-performance library for real-time human head detection and crowd counting.
This package encapsulates:
- Config parsing with type-safe dataclasses (config.py)
- YOLO model management, hardware routing, and batch inference (model.py)
- Thread-safe, multi-threaded video reader and writer queues (video.py)
- End-to-end pipeline coordination and metric aggregation (pipeline.py)

Usage:
    from head_counting import run_pipeline
    summary = run_pipeline("data_day.yaml")
"""

from .config import PipelineConfig, AppConfig, EnvConfig, PathsConfig, RuntimeConfig, InferenceConfig, CountingConfig, OutputConfig, OverlayConfig, MLflowConfig
from .model import YOLOModelHandler
from .video import VideoReader, VideoWriterWrapper
from .tracker import MLflowTracker
from .pipeline import CountingPipeline, run_pipeline

__all__ = [
    "PipelineConfig",
    "AppConfig",
    "EnvConfig",
    "PathsConfig",
    "RuntimeConfig",
    "InferenceConfig",
    "CountingConfig",
    "OutputConfig",
    "OverlayConfig",
    "MLflowConfig",
    "YOLOModelHandler",
    "VideoReader",
    "VideoWriterWrapper",
    "MLflowTracker",
    "CountingPipeline",
    "run_pipeline",
]
