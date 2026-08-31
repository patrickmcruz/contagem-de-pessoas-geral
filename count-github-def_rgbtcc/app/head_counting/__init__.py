"""
Head Counting Package
~~~~~~~~~~~~~~~~~~~~

A modular, high-performance library for real-time human head detection and crowd counting.
This package encapsulates:
- Config parsing with type-safe dataclasses (config.py)
- YOLO / DEF-rgbtcc model management, hardware routing, and batch inference (model.py)
- Thread-safe, multi-threaded video reader and writer queues (video.py)
- Decoupled RGBT image equalization & homography layer alignment (preprocessing.py)
- Medallion Data Architecture (Bronze -> Silver -> Gold) orchestrator (medallion.py)
- End-to-end pipeline coordination and metric aggregation (pipeline.py)

Usage:
    from head_counting import run_pipeline, MedallionPipelineRunner
    summary = run_pipeline("data_rgbt_images.yaml")
"""

from .config import (
    PipelineConfig,
    AppConfig,
    EnvConfig,
    PathsConfig,
    RuntimeConfig,
    InferenceConfig,
    PreprocessingConfig,
    CountingConfig,
    OutputConfig,
    OverlayConfig,
    MLflowConfig,
)
from .model import YOLOModelHandler
from .video import VideoReader, VideoWriterWrapper, DualStreamVideoReader, DualStreamVideoWriterWrapper
from .preprocessing import RGBTImageEqualizer
from .medallion import MedallionPipelineRunner
from .tracker import MLflowTracker
from .pipeline import CountingPipeline, run_pipeline

__all__ = [
    "PipelineConfig",
    "AppConfig",
    "EnvConfig",
    "PathsConfig",
    "RuntimeConfig",
    "InferenceConfig",
    "PreprocessingConfig",
    "CountingConfig",
    "OutputConfig",
    "OverlayConfig",
    "MLflowConfig",
    "YOLOModelHandler",
    "VideoReader",
    "VideoWriterWrapper",
    "DualStreamVideoReader",
    "DualStreamVideoWriterWrapper",
    "RGBTImageEqualizer",
    "MedallionPipelineRunner",
    "MLflowTracker",
    "CountingPipeline",
    "run_pipeline",
]
