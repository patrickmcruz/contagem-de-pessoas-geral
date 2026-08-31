"""
Pipeline Orchestration Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module acts as the core coordinator for the head counting pipeline. It sets up
environment variables, initializes model handlers and video streams, runs the main
prediction-aggregation loop, writes output reports (CSV, JSON), and ensures graceful
resource cleanup upon termination or exceptions.
"""

from __future__ import annotations
import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any
import numpy as np
import statistics

from .config import PipelineConfig
from .model import YOLOModelHandler
from .tracker import MLflowTracker
from .video import VideoReader, VideoWriterWrapper

logger = logging.getLogger(__name__)


def format_seconds(seconds: float) -> str:
    """Formats a duration in seconds into HH:MM:SS format.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A formatted string (e.g. '00:02:15').
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class CountingPipeline:
    """Orchestrates environment setup, model execution, video multithreading, and report exports.

    Attributes:
        config: Loaded system PipelineConfig configuration.
        config_path: Path to the loaded YAML configuration file.
        model_handler: Handler managing the YOLO model lifecycle and predictions.
        tracker: MLflowTracker manager handling experiment logs and metrics.
        counts: List of integers storing head count history frame-by-frame.
        video_meta: Cached metadata profile of the active video stream.
        summary: Result summaries compiled at the end of the run.
    """

    def __init__(self, config: PipelineConfig, config_path: Path | str | None = None):
        """Initializes the pipeline runner.

        Args:
            config: A PipelineConfig instance.
            config_path: Optional path to the active YAML config file.
        """
        self.config = config
        self.config_path = config_path
        self.model_handler = YOLOModelHandler(config)
        self.tracker = MLflowTracker(config)
        self.counts: list[int] = []
        self.video_meta: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}

    def _apply_environment(self) -> None:
        """Applies configured environment variables, ensuring directory paths exist.

        Sets up paths for YOLO, Matplotlib, and PyTorch home cache folders.
        Also configures random seeds for Python, NumPy, and PyTorch.
        """
        env_vars = {
            "YOLO_CONFIG_DIR": self.config.environment.yolo_config_dir,
            "MPLCONFIGDIR": self.config.environment.mpl_config_dir,
            "TORCH_HOME": self.config.environment.torch_home,
        }

        for var_name, path_str in env_vars.items():
            if path_str:
                p = Path(path_str)
                # Resolve relative to config directory if not absolute
                if not p.is_absolute():
                    p = self.config.config_dir / p
                p.mkdir(parents=True, exist_ok=True)
                os.environ[var_name] = str(p.resolve())
                logger.info(f"Environment variable set: {var_name}={p.resolve()}")

        # Set seeding
        seed = self.config.app.seed
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed initialized to: {seed}")

    def _build_summary(self, elapsed_sec: float, processed_frames: int) -> dict[str, Any]:
        """Compiles population statistics and execution metrics into a dictionary summary.

        Calculates min, max, average, median, and 95th percentile metrics for crowd
        sizes, alongside execution throughput speeds (FPS).

        Args:
            elapsed_sec: Processing time in seconds.
            processed_frames: Total count of frames successfully evaluated.

        Returns:
            A dictionary containing consolidated analytics summary metrics.
        """
        if self.counts:
            p95 = float(np.percentile(np.asarray(self.counts), 95))
            summary_counts = {
                "min_people_in_frame": int(min(self.counts)),
                "max_people_in_frame": int(max(self.counts)),
                "mean_people_per_frame": round(float(statistics.fmean(self.counts)), 3),
                "median_people_per_frame": round(float(statistics.median(self.counts)), 3),
                "p95_people_per_frame": round(p95, 3),
            }
        else:
            summary_counts = {
                "min_people_in_frame": 0,
                "max_people_in_frame": 0,
                "mean_people_per_frame": 0.0,
                "median_people_per_frame": 0.0,
                "p95_people_per_frame": 0.0,
            }

        return {
            "app": self.config.app.name,
            "video": self.config.paths.video,
            "weights": self.model_handler.weights_ref,
            "device": self.config.runtime.device,
            "video_meta": self.video_meta,
            "processed_frames": processed_frames,
            "elapsed_sec": round(float(elapsed_sec), 3),
            "fps_processed": round(processed_frames / elapsed_sec, 3) if elapsed_sec > 0 else 0.0,
            "inference": self.model_handler.predict_args,
            "counts": summary_counts,
            "outputs": {
                "annotated_video": self.config.paths.annotated_video,
                "frame_counts_csv": self.config.paths.frame_counts_csv,
                "summary_json": self.config.paths.summary_json,
                "snapshots_dir": self.config.paths.snapshots_dir,
            },
        }

    def run(self) -> dict[str, Any]:
        """Main execution flow coordinating readers, batch processors, and writers.

        Manages context lifecycles for VideoReader, VideoWriter, and MLflowTracker,
        handles batch assembly, triggers model evaluations, and writes output files.

        Returns:
            A dictionary containing consolidated analytics summary metrics.
        """
        # 1. Apply environment configuration and seed
        self._apply_environment()

        # 2. Setup system runtime variables & optimization flags
        self.model_handler.setup_runtime()

        # 3. Load YOLO model into CUDA device
        self.model_handler.load_model()

        # Ensure output directories exist
        out_dir = Path(self.config.paths.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with self.tracker.start_run(config_path=self.config_path):
            # 4. Initialize multithreaded readers
            vid_stride = self.config.inference.vid_stride
            batch_size = self.config.runtime.batch_size

            logger.info(f"Opening video file: {self.config.paths.video}")
            with VideoReader(self.config.paths.video, stride=vid_stride) as reader:
                self.video_meta = reader.metadata
                logger.info(f"Video metadata extracted: {json.dumps(self.video_meta)}")

                expected_processed = (self.video_meta["frame_count"] + vid_stride - 1) // vid_stride

                from tqdm.auto import tqdm
                start_time = time.time()

                # Create progress bar
                with tqdm(total=expected_processed, desc="Processing Video", unit="frame") as pbar:
                    self.pbar = pbar

                    # 5. Initialize multithreaded writers
                    with VideoWriterWrapper(self.config, self.video_meta) as writer:
                        # 6. Setup CSV writer if enabled
                        csv_file = None
                        csv_writer = None
                        if self.config.output.save_frame_counts:
                            csv_path = Path(self.config.paths.frame_counts_csv)
                            csv_path.parent.mkdir(parents=True, exist_ok=True)
                            csv_file = csv_path.open("w", newline="", encoding="utf-8")
                            csv_writer = csv.DictWriter(
                                csv_file, fieldnames=["frame_index", "timestamp_sec", "people_count"]
                            )
                            csv_writer.writeheader()

                        batch_frames = []
                        batch_indices = []

                        try:
                            for frame, frame_idx in reader.iter_frames():
                                batch_frames.append(frame)
                                batch_indices.append(frame_idx)

                                if len(batch_frames) >= batch_size:
                                    self._process_and_write_batch(
                                        batch_frames,
                                        batch_indices,
                                        writer,
                                        csv_writer,
                                    )

                            # Process remaining frames
                            if batch_frames:
                                self._process_and_write_batch(
                                    batch_frames,
                                    batch_indices,
                                    writer,
                                    csv_writer,
                                )

                            # Set progress bar total to actual decodable frames on successful completion
                            pbar.total = len(self.counts)
                            pbar.refresh()

                        finally:
                            if csv_file is not None:
                                csv_file.close()

                    elapsed_sec = time.time() - start_time
                    self.summary = self._build_summary(elapsed_sec, len(self.counts))

                    # 7. Write stats summary JSON report
                    if self.config.output.save_summary:
                        summary_path = Path(self.config.paths.summary_json)
                        summary_path.parent.mkdir(parents=True, exist_ok=True)
                        with summary_path.open("w", encoding="utf-8") as f:
                            json.dump(self.summary, f, indent=2)

                    # 8. Log metrics & artifacts to MLflow
                    self.tracker.log_summary_metrics(self.summary)
                    self.tracker.log_artifacts(self.summary)

                    print("\n" + "=" * 60)
                    print("PROCESSAMENTO FINALIZADO!")
                    print(f"Média final: {self.summary['fps_processed']} FPS")
                    print("=" * 60)

                    return self.summary

    def _process_and_write_batch(
        self,
        batch_frames: list[np.ndarray],
        batch_indices: list[int],
        writer: VideoWriterWrapper,
        csv_writer: Any | None,
    ) -> None:
        """Runs batch inference, collects statistics, and enqueues frames to writing threads.

        Args:
            batch_frames: Frames representing the active chunk batch.
            batch_indices: Frame index list for references.
            writer: Background VideoWriterWrapper queue handler.
            csv_writer: CSV writer interface for saving count statistics.
        """
        frames = list(batch_frames)
        indices = list(batch_indices)
        batch_frames.clear()
        batch_indices.clear()

        # Fast GPU prediction
        results = self.model_handler.predict_batch(frames)

        fps = self.video_meta.get("fps", 30.0)

        for result, frame_idx in zip(results, indices):
            timestamp_sec = frame_idx / fps if fps > 0 else 0.0
            
            # Extract count based on boxes (head counting specializes in boxes)
            count = len(result.boxes) if result.boxes is not None else 0
            self.counts.append(count)

            # Log step-level metric to MLflow
            self.tracker.log_step_metric("people_count", count, step=frame_idx)

            # Write row to CSV
            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "people_count": count,
                    }
                )

            # Enqueue task for background annotation and writing
            writer.write_result(result, count, frame_idx, timestamp_sec)

        # Update tqdm progress bar
        if hasattr(self, "pbar") and self.pbar is not None:
            self.pbar.update(len(results))


def run_pipeline(config_path: Path | str) -> dict[str, Any]:
    """Helper entrypoint to easily load config and execute the pipeline.

    Args:
        config_path: Path pointing to the target YAML config file.

    Returns:
        A dictionary containing consolidated analytics summary metrics.
    """
    config = PipelineConfig.from_yaml(config_path)
    pipeline = CountingPipeline(config, config_path=config_path)
    return pipeline.run()
