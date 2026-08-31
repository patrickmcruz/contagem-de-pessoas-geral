"""
MLflow Tracking Manager Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides the MLflowTracker class, managing experiment initialization,
context lifecycle, parameter logging, step-level time series metrics, summary metrics,
and artifact uploads for the head counting pipeline.
"""

from __future__ import annotations
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Dict

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Manages MLflow experiment tracking lifecycle, parameter logging, metrics, and artifacts.

    Attributes:
        config: Loaded system PipelineConfig configuration.
        is_active: Boolean indicating whether an active MLflow run context is running.
        run: The active MLflow run object (if active).
    """

    def __init__(self, config: PipelineConfig):
        """Initializes the MLflow tracker.

        Args:
            config: A PipelineConfig instance.
        """
        self.config = config
        self.mlflow_cfg = config.mlflow
        self.is_active: bool = False
        self.run: Any | None = None

    @contextmanager
    def start_run(self, config_path: Path | str | None = None) -> Generator[MLflowTracker, None, None]:
        """Context manager to start and automatically end an MLflow run safely.

        Args:
            config_path: Optional path to the active YAML config file for artifact logging.

        Yields:
            The active MLflowTracker instance.
        """
        if not self.mlflow_cfg.enabled:
            logger.info("MLflow tracking is disabled in config.")
            yield self
            return

        try:
            import mlflow

            # Set tracking URI and experiment
            mlflow.set_tracking_uri(self.mlflow_cfg.tracking_uri)
            mlflow.set_experiment(self.mlflow_cfg.experiment_name)

            # Optional Ultralytics autologging callback
            if self.mlflow_cfg.autolog_ultralytics:
                try:
                    mlflow.ultralytics.autolog()
                    logger.info("Ultralytics MLflow autologging activated.")
                except Exception as auto_err:
                    logger.warning(f"Could not enable Ultralytics autologging: {auto_err}")

            run_name = self.mlflow_cfg.run_name or self.config.app.name
            self.run = mlflow.start_run(run_name=run_name)
            self.is_active = True
            logger.info(f"MLflow Run started: ID={self.run.info.run_id}, Name='{run_name}'")

            # Set system and user tags
            tags = {
                "app.name": self.config.app.name,
                "model.weights": Path(self.config.paths.weights).name,
                "runtime.device": str(self.config.runtime.device),
            }
            tags.update(self.mlflow_cfg.tags)
            mlflow.set_tags(tags)

            # Log flattened configuration parameters
            if self.mlflow_cfg.log_params:
                self.log_parameters()

            # Log YAML config file as artifact if available
            if config_path and Path(config_path).exists():
                try:
                    mlflow.log_artifact(str(Path(config_path).resolve()), artifact_path="config")
                except Exception as artifact_err:
                    logger.warning(f"Failed to log config YAML artifact: {artifact_err}")

            yield self

        except Exception as e:
            logger.warning(f"[MLFLOW WARNING] Unable to connect or track run in MLflow: {e}. "
                           "Continuing execution without MLflow tracking.")
            self.is_active = False
            self.run = None
            yield self
        finally:
            if self.is_active:
                try:
                    import mlflow
                    mlflow.end_run()
                    logger.info("MLflow Run completed successfully.")
                except Exception as end_err:
                    logger.warning(f"Error ending MLflow run: {end_err}")
                finally:
                    self.is_active = False
                    self.run = None

    def log_parameters(self) -> None:
        """Flattens and logs system configuration parameters into MLflow."""
        if not self.is_active:
            return

        try:
            import mlflow

            params: Dict[str, Any] = {
                # App params
                "app.name": self.config.app.name,
                "app.seed": self.config.app.seed,
                # Runtime params
                "runtime.device": self.config.runtime.device,
                "runtime.batch_size": self.config.runtime.batch_size,
                "runtime.allow_tf32": self.config.runtime.allow_tf32,
                "runtime.torch_float32_matmul_precision": self.config.runtime.torch_float32_matmul_precision,
                "runtime.auto_reduce_batch_on_oom": self.config.runtime.auto_reduce_batch_on_oom,
                # Inference params
                "inference.task": self.config.inference.task,
                "inference.imgsz": self.config.inference.imgsz,
                "inference.conf": self.config.inference.conf,
                "inference.iou": self.config.inference.iou,
                "inference.max_det": self.config.inference.max_det,
                "inference.classes": str(self.config.inference.classes),
                "inference.augment": self.config.inference.augment,
                "inference.half": self.config.inference.half,
                "inference.vid_stride": self.config.inference.vid_stride,
                # Counting params
                "counting.count_source": self.config.counting.count_source,
                "counting.require_keypoints": self.config.counting.require_keypoints,
                # Path params
                "paths.weights": Path(self.config.paths.weights).name,
                "paths.video": Path(self.config.paths.video).name if self.config.paths.video else "",
            }

            mlflow.log_params(params)
            logger.info("MLflow parameters logged successfully.")
        except Exception as e:
            logger.warning(f"Failed to log parameters to MLflow: {e}")

    def log_step_metric(self, key: str, value: float | int, step: int) -> None:
        """Logs a single step-level time series metric to MLflow.

        Args:
            key: Metric name (e.g., 'people_count').
            value: Numerical metric value.
            step: Step index (e.g., frame_index).
        """
        if not self.is_active or not self.mlflow_cfg.log_step_metrics:
            return

        try:
            import mlflow
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.debug(f"Failed to log step metric '{key}' at step {step}: {e}")

    def log_summary_metrics(self, summary: Dict[str, Any]) -> None:
        """Logs aggregated performance and population analytics summary metrics to MLflow.

        Args:
            summary: Summary report dictionary compiled by CountingPipeline.
        """
        if not self.is_active:
            return

        try:
            import mlflow

            counts = summary.get("counts", {})
            metrics = {
                "processed_frames": summary.get("processed_frames", 0),
                "elapsed_sec": summary.get("elapsed_sec", 0.0),
                "fps_processed": summary.get("fps_processed", 0.0),
                "min_people_in_frame": counts.get("min_people_in_frame", 0),
                "max_people_in_frame": counts.get("max_people_in_frame", 0),
                "mean_people_per_frame": counts.get("mean_people_per_frame", 0.0),
                "median_people_per_frame": counts.get("median_people_per_frame", 0.0),
                "p95_people_per_frame": counts.get("p95_people_per_frame", 0.0),
            }

            mlflow.log_metrics(metrics)
            logger.info("MLflow summary metrics logged successfully.")
        except Exception as e:
            logger.warning(f"Failed to log summary metrics to MLflow: {e}")

    def log_artifacts(self, summary: Dict[str, Any]) -> None:
        """Uploads generated output reports and video assets to MLflow artifact storage.

        Args:
            summary: Summary report dictionary containing output file paths.
        """
        if not self.is_active or not self.mlflow_cfg.log_artifacts:
            return

        try:
            import mlflow

            outputs = summary.get("outputs", {})

            # 1. Log summary JSON
            summary_json = outputs.get("summary_json")
            if summary_json and Path(summary_json).exists():
                mlflow.log_artifact(str(Path(summary_json).resolve()), artifact_path="reports")

            # 2. Log frame counts CSV
            frame_counts_csv = outputs.get("frame_counts_csv")
            if frame_counts_csv and Path(frame_counts_csv).exists():
                mlflow.log_artifact(str(Path(frame_counts_csv).resolve()), artifact_path="reports")

            # 3. Log annotated video artifact if enabled
            if self.mlflow_cfg.log_annotated_video:
                annotated_video = outputs.get("annotated_video")
                if annotated_video and Path(annotated_video).exists():
                    logger.info(f"Uploading annotated video artifact to MLflow: {annotated_video}")
                    mlflow.log_artifact(str(Path(annotated_video).resolve()), artifact_path="videos")

            # 4. Log model artifact if enabled
            if self.mlflow_cfg.log_model:
                weights_path = summary.get("weights")
                if weights_path and Path(weights_path).exists():
                    logger.info(f"Uploading model weights artifact to MLflow: {weights_path}")
                    mlflow.log_artifact(str(Path(weights_path).resolve()), artifact_path="weights")

            logger.info("MLflow artifacts uploaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to upload artifacts to MLflow: {e}")
