"""
Medallion Data Architecture Pipeline Orchestrator (Landing -> Silver -> Gold)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements the Medallion Data Architecture for RGBT Crowd Counting:
- Landing Zone (Bronze): Ingests raw, unmodified sensor images/videos with original metadata
  from `input/landing/` (formerly `input/images/`).
- Silver Zone: Cleans, enhances (CLAHE), letterboxes, and spatially aligns RGB and Thermal
  imagery via Homography (ADR 001-005). Writes refined layers to `input/silver/` (formerly `input/images_equalized/`).
- Gold Zone: Evaluates crowd density predictions via CountingPipeline and exports decision-ready
  data products (annotated heatmaps, frame_counts.csv, summary.json, MLflow telemetry) to `output/gold/` (formerly `output/`).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Tuple
import cv2

from .config import PipelineConfig
from .preprocessing import RGBTImageEqualizer
from .pipeline import CountingPipeline

logger = logging.getLogger(__name__)


class MedallionPipelineRunner:
    """Orchestrates the 3-stage Medallion Data Architecture pipeline.

    Attributes:
        config: The PipelineConfig instance.
        config_path: Path to the YAML configuration file.
        equalizer: The RGBTImageEqualizer instance for Landing -> Silver processing.
    """

    def __init__(self, config: PipelineConfig, config_path: Path | str | None = None):
        """Initializes MedallionPipelineRunner with system configuration.

        Args:
            config: A PipelineConfig instance.
            config_path: Path to the active YAML config file.
        """
        self.config = config
        self.config_path = config_path

        # Setup Medallion directories
        self.landing_dir = Path(self.config.paths.landing_dir or self.config.paths.bronze_dir)
        self.silver_dir = Path(self.config.paths.silver_dir)
        self.gold_dir = Path(self.config.paths.gold_dir)

        self._ensure_directories()

        # Initialize preprocessor for Silver layer
        prep_cfg = self.config.preprocessing
        self.equalizer = RGBTImageEqualizer(
            target_size=tuple(prep_cfg.target_resolution),
            keep_aspect_ratio=prep_cfg.keep_aspect_ratio,
            thermal_clahe=prep_cfg.thermal_clahe,
            clahe_clip_limit=prep_cfg.clahe_clip_limit,
            mode=prep_cfg.mode,
        )

    def _ensure_directories(self) -> None:
        """Creates the directory structure for Landing (Bronze), Silver, and Gold zones."""
        self.landing_dir.mkdir(parents=True, exist_ok=True)
        (self.silver_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.silver_dir / "layer_blend_checks").mkdir(parents=True, exist_ok=True)
        (self.gold_dir / "heatmaps").mkdir(parents=True, exist_ok=True)
        (self.gold_dir / "telemetry").mkdir(parents=True, exist_ok=True)
        (self.gold_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    def run_silver(self) -> Tuple[Path, Path]:
        """Executes the Landing (Bronze) -> Silver transformation phase.

        Ingests raw media from Landing zone (or configured paths), applies homography alignment,
        letterboxing, and thermal CLAHE, and saves refined equalized layers into Silver zone.

        Returns:
            A tuple of (silver_rgb_path, silver_thermal_path).
        """
        logger.info("============================================================")
        logger.info("       MEDALLION PIPELINE: LANDING -> SILVER TRANSFORMATION ")
        logger.info("============================================================")

        rgb_input = Path(self.config.paths.video_rgb)
        thermal_input = Path(self.config.paths.video_thermal)

        if not rgb_input.exists():
            raise FileNotFoundError(f"Landing RGB input not found: {rgb_input}")

        logger.info(f"[LANDING -> SILVER] Ingesting raw media: RGB='{rgb_input.name}'")

        # 1. Read raw images
        frame_rgb = cv2.imread(str(rgb_input))
        if frame_rgb is None:
            raise RuntimeError(f"Failed to read Landing RGB image: {rgb_input}")

        frame_thermal = None
        if thermal_input and thermal_input.exists():
            frame_thermal = cv2.imread(str(thermal_input))
        
        if frame_thermal is None:
            frame_thermal = frame_rgb.copy()

        # 2. Apply Silver layer equalization (Homography, Letterboxing, CLAHE)
        eq_rgb, eq_thermal = self.equalizer.process_pair(frame_rgb, frame_thermal)

        # 3. Save Silver refined layers
        silver_rgb_path = self.silver_dir / "images" / f"{rgb_input.stem}_equalized{rgb_input.suffix}"
        silver_thermal_path = self.silver_dir / "images" / f"{thermal_input.stem if thermal_input else 'thermal'}_equalized.jpg"
        blend_check_path = self.silver_dir / "layer_blend_checks" / f"{rgb_input.stem}_blend_check.jpg"

        cv2.imwrite(str(silver_rgb_path), eq_rgb)
        cv2.imwrite(str(silver_thermal_path), eq_thermal)

        blend_overlay = self.equalizer.create_blend_overlay(eq_rgb, eq_thermal)
        cv2.imwrite(str(blend_check_path), blend_overlay)

        logger.info(f"[SILVER LAYER CREATED] Aligned RGB Layer:     {silver_rgb_path}")
        logger.info(f"[SILVER LAYER CREATED] Equalized Thermal:    {silver_thermal_path}")
        logger.info(f"[SILVER LAYER CREATED] Blend Check Overlay: {blend_check_path}")

        return silver_rgb_path, silver_thermal_path

    def run_gold(self, silver_rgb_path: Path | str | None = None, silver_thermal_path: Path | str | None = None) -> dict[str, Any]:
        """Executes the Silver -> Gold prediction and analytics phase.

        Feeds Silver aligned layers into CountingPipeline, evaluates crowd density,
        and saves decision-ready outputs into Gold zone.

        Args:
            silver_rgb_path: Optional path to Silver RGB media.
            silver_thermal_path: Optional path to Silver Thermal media.

        Returns:
            A dictionary containing final Gold analytics metrics.
        """
        logger.info("============================================================")
        logger.info("       MEDALLION PIPELINE: SILVER -> GOLD ANALYTICS        ")
        logger.info("============================================================")

        # Override paths to point to Silver layer inputs
        if silver_rgb_path:
            self.config.paths.video_rgb = str(silver_rgb_path)
            self.config.paths.video = str(silver_rgb_path)
        if silver_thermal_path:
            self.config.paths.video_thermal = str(silver_thermal_path)

        # Redirect output targets to Gold zone subfolders
        self.config.paths.output_dir = str(self.gold_dir)
        self.config.paths.annotated_video = str(self.gold_dir / "heatmaps" / "annotated_heatmap.mp4")
        self.config.paths.frame_counts_csv = str(self.gold_dir / "telemetry" / "frame_counts.csv")
        self.config.paths.summary_json = str(self.gold_dir / "telemetry" / "summary.json")
        self.config.paths.snapshots_dir = str(self.gold_dir / "snapshots")

        # Disable further preprocessing in CountingPipeline since Silver layer is already preprocessed
        self.config.preprocessing.enabled = False

        pipeline = CountingPipeline(config=self.config, config_path=self.config_path)
        summary = pipeline.run()

        logger.info(f"[GOLD LAYER CREATED] Density Heatmaps: {self.config.paths.annotated_video}")
        logger.info(f"[GOLD LAYER CREATED] Telemetry CSV:    {self.config.paths.frame_counts_csv}")
        logger.info(f"[GOLD LAYER CREATED] Summary JSON:     {self.config.paths.summary_json}")

        return summary

    def run_all(self) -> dict[str, Any]:
        """Executes full Medallion pipeline: Landing -> Silver -> Gold."""
        silver_rgb, silver_thermal = self.run_silver()
        return self.run_gold(silver_rgb, silver_thermal)
