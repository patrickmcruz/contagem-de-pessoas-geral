"""
Configuration Module
~~~~~~~~~~~~~~~~~~~~

This module provides dataclass schemas and parsing methods to load, validate, and
resolve system paths for the head counting pipeline. Path parameters are dynamically
resolved relative to the directory containing the active YAML configuration file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class AppConfig:
    """General application metadata configurations.

    Attributes:
        name: Name of the application setup for logs and reports.
        seed: Random seed for reproducibility in sampling and shuffling operations.
    """
    name: str = "head-counting-pipeline"
    seed: int = 42


@dataclass
class EnvConfig:
    """Local cache and configuration folders for machine learning libraries.

    Attributes:
        yolo_config_dir: Directory where YOLO keeps local parameters.
        mpl_config_dir: Cache folder for Matplotlib parameters.
        torch_home: Directory where PyTorch downloads pre-trained weights.
    """
    yolo_config_dir: str = ".yolo"
    mpl_config_dir: str = ".cache/matplotlib"
    torch_home: str = ".cache/torch"


@dataclass
class PathsConfig:
    """Input and output asset file paths. Relative paths are resolved to absolute.

    Attributes:
        video: Path to the input MP4/AVI video file (legacy / fallback single channel).
        video_rgb: Path to the input RGB MP4/AVI video file.
        video_thermal: Path to the input Thermal/Infrared MP4/AVI video file.
        images: Glob pattern for input images (legacy / fallback).
        images_rgb: Directory pattern for RGB input images.
        images_thermal: Directory pattern for Thermal input images.
        weights: Path to local DEF-rgbtcc weights (.pth, .safetensors, .onnx, .trt, .engine).
        weights_search_dirs: Directory paths searched in order to locate model weights.
        output_dir: Main folder where output files are created.
        annotated_video: Target path for the output annotated MP4 video.
        frame_counts_csv: Target path for the frame-by-frame counts CSV table.
        summary_json: Target path for the final executive summary JSON report.
        snapshots_dir: Directory where JPEG snapshots of selected frames are stored.
    """
    video: str = ""
    video_rgb: str = ""
    video_thermal: str = ""
    images: str = ""
    images_rgb: str = ""
    images_thermal: str = ""
    weights: str = "model.pth"
    weights_search_dirs: list[str] = field(default_factory=list)
    output_dir: str = "data/gold"
    bronze_dir: str = "data/bronze"
    silver_dir: str = "data/silver"
    gold_dir: str = "data/gold"
    annotated_video: str = ""
    frame_counts_csv: str = ""
    summary_json: str = ""
    snapshots_dir: str = ""


@dataclass
class RuntimeConfig:
    """PyTorch device routing, multi-threading, and VRAM options.

    Attributes:
        require_cuda: If True, halts execution if CUDA is unavailable.
        device: CUDA device ID (e.g. 0 for GPU 0).
        batch_size: Number of frames processed in a single batch.
        auto_reduce_batch_on_oom: Halves batch size on CUDA Out-of-Memory.
        empty_cuda_cache_every_batches: Cache cleanup interval (0 to disable).
        torch_float32_matmul_precision: Precision level for PyTorch matrix operations.
        allow_tf32: Enables TensorFloat32 math kernels on Ampere+ GPUs (RTX 4090).
    """
    require_cuda: bool = True
    device: int = 0
    batch_size: int = 16
    auto_reduce_batch_on_oom: bool = True
    empty_cuda_cache_every_batches: int = 0
    torch_float32_matmul_precision: str = "highest"
    allow_tf32: bool = True


@dataclass
class InferenceConfig:
    """Model task, weights format and normalization thresholds for density estimation.

    Attributes:
        task: Pipeline task ('density_count' or 'detect').
        weight_format: Preferred weights engine format ('auto', 'pth', 'safetensors', 'onnx', 'trt').
        imgsz: Resolution of frames fed into the network (e.g. 640 or 1920).
        conf: Confidence threshold limit.
        iou: Intersection-Over-Union threshold.
        max_det: Maximum detections limit per frame.
        classes: List of integer class IDs.
        augment: Enables Test-Time Augmentation (TTA).
        half: Runs network inference in FP16 precision.
        verbose: Prints batch prediction details.
        vid_stride: Evaluates every N-th frame of the video.
        rgb_mean: Channel-wise mean for RGB image normalization.
        rgb_std: Channel-wise std for RGB image normalization.
        thermal_mean: Channel-wise mean for Thermal image normalization.
        thermal_std: Channel-wise std for Thermal image normalization.
    """
    task: str = "density_count"
    weight_format: str = "auto"
    imgsz: int = 640
    conf: float = 0.18
    iou: float = 0.65
    max_det: int = 2000
    classes: list[int] = field(default_factory=lambda: [0])
    augment: bool = False
    half: bool = False
    verbose: bool = False
    vid_stride: int = 1
    rgb_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    rgb_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    thermal_mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    thermal_std: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclass
class PreprocessingConfig:
    """Settings for RGBT spatial alignment, homography, letterboxing, and thermal CLAHE.

    Attributes:
        enabled: Controls whether pre-processing alignment is applied to frames.
        mode: Alignment algorithm ('homography', 'crop', 'none').
        target_resolution: Target [width, height] tuple for equalized output matrices.
        thermal_clahe: Enables CLAHE thermal contrast enhancement.
        keep_aspect_ratio: Enables letterboxing to preserve aspect ratio.
        clahe_clip_limit: Threshold limit for CLAHE.
        ground_pitch_compensation: Enables ADR 007 ground-plane pitch-aware affine warp.
        pitch_gradient_x: Horizontal disparity gradient per vertical unit.
        pitch_gradient_y: Vertical disparity gradient per vertical unit.
    """
    enabled: bool = False
    mode: str = "homography"
    target_resolution: list[int] = field(default_factory=lambda: [1280, 1024])
    thermal_clahe: bool = True
    keep_aspect_ratio: bool = True
    clahe_clip_limit: float = 2.5
    ground_pitch_compensation: bool = False
    pitch_gradient_x: float = 0.0
    pitch_gradient_y: float = 0.0


@dataclass
class CountingConfig:
    """Target output detection features for people counts.

    Attributes:
        count_source: Source for counts ('density_map', 'boxes' or 'keypoints').
        require_keypoints: Requires pose checkpoints for counts validation.
    """
    count_source: str = "density_map"
    require_keypoints: bool = False


@dataclass
class OverlayConfig:
    """Visual style settings for annotations drawn on the output video.

    Attributes:
        enabled: If True, draws annotations on frames.
        font_scale: Font scale multiplier for rendered text.
        thickness: Line thickness and text weight.
    """
    enabled: bool = True
    font_scale: float = 0.6
    thickness: int = 1


@dataclass
class OutputConfig:
    """Report and export specifications.

    Attributes:
        output_resolution: Resizes video frames to [width, height] before compression.
        save_annotated_video: Enables saving of the output MP4 video.
        save_frame_counts: Enables saving of the CSV frame records.
        save_summary: Enables saving of the JSON report summary.
        save_snapshot_every_n_frames: Interval to save JPEG audit snapshots (0 to disable).
        save_density_heatmap: Enables rendering of colored density heatmaps.
        heatmap_colormap: OpenCV colormap for density maps ('JET', 'VIRIDIS', 'INFERNO').
        heatmap_alpha: Blend opacity ratio for heatmap overlay (0.0 to 1.0).
        video_codec: OpenCV fourcc codec string (e.g. 'mp4v').
        progress_every_n_frames: Console logging interval.
        overlay: Aesthetic overlay properties.
    """
    output_resolution: list[int] | None = None
    save_annotated_video: bool = True
    save_frame_counts: bool = True
    save_summary: bool = True
    save_snapshot_every_n_frames: int = 30
    save_density_heatmap: bool = True
    heatmap_colormap: str = "JET"
    heatmap_alpha: float = 0.5
    video_codec: str = "mp4v"
    progress_every_n_frames: int = 300
    overlay: OverlayConfig = field(default_factory=OverlayConfig)


@dataclass
class MLflowConfig:
    """MLflow tracking and experiment parameterization settings.

    Attributes:
        enabled: Controls whether MLflow tracking is active.
        experiment_name: MLflow experiment grouping name.
        tracking_uri: MLflow tracking server URI or local directory.
        run_name: Optional explicit name for the run.
        tags: Key-value dictionary of metadata tags.
        log_params: Logs pipeline parameters to MLflow.
        log_step_metrics: Logs frame-level time series metrics.
        log_artifacts: Uploads output summary and report files as artifacts.
        log_annotated_video: Uploads annotated output video as artifact.
        log_model: Uploads model weights as artifact.
        autolog_ultralytics: Enables automatic Ultralytics callback tracking.
    """
    enabled: bool = True
    experiment_name: str = "head-counting-pipeline"
    tracking_uri: str = "http://127.0.0.1:5000"
    run_name: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    log_params: bool = True
    log_step_metrics: bool = True
    log_artifacts: bool = True
    log_annotated_video: bool = False
    log_model: bool = False
    autolog_ultralytics: bool = False


@dataclass
class PipelineConfig:
    """Aggregates all system configurations into a single type-safe interface."""
    app: AppConfig = field(default_factory=AppConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # Config folder directory (used to resolve relative paths)
    config_dir: Path = field(default_factory=lambda: Path.cwd())

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> PipelineConfig:
        """Loads configuration from a YAML file, parses and checks fields, and resolves paths.

        Args:
            yaml_path: Absolute or relative path to the target YAML file.

        Returns:
            A validated PipelineConfig instance with absolute path settings.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        config_dir = yaml_path.parent

        # App Configuration
        raw_app = raw.get("app", {})
        app = AppConfig(
            name=raw_app.get("name", "head-counting-pipeline"),
            seed=int(raw_app.get("seed", 42)),
        )

        # Environment Configuration
        raw_env = raw.get("environment", {})
        environment = EnvConfig(
            yolo_config_dir=raw_env.get("YOLO_CONFIG_DIR", ".yolo"),
            mpl_config_dir=raw_env.get("MPLCONFIGDIR", ".cache/matplotlib"),
            torch_home=raw_env.get("TORCH_HOME", ".cache/torch"),
        )

        # Paths Configuration
        raw_paths = raw.get("paths", {})
        paths = PathsConfig(
            video=raw_paths.get("video", ""),
            video_rgb=raw_paths.get("video_rgb", ""),
            video_thermal=raw_paths.get("video_thermal", ""),
            images=raw_paths.get("images", ""),
            images_rgb=raw_paths.get("images_rgb", ""),
            images_thermal=raw_paths.get("images_thermal", ""),
            weights=raw_paths.get("weights", "model.pth"),
            weights_search_dirs=list(raw_paths.get("weights_search_dirs", [])),
            output_dir=raw_paths.get("output_dir", "data/gold"),
            bronze_dir=raw_paths.get("bronze_dir", "data/bronze"),
            silver_dir=raw_paths.get("silver_dir", "data/silver"),
            gold_dir=raw_paths.get("gold_dir", "data/gold"),
            annotated_video=raw_paths.get("annotated_video", ""),
            frame_counts_csv=raw_paths.get("frame_counts_csv", ""),
            summary_json=raw_paths.get("summary_json", ""),
            snapshots_dir=raw_paths.get("snapshots_dir", ""),
        )

        # Runtime Configuration
        raw_runtime = raw.get("runtime", {})
        runtime = RuntimeConfig(
            require_cuda=bool(raw_runtime.get("require_cuda", True)),
            device=int(raw_runtime.get("device", 0)),
            batch_size=int(raw_runtime.get("batch_size", 16)),
            auto_reduce_batch_on_oom=bool(raw_runtime.get("auto_reduce_batch_on_oom", True)),
            empty_cuda_cache_every_batches=int(raw_runtime.get("empty_cuda_cache_every_batches", 0)),
            torch_float32_matmul_precision=raw_runtime.get("torch_float32_matmul_precision", "highest"),
            allow_tf32=bool(raw_runtime.get("allow_tf32", True)),
        )

        # Inference Configuration
        raw_inference = raw.get("inference", {})
        inference = InferenceConfig(
            task=raw_inference.get("task", "density_count"),
            weight_format=raw_inference.get("weight_format", "auto"),
            imgsz=int(raw_inference.get("imgsz", 640)),
            conf=float(raw_inference.get("conf", 0.18)),
            iou=float(raw_inference.get("iou", 0.65)),
            max_det=int(raw_inference.get("max_det", 2000)),
            classes=list(raw_inference.get("classes", [0])),
            augment=bool(raw_inference.get("augment", False)),
            half=bool(raw_inference.get("half", False)),
            verbose=bool(raw_inference.get("verbose", False)),
            vid_stride=int(raw_inference.get("vid_stride", 1)),
            rgb_mean=list(raw_inference.get("rgb_mean", [0.485, 0.456, 0.406])),
            rgb_std=list(raw_inference.get("rgb_std", [0.229, 0.224, 0.225])),
            thermal_mean=list(raw_inference.get("thermal_mean", [0.5, 0.5, 0.5])),
            thermal_std=list(raw_inference.get("thermal_std", [0.5, 0.5, 0.5])),
        )

        # Preprocessing Configuration
        raw_prep = raw.get("preprocessing", {})
        preprocessing = PreprocessingConfig(
            enabled=bool(raw_prep.get("enabled", False)),
            mode=str(raw_prep.get("mode", "homography")),
            target_resolution=list(raw_prep.get("target_resolution", [1280, 1024])),
            thermal_clahe=bool(raw_prep.get("thermal_clahe", True)),
            keep_aspect_ratio=bool(raw_prep.get("keep_aspect_ratio", True)),
            clahe_clip_limit=float(raw_prep.get("clahe_clip_limit", 2.5)),
        )

        # Counting Configuration
        raw_counting = raw.get("counting", {})
        counting = CountingConfig(
            count_source=raw_counting.get("count_source", "density_map"),
            require_keypoints=bool(raw_counting.get("require_keypoints", False)),
        )

        # Output & Overlay Configuration
        raw_output = raw.get("output", {})
        raw_overlay = raw_output.get("overlay", {})
        overlay = OverlayConfig(
            enabled=bool(raw_overlay.get("enabled", True)),
            font_scale=float(raw_overlay.get("font_scale", 0.6)),
            thickness=int(raw_overlay.get("thickness", 1)),
        )
        
        output = OutputConfig(
            output_resolution=raw_output.get("output_resolution"),
            save_annotated_video=bool(raw_output.get("save_annotated_video", True)),
            save_frame_counts=bool(raw_output.get("save_frame_counts", True)),
            save_summary=bool(raw_output.get("save_summary", True)),
            save_snapshot_every_n_frames=int(raw_output.get("save_snapshot_every_n_frames", 30)),
            save_density_heatmap=bool(raw_output.get("save_density_heatmap", True)),
            heatmap_colormap=str(raw_output.get("heatmap_colormap", "JET")),
            heatmap_alpha=float(raw_output.get("heatmap_alpha", 0.5)),
            video_codec=raw_output.get("video_codec", "mp4v"),
            progress_every_n_frames=int(raw_output.get("progress_every_n_frames", 300)),
            overlay=overlay,
        )

        # MLflow Configuration
        raw_mlflow = raw.get("mlflow", {})
        mlflow_cfg = MLflowConfig(
            enabled=bool(raw_mlflow.get("enabled", True)),
            experiment_name=str(raw_mlflow.get("experiment_name", "head-counting-pipeline")),
            tracking_uri=str(raw_mlflow.get("tracking_uri", "http://127.0.0.1:5000")),
            run_name=str(raw_mlflow.get("run_name", "")),
            tags=dict(raw_mlflow.get("tags", {})),
            log_params=bool(raw_mlflow.get("log_params", True)),
            log_step_metrics=bool(raw_mlflow.get("log_step_metrics", True)),
            log_artifacts=bool(raw_mlflow.get("log_artifacts", True)),
            log_annotated_video=bool(raw_mlflow.get("log_annotated_video", False)),
            log_model=bool(raw_mlflow.get("log_model", False)),
            autolog_ultralytics=bool(raw_mlflow.get("autolog_ultralytics", False)),
        )

        config = cls(
            app=app,
            environment=environment,
            paths=paths,
            runtime=runtime,
            inference=inference,
            preprocessing=preprocessing,
            counting=counting,
            output=output,
            mlflow=mlflow_cfg,
            config_dir=config_dir,
        )
        config._resolve_paths()
        return config

    def _resolve_paths(self) -> None:
        """Resolves raw relative paths in settings into absolute paths relative to the config file's parent folder."""
        
        def to_absolute(val: str) -> str:
            if not val:
                return ""
            p = Path(val)
            if p.is_absolute():
                return str(p.resolve())
            return str((self.config_dir / p).resolve())

        # Resolve paths in PathConfig
        self.paths.video = to_absolute(self.paths.video)
        self.paths.video_rgb = to_absolute(self.paths.video_rgb)
        self.paths.video_thermal = to_absolute(self.paths.video_thermal)
        self.paths.images = to_absolute(self.paths.images)
        self.paths.images_rgb = to_absolute(self.paths.images_rgb)
        self.paths.images_thermal = to_absolute(self.paths.images_thermal)
        self.paths.weights = to_absolute(self.paths.weights)
        self.paths.weights_search_dirs = [to_absolute(d) for d in self.paths.weights_search_dirs]
        self.paths.output_dir = to_absolute(self.paths.output_dir)
        self.paths.bronze_dir = to_absolute(self.paths.bronze_dir)
        self.paths.silver_dir = to_absolute(self.paths.silver_dir)
        self.paths.gold_dir = to_absolute(self.paths.gold_dir)

        # Fallback single video to video_rgb if video_rgb not specified
        if not self.paths.video_rgb and self.paths.video:
            self.paths.video_rgb = self.paths.video
        elif not self.paths.video and self.paths.video_rgb:
            self.paths.video = self.paths.video_rgb

        # Setup defaults for derived paths if not custom-specified
        out_dir = Path(self.paths.output_dir)
        
        if not self.paths.annotated_video:
            target_video = self.paths.video_rgb or self.paths.video
            video_name = Path(target_video).stem if target_video else "video"
            self.paths.annotated_video = str(out_dir / f"{video_name}_annotated.mp4")
        else:
            self.paths.annotated_video = to_absolute(self.paths.annotated_video)
            
        if not self.paths.frame_counts_csv:
            self.paths.frame_counts_csv = str(out_dir / "frame_counts.csv")
        else:
            self.paths.frame_counts_csv = to_absolute(self.paths.frame_counts_csv)
            
        if not self.paths.summary_json:
            self.paths.summary_json = str(out_dir / "summary.json")
        else:
            self.paths.summary_json = to_absolute(self.paths.summary_json)
            
        if not self.paths.snapshots_dir:
            self.paths.snapshots_dir = str(out_dir / "snapshots")
        else:
            self.paths.snapshots_dir = to_absolute(self.paths.snapshots_dir)
