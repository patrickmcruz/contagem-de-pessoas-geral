"""
DEF-rgbtcc Model Management Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module manages instantiation, hardware allocation, precision settings, and inference
operations for the DEF-rgbtcc dual-modality (RGB-T) crowd counting model (ArXiv 2509.17079).
It includes adaptive logic to handle CUDA Out-of-Memory (OOM) errors by recursively halving
batch sizes during runtime and supports TensorRT FP16/FP32, SafeTensors, PyTorch (.pth) and ONNX formats.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# Attempt importing RGBTCCInference from def_rgbtcc.serve
try:
    from def_rgbtcc.serve import RGBTCCInference  # type: ignore[import-untyped]
except ImportError:
    RGBTCCInference = None
    logger.warning("def_rgbtcc module not installed. Running with fallback model handler.")

# Placeholders for legacy Ultralytics YOLO compatibility
YOLO: Any = None


class DEFModelHandler:
    """Manages the DEF-rgbtcc model instance, configures hardware settings, and executes batch predictions.

    Attributes:
        config: The parsed PipelineConfig instance.
        device: CUDA device index or CPU identifier.
        require_cuda: If True, blocks CPU fallback if GPU fails.
        weights_ref: Resolved path to local model weights file or HuggingFace repo.
        model: Loaded RGBTCCInference or PyTorch model instance.
        predict_args: Inference configuration parameters.
    """

    def __init__(self, config: PipelineConfig):
        """Initializes the model handler using system config.

        Args:
            config: A PipelineConfig instance.
        """
        self.config = config
        self.device = config.runtime.device
        self.require_cuda = config.runtime.require_cuda
        self.weights_ref = self._resolve_weight_reference()
        self.model: Any | None = None
        self.predict_args: dict[str, Any] = {}

    def _resolve_weight_reference(self) -> str:
        """Resolves the location of model weights, prioritizing local TensorRT (.trt/.engine) formats.

        Search Priority:
        1. Local TensorRT engines (`model_fp16.trt`, `model_fp32.trt`, `best.engine`).
        2. Local SafeTensors weights (`model.safetensors`).
        3. Local PyTorch state dicts (`model.pth`, `best.pt`).
        4. Local ONNX models (`model.onnx`).
        5. Remote HuggingFace repository ID (`ilessio-aiflowlab/DEF-rgbtcc`).

        Returns:
            An absolute path string of resolved weights, or raw model reference.
        """
        weight_ref = self.config.paths.weights
        weight_path = Path(weight_ref)

        search_dirs = self.config.paths.weights_search_dirs
        if not search_dirs:
            search_dirs = ["weights", "."]

        # Always check for TensorRT engines first for RTX acceleration
        engine_candidates = ["model_fp16.trt", "model_fp32.trt", "best.engine"]
        for cand in engine_candidates:
            for sdir in search_dirs:
                p = Path(sdir) / cand
                if p.exists():
                    logger.info(f"[RTX BOOST] TensorRT engine weights found: {p.resolve()}")
                    return str(p.resolve())

        # If weight_path is absolute and exists, use it
        if weight_path.is_absolute() and weight_path.exists():
            return str(weight_path.resolve())

        # Check remaining weight format candidates in search directories
        candidate_names = [
            "model.safetensors",
            weight_path.name,
            "model.pth",
            "model.onnx",
        ]

        for cand in candidate_names:
            for sdir in search_dirs:
                p = Path(sdir) / cand
                if p.exists():
                    return str(p.resolve())

        for cand in candidate_names:
            p = Path(cand)
            if p.exists():
                return str(p.resolve())

        fallback_repo = "ilessio-aiflowlab/DEF-rgbtcc"
        logger.warning(f"Weights file not found locally. Fallback to model reference: {weight_ref} / {fallback_repo}")
        return weight_ref

    def setup_runtime(self) -> None:
        """Configures OpenCV settings, PyTorch backends, and CUDA optimization flags.

        Raises:
            RuntimeError: If require_cuda is True but CUDA is unavailable.
        """
        if self.require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required by config, but PyTorch cannot identify any active GPU."
            )

        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

            allow_tf32 = self.config.runtime.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = True

            precision = self.config.runtime.torch_float32_matmul_precision
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(precision)

            logger.info(f"CUDA initialized on device {self.device} successfully.")

    def load_model(self) -> None:
        """Instantiates the DEF-rgbtcc model and transfers weights to target hardware.

        Uses RGBTCCInference from def_rgbtcc.serve if available, or PyTorch fallback.
        """
        inference_cls = RGBTCCInference
        if inference_cls is not None:
            try:
                self.model = inference_cls(self.weights_ref)
                logger.info(f"DEF-rgbtcc model loaded successfully via RGBTCCInference from: {self.weights_ref}")
            except Exception as e:
                logger.warning(f"RGBTCCInference load failed: {e}. Switching to internal PyTorch handler.")
                self.model = self._create_fallback_model()
        else:
            self.model = self._create_fallback_model()

        if torch.cuda.is_available() and hasattr(self.model, "to"):
            try:
                self.model.to(f"cuda:{self.device}")
            except Exception:
                pass

        self.predict_args = {
            "device": self.device if torch.cuda.is_available() else "cpu",
            "imgsz": self.config.inference.imgsz,
            "half": self.config.inference.half,
            "weight_format": self.config.inference.weight_format,
        }
        logger.info(f"Model predict arguments: {json.dumps(self.predict_args, default=str)}")

    def _create_fallback_model(self) -> Any:
        """Creates a mock / fallback inference object for testing and clean standalone runs."""
        class FallbackRGBTModel:
            def predict(self, rgb_img: np.ndarray, thermal_img: np.ndarray) -> dict[str, Any]:
                h, w = rgb_img.shape[:2]
                density_map = np.full((h, w), 0.01, dtype=np.float32)
                estimated_count = float(np.sum(density_map))
                return {
                    "density_map": density_map,
                    "count": estimated_count,
                    "orig_img": rgb_img,
                }
        return FallbackRGBTModel()

    def predict_batch(
        self,
        rgb_frames: list[np.ndarray],
        thermal_frames: list[np.ndarray] | None = None,
    ) -> list[dict[str, Any]]:
        """Runs batch inference on pairs of RGB and Thermal video frames. Handles CUDA OOM gracefully.

        Args:
            rgb_frames: List of NumPy BGR arrays representing RGB frames.
            thermal_frames: List of NumPy BGR arrays representing Thermal frames (optional).

        Returns:
            A list of dicts containing orig_img, density_map, and count.

        Raises:
            RuntimeError: If OOM is triggered and cannot be resolved by halving batch size.
        """
        if not rgb_frames:
            return []

        if self.model is None:
            raise RuntimeError("Model has not been loaded yet! Call load_model() first.")

        if thermal_frames is None or len(thermal_frames) != len(rgb_frames):
            thermal_frames = [f.copy() for f in rgb_frames]

        batch_size = len(rgb_frames)

        try:
            results = []
            with torch.inference_mode():
                for rgb_img, thermal_img in zip(rgb_frames, thermal_frames):
                    res = self.model.predict(rgb_img, thermal_img)
                    
                    if isinstance(res, dict):
                        density_map = res.get("density_map")
                        count = res.get("count", float(np.sum(density_map)) if density_map is not None else 0.0)
                    else:
                        density_map = getattr(res, "density_map", np.zeros(rgb_img.shape[:2], dtype=np.float32))
                        count = getattr(res, "count", float(np.sum(density_map)))

                    results.append({
                        "orig_img": rgb_img,
                        "density_map": density_map,
                        "count": count,
                    })

            empty_interval = self.config.runtime.empty_cuda_cache_every_batches
            if empty_interval > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return results

        except torch.cuda.OutOfMemoryError:
            is_oom = True
        except RuntimeError as error:
            is_oom = "out of memory" in str(error).lower()
            if not is_oom:
                raise

        if is_oom:
            if not self.config.runtime.auto_reduce_batch_on_oom or batch_size == 1:
                raise RuntimeError("CUDA Out-Of-Memory encountered even at batch size 1.")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            midpoint = max(1, batch_size // 2)
            logger.warning(
                f"[CUDA OOM] Reducing batch size from {batch_size} to {midpoint} and retrying."
            )
            return (
                self.predict_batch(rgb_frames[:midpoint], thermal_frames[:midpoint])
                + self.predict_batch(rgb_frames[midpoint:], thermal_frames[midpoint:])
            )


# ============================================================================
# LEGACY SINGLE-STREAM ADAPTER FOR BACKWARD COMPATIBILITY
# ============================================================================

class YOLOModelHandler(DEFModelHandler):
    """Backward-compatible YOLOModelHandler adapter."""

    def load_model(self) -> None:
        expected_task = self.config.inference.task
        yolo_cls = globals().get("YOLO")
        if expected_task == "detect" and yolo_cls is not None and callable(yolo_cls):
            try:
                self.model = yolo_cls(self.weights_ref, task=expected_task)
                if torch.cuda.is_available() and hasattr(self.model, "to"):
                    self.model.to(f"cuda:{self.device}")
                self.predict_args = {
                    "device": self.device if torch.cuda.is_available() else "cpu",
                    "imgsz": self.config.inference.imgsz,
                    "conf": self.config.inference.conf,
                }
                return
            except Exception:
                pass
        super().load_model()

    def predict_batch(self, frames: list[np.ndarray], **kwargs: Any) -> list[Any]:  # type: ignore[override]
        if (
            self.model is not None
            and hasattr(self.model, "predict")
            and getattr(self.config.inference, "task", None) == "detect"
        ):
            try:
                with torch.inference_mode():
                    return list(self.model.predict(source=frames, **self.predict_args))
            except Exception:
                pass
        return super().predict_batch(frames, thermal_frames=None)
