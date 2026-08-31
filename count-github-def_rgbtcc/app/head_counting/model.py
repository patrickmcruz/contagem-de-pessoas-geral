"""
YOLO Model Management Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module manages the instantiation, hardware allocation, precision modes, and
inference operations for the YOLO model. It includes adaptive logic to handle CUDA
Out-of-Memory (OOM) errors by recursively reducing the batch size during runtime.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import torch
from ultralytics import YOLO

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class YOLOModelHandler:
    """Manages the YOLO model instance, configures hardware settings, and executes batch predictions.

    Attributes:
        config: The parsed system PipelineConfig instance.
        device: CUDA device index or identifier.
        require_cuda: If True, blocks CPU fallback if GPU fails.
        weights_ref: Resolved path to the local model weights file.
        model: Loaded YOLO model instance.
        predict_args: Inference parameter dictionary passed to YOLO's predict method.
    """

    def __init__(self, config: PipelineConfig):
        """Initializes the model handler using a system config.

        Args:
            config: A PipelineConfig instance.
        """
        self.config = config
        self.device = config.runtime.device
        self.require_cuda = config.runtime.require_cuda
        self.weights_ref = self._resolve_weight_reference()
        self.model: YOLO | None = None
        self.predict_args: dict[str, Any] = {}

    def _resolve_weight_reference(self) -> str:
        """Resolves the location of model weights, prioritizing local TensorRT (.engine) formats.

        Iterates through the local weight paths and directories specified in config
        to locate precompiled engines. If none are found, falls back to .pt files or
        notifies YOLO to download.

        Returns:
            An absolute path string of the resolved weights file, or raw weight reference.
        """
        weight_ref = self.config.paths.weights
        weight_path = Path(weight_ref)

        engine_ref = weight_ref.replace(".pt", ".engine") if weight_ref.endswith(".pt") else None

        # 1. If absolute path exists
        if weight_path.is_absolute() and weight_path.exists():
            return str(weight_path.resolve())

        # 2. Check in weights search directories
        search_dirs = self.config.paths.weights_search_dirs
        
        # Priority 1: look for Engine version (.engine)
        if engine_ref:
            for search_dir in search_dirs:
                candidate = Path(search_dir) / Path(engine_ref).name
                if candidate.exists():
                    logger.info(f"[RTX BOOST] TensorRT engine weights found: {candidate.resolve()}")
                    return str(candidate.resolve())

        # Priority 2: look for PyTorch weights (.pt)
        for search_dir in search_dirs:
            candidate = Path(search_dir) / weight_path.name
            if candidate.exists():
                return str(candidate.resolve())

        # 3. Check locally in current directory
        if engine_ref:
            local_engine = Path(engine_ref).name
            if Path(local_engine).exists():
                logger.info(f"[RTX BOOST] TensorRT engine weights found locally: {local_engine}")
                return str(Path(local_engine).resolve())

        if Path(weight_path.name).exists():
            return str(Path(weight_path.name).resolve())

        # 4. Fallback to raw reference (e.g. HuggingFace ID or download)
        logger.warning(f"Weights file not found locally. Fallback to Ultralytics download for: {weight_ref}")
        return weight_ref

    def setup_runtime(self) -> None:
        """Configures OpenCV settings, PyTorch backends, and CUDA optimization flags.

        Activates/deactivates TensorFloat32 (TF32) kernels, benchmarking, and matmul
        precision modes based on the loaded configuration.

        Raises:
            RuntimeError: If require_cuda is True but CUDA is unavailable.
        """
        # Disable synchronous YOLO settings to boost batch inference throughput
        try:
            from ultralytics.utils import SETTINGS
            SETTINGS.update({"sync": False})
        except Exception as e:
            logger.debug(f"Could not disable YOLO sync setting: {e}")

        # Validate CUDA availability
        if self.require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required by config, but PyTorch cannot identify any active GPU."
            )

        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            
            # TensorFloat32 settings
            allow_tf32 = self.config.runtime.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = True

            # Set float32 matmul precision
            precision = self.config.runtime.torch_float32_matmul_precision
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(precision)

            logger.info(f"CUDA initialized on device {self.device} successfully.")

    def load_model(self) -> None:
        """Instantiates the YOLO model and transfers weights to target GPU/CPU hardware.

        Sets up the prediction arguments dict used during inference.

        Raises:
            RuntimeError: If the YOLO architecture task (e.g. pose) mismatches config task.
        """
        expected_task = self.config.inference.task
        
        # Load YOLO
        self.model = YOLO(self.weights_ref, task=expected_task)

        # Validate architecture task
        loaded_task = getattr(self.model, "task", None)
        if loaded_task != expected_task:
            raise RuntimeError(
                f"Model task mismatch! Loaded model task is '{loaded_task}', "
                f"but config expected '{expected_task}'. Check weights reference."
            )

        if torch.cuda.is_available():
            self.model.to(f"cuda:{self.device}")
            logger.info(f"YOLO model loaded and allocated on CUDA device {self.device}.")
        else:
            logger.info("YOLO model loaded on CPU.")

        # Setup inference arguments
        classes = self.config.inference.classes
        classes_arg = None if not classes else classes

        self.predict_args = {
            "device": self.device if torch.cuda.is_available() else "cpu",
            "imgsz": self.config.inference.imgsz,
            "conf": self.config.inference.conf,
            "iou": self.config.inference.iou,
            "max_det": self.config.inference.max_det,
            "classes": classes_arg,
            "augment": self.config.inference.augment,
            "half": self.config.inference.half,
            "verbose": self.config.inference.verbose,
            "stream": False,
        }

        logger.info(f"Model predict arguments: {json.dumps(self.predict_args, default=str)}")

    def predict_batch(self, frames: list[np.ndarray]) -> list[Any]:
        """Runs batch inference on the list of frames. Handles CUDA Out-Of-Memory gracefully.

        If a CUDA Out-Of-Memory (OOM) error occurs and auto_reduce_batch_on_oom is True,
        recursively splits the batch into two halves and attempts to process again.

        Args:
            frames: A list of NumPy arrays representing video frames.

        Returns:
            A list of YOLO Result objects containing bounding boxes and confidences.

        Raises:
            RuntimeError: If OOM is triggered and cannot be resolved by halving batch size.
        """
        if not frames:
            return []
        
        if self.model is None:
            raise RuntimeError("Model has not been loaded yet! Call load_model() first.")

        args = dict(self.predict_args)
        args["batch"] = len(frames)

        try:
            with torch.inference_mode():
                return list(self.model.predict(source=frames, **args))
        except torch.cuda.OutOfMemoryError:
            is_oom = True
        except RuntimeError as error:
            is_oom = "out of memory" in str(error).lower()
            if not is_oom:
                raise

        if is_oom:
            if not self.config.runtime.auto_reduce_batch_on_oom or len(frames) == 1:
                raise RuntimeError("CUDA Out-Of-Memory encountered even at batch size 1.")

            torch.cuda.empty_cache()
            midpoint = max(1, len(frames) // 2)
            logger.warning(
                f"[CUDA OOM] Reducing batch size from {len(frames)} to {midpoint} and retrying."
            )
            return self.predict_batch(frames[:midpoint]) + self.predict_batch(frames[midpoint:])
