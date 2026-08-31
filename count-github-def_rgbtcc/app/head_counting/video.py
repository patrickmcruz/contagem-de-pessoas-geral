"""
Video I/O Threading Module for Dual-Stream RGB-T
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides multi-threaded I/O classes for dual-modality (RGB + Thermal) video processing:
- `DualStreamVideoReader`: Decodes RGB and Thermal video feeds in background threads.
- `DualStreamVideoWriterWrapper`: Renders colored density heatmaps, overlays banners,
  resizes, writes to disk, and saves audit snapshots in background worker threads.
- `VideoReader` / `VideoWriterWrapper`: Legacy single-stream compatibility adapters.
"""

from __future__ import annotations
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Generator, Tuple
import cv2
import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class DualStreamVideoReader:
    """Multi-threaded video reader that decodes RGB and Thermal frames in background threads.

    Attributes:
        video_rgb_path: Path to the target RGB video file.
        video_thermal_path: Path to the target Thermal/Infrared video file (optional).
        stride: Stride factor; process every N-th frame.
        queue: Thread-safe queue containing decompressed (frame_rgb, frame_thermal, frame_idx).
        stop_event: Thread shutdown signal.
        thread: Background thread instance.
        metadata: Dict containing resolution, fps, frame count and stream details.
    """

    def __init__(
        self,
        video_rgb_path: str | Path,
        video_thermal_path: str | Path | None = None,
        stride: int = 1,
        queue_size: int = 128,
    ):
        """Initializes DualStreamVideoReader and extracts stream metadata.

        Args:
            video_rgb_path: Path to the primary RGB video file.
            video_thermal_path: Path to the secondary Thermal video file.
            stride: Stride factor for skipping frames.
            queue_size: Maximum capacity of the frame buffer queue.
        """
        self.video_rgb_path = Path(video_rgb_path)
        self.video_thermal_path = Path(video_thermal_path) if video_thermal_path else None
        self.stride = max(1, stride)
        self.queue: queue.Queue[
            Tuple[np.ndarray, np.ndarray, int] | Tuple[None, None, None]
        ] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.metadata: dict[str, Any] = {}

        self._validate_and_extract_metadata()

    def _validate_and_extract_metadata(self) -> None:
        """Validates video file availability and retrieves dimensions and fps settings.

        Raises:
            FileNotFoundError: If the RGB video file does not exist.
            RuntimeError: If OpenCV fails to open the video stream.
        """
        if not self.video_rgb_path.exists():
            raise FileNotFoundError(f"RGB video file not found at: {self.video_rgb_path}")

        cap_rgb = cv2.VideoCapture(str(self.video_rgb_path))
        if not cap_rgb.isOpened():
            raise RuntimeError(f"OpenCV failed to open RGB video file: {self.video_rgb_path}")

        width_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_rgb = float(cap_rgb.get(cv2.CAP_PROP_FPS))
        count_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_rgb.release()

        metadata_thermal = {"width": width_rgb, "height": height_rgb, "fps": fps_rgb, "frame_count": count_rgb}

        if self.video_thermal_path and self.video_thermal_path.exists():
            cap_th = cv2.VideoCapture(str(self.video_thermal_path))
            if cap_th.isOpened():
                metadata_thermal = {
                    "width": int(cap_th.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap_th.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": float(cap_th.get(cv2.CAP_PROP_FPS)),
                    "frame_count": int(cap_th.get(cv2.CAP_PROP_FRAME_COUNT)),
                }
                cap_th.release()

        fps = fps_rgb if fps_rgb > 0 else 24.0
        duration_sec = count_rgb / fps if fps > 0 else 0.0

        self.metadata = {
            "width": width_rgb,
            "height": height_rgb,
            "fps": fps,
            "frame_count": count_rgb,
            "duration_sec": duration_sec,
            "rgb": {"width": width_rgb, "height": height_rgb, "fps": fps_rgb, "frame_count": count_rgb},
            "thermal": metadata_thermal,
        }

    def __enter__(self) -> DualStreamVideoReader:
        """Context manager entry; automatically starts background reader thread."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; stops reader thread and cleans up resources."""
        self.stop()

    def start(self) -> None:
        """Starts background video decoding worker thread."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.thread.start()
        logger.info(f"DualStreamVideoReader thread started for: {self.video_rgb_path.name}")

    def stop(self) -> None:
        """Signals background reader thread to stop and blocks until joined."""
        self.stop_event.set()

        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            logger.info("DualStreamVideoReader thread joined successfully.")

    def _reader_worker(self) -> None:
        """Worker loop reading synchronized frames from RGB and Thermal feeds."""
        cap_rgb = cv2.VideoCapture(str(self.video_rgb_path))
        cap_thermal = (
            cv2.VideoCapture(str(self.video_thermal_path))
            if self.video_thermal_path and self.video_thermal_path.exists()
            else None
        )

        frame_idx = 0

        try:
            while cap_rgb.isOpened() and not self.stop_event.is_set():
                ok_rgb, frame_rgb = cap_rgb.read()
                if not ok_rgb:
                    break

                if cap_thermal and cap_thermal.isOpened():
                    ok_th, frame_thermal = cap_thermal.read()
                    if not ok_th or frame_thermal is None:
                        frame_thermal = frame_rgb.copy()
                else:
                    frame_thermal = frame_rgb.copy()

                if frame_idx % self.stride == 0:
                    try:
                        self.queue.put((frame_rgb, frame_thermal, frame_idx), block=True, timeout=0.1)
                    except queue.Full:
                        continue

                frame_idx += 1
        except Exception as e:
            logger.error(f"Error in DualStreamVideoReader worker: {e}", exc_info=True)
        finally:
            cap_rgb.release()
            if cap_thermal:
                cap_thermal.release()
            try:
                self.queue.put((None, None, None), block=True, timeout=2.0)
            except queue.Full:
                pass

    def iter_frames(self) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
        """Iterates over decoded frames as they arrive in the buffer queue.

        Yields:
            A tuple of (frame_rgb, frame_thermal, frame_index).
        """
        while not self.stop_event.is_set():
            frame_rgb, frame_thermal, frame_idx = self.queue.get(block=True)
            if frame_rgb is None:
                break
            yield frame_rgb, frame_thermal, frame_idx


class DualStreamVideoWriterWrapper:
    """Multi-threaded writer rendering density heatmaps, text overlays, and writing output videos.

    Attributes:
        config: The PipelineConfig instance.
        video_meta: Source video metadata.
        queue: Thread-safe queue containing prediction frames.
        stop_event: Shutdown signal.
        thread: Worker thread instance.
    """

    def __init__(self, config: PipelineConfig, video_meta: dict[str, Any]):
        """Initializes DualStreamVideoWriterWrapper and configures OpenCV VideoWriter.

        Args:
            config: A PipelineConfig instance.
            video_meta: Dictionary containing video dimensions and FPS.
        """
        self.config = config
        self.video_meta = video_meta
        self.queue: queue.Queue[Tuple[Any, float | int, int, float] | None] = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._writer: cv2.VideoWriter | None = None

        self._init_writer()

    def _init_writer(self) -> None:
        """Configures and opens the OpenCV VideoWriter object."""
        if not self.config.output.save_annotated_video:
            return

        annotated_path = Path(self.config.paths.annotated_video)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)

        res = self.config.output.output_resolution
        if res and len(res) == 2:
            self.out_w, self.out_h = int(res[0]), int(res[1])
        else:
            self.out_w, self.out_h = self.video_meta["width"], self.video_meta["height"]

        fps = max(1, round(self.video_meta["fps"])) if self.video_meta.get("fps") else 24
        codec = self.config.output.video_codec
        fourcc = cv2.VideoWriter_fourcc(*codec)

        try:
            self._writer = cv2.VideoWriter(
                str(annotated_path),
                fourcc,
                fps,
                (self.out_w, self.out_h),
            )
            logger.info(f"Initialized VideoWriter: {annotated_path.name} at {self.out_w}x{self.out_h} @ {fps}fps")
        except Exception as e:
            logger.error(f"Failed to initialize VideoWriter: {e}")
            self._writer = None

    def __enter__(self) -> DualStreamVideoWriterWrapper:
        """Context manager entry; starts background writer thread."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; stops writer thread and releases resources."""
        self.stop()

    def start(self) -> None:
        """Spawns background writer worker thread."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.thread.start()
        logger.info("DualStreamVideoWriterWrapper worker thread started.")

    def stop(self) -> None:
        """Pushes stop sentinel, waits for writer to finish queue items, and releases outputs."""
        if self.thread and self.thread.is_alive():
            try:
                self.queue.put(None, block=True, timeout=2.0)
            except queue.Full:
                pass
            self.thread.join(timeout=10.0)
            logger.info("DualStreamVideoWriterWrapper worker thread joined.")

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Released VideoWriter resource.")

    def write_result(self, result: Any, count: float | int, frame_idx: int, timestamp_sec: float) -> None:
        """Enqueues inference results for heatmap rendering and frame writing.

        Args:
            result: Result dictionary or object containing orig_img and density_map.
            count: Number of heads/people estimated.
            frame_idx: Index of current frame.
            timestamp_sec: Frame timestamp in seconds.
        """
        try:
            self.queue.put((result, count, frame_idx, timestamp_sec), block=True, timeout=1.0)
        except queue.Full:
            logger.warning("Writer queue full. Dropping frame index: %d", frame_idx)

    def _annotate_frame(self, result: Any, count: float | int, frame_idx: int, timestamp_sec: float) -> np.ndarray:
        """Renders colored density heatmaps and text banners on top of the frame.

        Args:
            result: Result object or dictionary with keys 'orig_img' and 'density_map'.
            count: Estimated crowd count.
            frame_idx: Current frame index.
            timestamp_sec: Frame time stamp in seconds.

        Returns:
            A NumPy array of the annotated image.
        """
        overlay_cfg = self.config.output.overlay
        
        # Extract original RGB image
        if isinstance(result, dict):
            orig_img = result.get("orig_img")
            density_map = result.get("density_map")
        elif hasattr(result, "orig_img"):
            orig_img = result.orig_img
            density_map = getattr(result, "density_map", None)
        else:
            orig_img = None
            density_map = None

        if orig_img is None:
            return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)

        annotated = orig_img.copy()

        # Render density map heatmap overlay
        if (
            self.config.output.save_density_heatmap
            and density_map is not None
            and isinstance(density_map, np.ndarray)
        ):
            # Normalize density map to 0..255 range
            d_min, d_max = density_map.min(), density_map.max()
            if d_max > d_min:
                normalized = ((density_map - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros_like(density_map, dtype=np.uint8)

            colormap_name = self.config.output.heatmap_colormap.upper()
            colormap_id = getattr(cv2, f"COLORMAP_{colormap_name}", cv2.COLORMAP_JET)
            heatmap = cv2.applyColorMap(normalized, colormap_id)

            # Resize heatmap to match image dimensions if needed
            if heatmap.shape[:2] != annotated.shape[:2]:
                heatmap = cv2.resize(heatmap, (annotated.shape[1], annotated.shape[0]))

            alpha = self.config.output.heatmap_alpha
            annotated = cv2.addWeighted(annotated, 1.0 - alpha, heatmap, alpha, 0)

        # Legacy bounding box plot fallback if YOLO result object is provided
        elif hasattr(result, "plot") and overlay_cfg.enabled:
            annotated = result.plot(labels=False, conf=False, boxes=True, line_width=overlay_cfg.thickness)

        if not overlay_cfg.enabled:
            return annotated

        font_scale = overlay_cfg.font_scale
        thickness = overlay_cfg.thickness

        # Draw red/orange RGBT count text banner at top-left
        cv2.putText(
            annotated,
            f"Contagem RGBT: {count:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.5,
            (0, 85, 255),
            thickness + 1,
            cv2.LINE_AA,
        )

        # Draw frame index and timestamp at bottom-left
        cv2.putText(
            annotated,
            f"Frame: {frame_idx} | Tempo: {timestamp_sec:.2f}s",
            (20, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness),
            cv2.LINE_AA,
        )
        return annotated

    def _writer_worker(self) -> None:
        """Background loop pulling items, rendering heatmaps, writing video frames, and saving snapshots."""
        snapshot_every = self.config.output.save_snapshot_every_n_frames
        snapshots_dir = Path(self.config.paths.snapshots_dir)

        if snapshot_every > 0:
            snapshots_dir.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                item = self.queue.get(block=True)
                if item is None:
                    break

                result, count, frame_idx, timestamp_sec = item

                annotated = self._annotate_frame(result, count, frame_idx, timestamp_sec)

                if self._writer is not None:
                    if annotated.shape[1] != self.out_w or annotated.shape[0] != self.out_h:
                        annotated = cv2.resize(annotated, (self.out_w, self.out_h))
                    self._writer.write(annotated)

                if snapshot_every > 0 and frame_idx % snapshot_every == 0:
                    filename = snapshots_dir / f"snapshot_frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(filename), annotated)

        except Exception as e:
            logger.error(f"Error in DualStreamVideoWriterWrapper worker: {e}", exc_info=True)


# ============================================================================
# LEGACY SINGLE-STREAM ADAPTERS FOR BACKWARD COMPATIBILITY
# ============================================================================

class VideoReader(DualStreamVideoReader):
    """Backward-compatible single stream VideoReader adapter."""

    def __init__(self, video_path: str | Path, stride: int = 1, queue_size: int = 128):
        super().__init__(
            video_rgb_path=video_path,
            video_thermal_path=None,
            stride=stride,
            queue_size=queue_size,
        )

    def iter_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:  # type: ignore[override]
        for frame_rgb, _frame_thermal, idx in super().iter_frames():
            yield frame_rgb, idx


class VideoWriterWrapper(DualStreamVideoWriterWrapper):
    """Backward-compatible VideoWriterWrapper adapter."""
    pass
