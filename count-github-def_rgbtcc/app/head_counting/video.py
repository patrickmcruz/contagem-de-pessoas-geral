"""
Video & Image I/O Threading Module for Dual-Stream RGB-T
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides multi-threaded I/O classes for dual-modality (RGB + Thermal) media processing:
- `DualStreamVideoReader`: Decodes RGB and Thermal video feeds or image files/directories in background threads,
  with optional pre-processing alignment (ADR 001 Homography).
- `DualStreamVideoWriterWrapper`: Renders colored density heatmaps, overlays banners,
  resizes, writes to disk, and saves audit snapshots/annotated images in background worker threads.
- `VideoReader` / `VideoWriterWrapper`: Legacy single-stream compatibility adapters.
"""

from __future__ import annotations
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Generator, Tuple, List
import cv2
import numpy as np

from .config import PipelineConfig, PreprocessingConfig
from .preprocessing import RGBTImageEqualizer

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    """Checks whether a given path points to a supported image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


class DualStreamVideoReader:
    """Multi-threaded reader that decodes RGB and Thermal video feeds or image pairs in background threads.

    Attributes:
        video_rgb_path: Path to the target RGB video/image file or directory.
        video_thermal_path: Path to the target Thermal video/image file or directory (optional).
        stride: Stride factor; process every N-th frame.
        preprocessing_config: Optional PreprocessingConfig for homography and CLAHE.
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
        preprocessing_config: PreprocessingConfig | None = None,
    ):
        """Initializes DualStreamVideoReader and extracts stream metadata.

        Args:
            video_rgb_path: Path to the primary RGB video or image file/dir.
            video_thermal_path: Path to the secondary Thermal video or image file/dir.
            stride: Stride factor for skipping frames.
            queue_size: Maximum capacity of the frame buffer queue.
            preprocessing_config: Optional PreprocessingConfig instance.
        """
        self.video_rgb_path = Path(video_rgb_path)
        self.video_thermal_path = Path(video_thermal_path) if video_thermal_path else None
        self.stride = max(1, stride)
        self.preprocessing_config = preprocessing_config
        self.queue: queue.Queue[
            Tuple[np.ndarray, np.ndarray, int] | Tuple[None, None, None]
        ] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.metadata: dict[str, Any] = {}

        self._is_image_mode = False
        self._is_dir_mode = False
        self._rgb_files: List[Path] = []
        self._thermal_files: List[Path] = []

        self.equalizer: RGBTImageEqualizer | None = None
        if self.preprocessing_config and self.preprocessing_config.enabled:
            target_res = tuple(self.preprocessing_config.target_resolution)
            self.equalizer = RGBTImageEqualizer(
                target_size=(target_res[0], target_res[1]),
                keep_aspect_ratio=self.preprocessing_config.keep_aspect_ratio,
                thermal_clahe=self.preprocessing_config.thermal_clahe,
                clahe_clip_limit=self.preprocessing_config.clahe_clip_limit,
                mode=self.preprocessing_config.mode,
            )
            logger.info(
                f"[PREPROCESSING] Enabled: Mode={self.preprocessing_config.mode.upper()}, "
                f"TargetRes={target_res}, CLAHE={self.preprocessing_config.thermal_clahe}"
            )

        self._validate_and_extract_metadata()

    def _validate_and_extract_metadata(self) -> None:
        """Validates file/directory availability and retrieves dimensions and fps settings."""
        if not self.video_rgb_path.exists():
            raise FileNotFoundError(f"RGB input not found at: {self.video_rgb_path}")

        target_w, target_h = None, None
        if self.equalizer:
            target_w, target_h = self.equalizer.target_w, self.equalizer.target_h

        # 1. Directory of Images Mode
        if self.video_rgb_path.is_dir():
            self._is_dir_mode = True
            all_rgb = sorted([p for p in self.video_rgb_path.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
            rgb_candidates = [p for p in all_rgb if "_T." not in p.name.upper()]
            self._rgb_files = rgb_candidates if rgb_candidates else all_rgb

            if not self._rgb_files:
                raise FileNotFoundError(f"No image files found in directory: {self.video_rgb_path}")

            if self.video_thermal_path and self.video_thermal_path.is_dir():
                all_th = sorted([p for p in self.video_thermal_path.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
                th_candidates = [p for p in all_th if "_W." not in p.name.upper()]
                self._thermal_files = th_candidates if th_candidates else all_th

            first_img = cv2.imread(str(self._rgb_files[0]))
            if first_img is None:
                raise RuntimeError(f"OpenCV failed to read image file: {self._rgb_files[0]}")

            h, w = first_img.shape[:2]
            out_w = target_w if target_w else w
            out_h = target_h if target_h else h
            count = len(self._rgb_files)

            self.metadata = {
                "width": out_w,
                "height": out_h,
                "fps": 1.0,
                "frame_count": count,
                "duration_sec": float(count),
                "rgb": {"width": out_w, "height": out_h, "fps": 1.0, "frame_count": count},
                "thermal": {"width": out_w, "height": out_h, "fps": 1.0, "frame_count": count},
            }
            return

        # 2. Single Image File Mode
        if is_image_file(self.video_rgb_path):
            self._is_image_mode = True
            img_rgb = cv2.imread(str(self.video_rgb_path))
            if img_rgb is None:
                raise RuntimeError(f"OpenCV failed to read RGB image: {self.video_rgb_path}")

            h, w = img_rgb.shape[:2]
            th_w, th_h = w, h

            if self.video_thermal_path and is_image_file(self.video_thermal_path):
                img_th = cv2.imread(str(self.video_thermal_path))
                if img_th is not None:
                    th_h, th_w = img_th.shape[:2]

            out_w = target_w if target_w else w
            out_h = target_h if target_h else h

            self.metadata = {
                "width": out_w,
                "height": out_h,
                "fps": 1.0,
                "frame_count": 1,
                "duration_sec": 1.0,
                "rgb": {"width": out_w, "height": out_h, "fps": 1.0, "frame_count": 1},
                "thermal": {"width": out_w, "height": out_h, "fps": 1.0, "frame_count": 1},
            }
            return

        # 3. Video File Mode
        cap_rgb = cv2.VideoCapture(str(self.video_rgb_path))
        if not cap_rgb.isOpened():
            raise RuntimeError(f"OpenCV failed to open RGB video file: {self.video_rgb_path}")

        width_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_rgb = float(cap_rgb.get(cv2.CAP_PROP_FPS))
        count_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_rgb.release()

        out_w = target_w if target_w else width_rgb
        out_h = target_h if target_h else height_rgb

        metadata_thermal = {"width": out_w, "height": out_h, "fps": fps_rgb, "frame_count": count_rgb}

        if self.video_thermal_path and self.video_thermal_path.exists() and not self.video_thermal_path.is_dir():
            cap_th = cv2.VideoCapture(str(self.video_thermal_path))
            if cap_th.isOpened():
                metadata_thermal = {
                    "width": out_w,
                    "height": out_h,
                    "fps": float(cap_th.get(cv2.CAP_PROP_FPS)),
                    "frame_count": int(cap_th.get(cv2.CAP_PROP_FRAME_COUNT)),
                }
                cap_th.release()

        fps = fps_rgb if fps_rgb > 0 else 24.0
        duration_sec = count_rgb / fps if fps > 0 else 0.0

        self.metadata = {
            "width": out_w,
            "height": out_h,
            "fps": fps,
            "frame_count": count_rgb,
            "duration_sec": duration_sec,
            "rgb": {"width": out_w, "height": out_h, "fps": fps_rgb, "frame_count": count_rgb},
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
        try:
            # Handle Directory of Images Mode
            if self._is_dir_mode:
                for idx, rgb_file in enumerate(self._rgb_files):
                    if self.stop_event.is_set():
                        break
                    frame_rgb = cv2.imread(str(rgb_file))
                    if frame_rgb is None:
                        continue

                    frame_thermal = None
                    if idx < len(self._thermal_files):
                        frame_thermal = cv2.imread(str(self._thermal_files[idx]))

                    if frame_thermal is None:
                        frame_thermal = frame_rgb.copy()

                    if self.equalizer:
                        frame_rgb, frame_thermal = self.equalizer.process_pair(frame_rgb, frame_thermal)

                    if idx % self.stride == 0:
                        self.queue.put((frame_rgb, frame_thermal, idx), block=True)
                return

            # Handle Single Image File Mode
            if self._is_image_mode:
                frame_rgb = cv2.imread(str(self.video_rgb_path))
                frame_thermal = None
                if self.video_thermal_path and is_image_file(self.video_thermal_path):
                    frame_thermal = cv2.imread(str(self.video_thermal_path))

                if frame_thermal is None:
                    frame_thermal = frame_rgb.copy() if frame_rgb is not None else np.zeros((100, 100, 3), np.uint8)

                if frame_rgb is not None:
                    if self.equalizer:
                        frame_rgb, frame_thermal = self.equalizer.process_pair(frame_rgb, frame_thermal)
                    self.queue.put((frame_rgb, frame_thermal, 0), block=True)
                return

            # Handle Video File Mode
            cap_rgb = cv2.VideoCapture(str(self.video_rgb_path))
            cap_thermal = (
                cv2.VideoCapture(str(self.video_thermal_path))
                if self.video_thermal_path and self.video_thermal_path.exists() and not self.video_thermal_path.is_dir()
                else None
            )

            frame_idx = 0
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
                    if self.equalizer:
                        frame_rgb, frame_thermal = self.equalizer.process_pair(frame_rgb, frame_thermal)
                    try:
                        self.queue.put((frame_rgb, frame_thermal, frame_idx), block=True, timeout=0.1)
                    except queue.Full:
                        continue

                frame_idx += 1

            cap_rgb.release()
            if cap_thermal:
                cap_thermal.release()

        except Exception as e:
            logger.error(f"Error in DualStreamVideoReader worker: {e}", exc_info=True)
        finally:
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
    """Multi-threaded writer rendering density heatmaps, text overlays, and writing output videos and images.

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
            logger.error(f"Failed to initialize VideoWriter: {e}", exc_info=True)

    def __enter__(self) -> DualStreamVideoWriterWrapper:
        """Context manager entry; automatically starts background writer thread."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; stops writer thread and releases resources."""
        self.stop()

    def start(self) -> None:
        """Starts background frame rendering worker thread."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.thread.start()
        logger.info("DualStreamVideoWriterWrapper thread started.")

    def stop(self) -> None:
        """Signals background writer thread to stop and blocks until finished."""
        self.stop_event.set()
        try:
            self.queue.put(None, block=True, timeout=1.0)
        except queue.Full:
            pass

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            logger.info("DualStreamVideoWriterWrapper thread joined successfully.")

        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def write_result(self, result: Any, count: float | int, frame_idx: int, timestamp_sec: float) -> None:
        """Enqueues prediction output for background rendering and writing.

        Args:
            result: Raw inference result dict or object containing orig_img and density_map.
            count: Estimated crowd count.
            frame_idx: Active frame index.
            timestamp_sec: Frame timestamp in seconds.
        """
        try:
            self.queue.put((result, count, frame_idx, timestamp_sec), block=True, timeout=0.5)
        except queue.Full:
            logger.warning(f"Writer queue full. Dropped output frame {frame_idx}.")

    def _render_density_overlay(self, orig_img: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """Renders 2D density map as colored heatmap and blends over original image.

        Args:
            orig_img: NumPy BGR array of original frame.
            density_map: NumPy 2D array of predicted density values.

        Returns:
            Blended BGR NumPy array frame.
        """
        h, w = orig_img.shape[:2]

        if density_map is None or density_map.size == 0:
            return orig_img.copy()

        if density_map.shape[:2] != (h, w):
            density_resized = cv2.resize(density_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            density_resized = density_map.astype(np.float32)

        d_min, d_max = density_resized.min(), density_resized.max()
        if d_max > d_min:
            norm_density = ((density_resized - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        else:
            norm_density = np.zeros((h, w), dtype=np.uint8)

        cmap_name = self.config.output.heatmap_colormap.upper()
        cmap_code = getattr(cv2, f"COLORMAP_{cmap_name}", cv2.COLORMAP_JET)
        heatmap_color = cv2.applyColorMap(norm_density, cmap_code)

        alpha = float(self.config.output.heatmap_alpha)
        blended = cv2.addWeighted(orig_img, 1.0 - alpha, heatmap_color, alpha, 0)
        return blended

    def _writer_worker(self) -> None:
        """Worker loop consuming result tasks and writing annotated frames to disk."""
        snapshot_interval = self.config.output.save_snapshot_every_n_frames
        snapshots_dir = Path(self.config.paths.snapshots_dir)
        if snapshot_interval > 0:
            snapshots_dir.mkdir(parents=True, exist_ok=True)

        output_dir = Path(self.config.paths.output_dir)

        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                task = self.queue.get(block=True, timeout=0.2)
                if task is None:
                    break

                result, count, frame_idx, timestamp_sec = task

                if isinstance(result, dict):
                    orig_img = result.get("orig_img")
                    density_map = result.get("density_map")
                else:
                    orig_img = getattr(result, "orig_img", None)
                    density_map = getattr(result, "density_map", None)

                if orig_img is None:
                    continue

                if self.config.output.save_density_heatmap and density_map is not None:
                    canvas = self._render_density_overlay(orig_img, density_map)
                else:
                    canvas = orig_img.copy()

                # Overlay banner
                h_c, w_c = canvas.shape[:2]
                banner_text = f"Pessoas: {count:.1f} | Frame: {frame_idx} | Tempo: {timestamp_sec:.2f}s"
                cv2.rectangle(canvas, (10, 10), (min(w_c - 10, 520), 55), (0, 0, 0), -1)
                cv2.putText(canvas, banner_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                # Save single image file if frame_count is 1
                if self.video_meta.get("frame_count", 0) == 1:
                    image_out_path = output_dir / "annotated_heatmap.jpg"
                    cv2.imwrite(str(image_out_path), canvas)
                    logger.info(f"Annotated heatmap image saved to: {image_out_path}")

                # Save periodic snapshot
                if snapshot_interval > 0 and frame_idx % snapshot_interval == 0:
                    snap_path = snapshots_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(snap_path), canvas)

                # Write frame to video
                if self._writer is not None:
                    if (w_c, h_c) != (self.out_w, self.out_h):
                        canvas = cv2.resize(canvas, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
                    self._writer.write(canvas)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in DualStreamVideoWriterWorker: {e}", exc_info=True)


# ============================================================================
# LEGACY ADAPTERS FOR BACKWARD COMPATIBILITY
# ============================================================================

class VideoReader(DualStreamVideoReader):
    """Backward-compatible VideoReader adapter."""
    def __init__(self, video_path: str | Path, stride: int = 1, queue_size: int = 128):
        super().__init__(video_rgb_path=video_path, video_thermal_path=None, stride=stride, queue_size=queue_size)

    def iter_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:  # type: ignore[override]
        for f_rgb, _, idx in super().iter_frames():
            yield f_rgb, idx


class VideoWriterWrapper(DualStreamVideoWriterWrapper):
    """Backward-compatible VideoWriterWrapper adapter."""
    def write_result(self, result: Any, count: float | int, frame_idx: int, timestamp_sec: float) -> None:
        super().write_result(result, count, frame_idx, timestamp_sec)
