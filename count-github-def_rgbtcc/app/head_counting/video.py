"""
Video I/O Threading Module
~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides classes for multi-threaded video operations:
- `VideoReader`: Decodes video frames in a background thread and enqueues them.
- `VideoWriterWrapper`: Pulls prediction frames, applies visual overlays/annotations,
  resizes, writes to disk, and saves periodic audit snapshots in a background thread.
"""

from __future__ import annotations
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Generator, Tuple
import cv2
import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class VideoReader:
    """Multi-threaded video reader that decodes frames in a background thread to prevent disk bottlenecks.

    Attributes:
        video_path: Path to the target video.
        stride: Stride multiplier; process every N-th frame.
        queue: Thread-safe queue containing decompressed frames and indices.
        stop_event: Event to flag thread shutdown.
        thread: Thread instance executing decoder worker.
        metadata: Dict containing dimensions, framerate, frame count, duration, and size.
    """

    def __init__(self, video_path: str | Path, stride: int = 1, queue_size: int = 128):
        """Initializes VideoReader and extracts video file metadata.

        Args:
            video_path: File system path to the target video file.
            stride: Stride factor for skipping frames.
            queue_size: Maximum capacity of the frame buffer queue.
        """
        self.video_path = Path(video_path)
        self.stride = max(1, stride)
        self.queue: queue.Queue[Tuple[np.ndarray, int] | Tuple[None, None]] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.metadata: dict[str, Any] = {}
        
        self._validate_and_extract_metadata()

    def _validate_and_extract_metadata(self) -> None:
        """Validates video file availability and retrieves dimensions and fps settings.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If OpenCV fails to open the video stream.
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file: {self.video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        decoded_size_gib = (width * height * 3 * frame_count) / (1024**3)

        self.metadata = {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "decoded_size_gib": decoded_size_gib,
        }
        cap.release()

    def __enter__(self) -> VideoReader:
        """Context manager entry; automatically starts the background reader thread."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; automatically stops the reader thread and cleans up."""
        self.stop()

    def start(self) -> None:
        """Starts the background video decoding thread."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.thread.start()
        logger.info(f"Video reader thread started for: {self.video_path.name}")

    def stop(self) -> None:
        """Signals background reader thread to stop and blocks until joined."""
        self.stop_event.set()
        
        # Drain queue if thread is blocked on a put
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            logger.info("Video reader thread joined successfully.")

    def _reader_worker(self) -> None:
        """Worker loop that runs in the background thread.

        Sequentially reads frames, skips according to stride, and puts them into the queue.
        Pushes (None, None) sentinel when complete.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        frame_idx = 0

        try:
            while cap.isOpened() and not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                
                if frame_idx % self.stride == 0:
                    try:
                        # Put with timeout to allow checking stop_event periodically
                        self.queue.put((frame, frame_idx), block=True, timeout=0.1)
                    except queue.Full:
                        continue
                
                frame_idx += 1
        except Exception as e:
            logger.error(f"Error in video reader thread: {e}", exc_info=True)
        finally:
            cap.release()
            # Push sentinel indicating end of video
            try:
                self.queue.put((None, None), block=True, timeout=2.0)
            except queue.Full:
                pass

    def iter_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Iterates over frames as they become decoded and pushed to the queue.

        Yields:
            A tuple of (frame, frame_index).
        """
        while not self.stop_event.is_set():
            frame, frame_idx = self.queue.get(block=True)
            if frame is None:
                break
            yield frame, frame_idx


class VideoWriterWrapper:
    """Multi-threaded writer that annotates, resizes, and writes video frames in a background thread.

    Attributes:
        config: The parsed PipelineConfig configuration.
        video_meta: Video metadata used to match output resolution and framerates.
        queue: Thread-safe queue containing inference results and annotations tasks.
        stop_event: Event to flag thread shutdown.
        thread: Thread instance executing writing worker.
    """

    def __init__(self, config: PipelineConfig, video_meta: dict[str, Any]):
        """Initializes VideoWriterWrapper and opens the target VideoWriter output.

        Args:
            config: A PipelineConfig instance.
            video_meta: A dictionary containing dimensions and fps of the source video.
        """
        self.config = config
        self.video_meta = video_meta
        self.queue: queue.Queue[Tuple[Any, int, int, float] | None] = queue.Queue(maxsize=128)
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

        fps = max(1, round(self.video_meta["fps"])) if self.video_meta["fps"] else 24
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

    def __enter__(self) -> VideoWriterWrapper:
        """Context manager entry; automatically starts the background writer worker."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit; automatically stops and clean up writer worker threads."""
        self.stop()

    def start(self) -> None:
        """Spawns and starts the background writer worker thread."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.thread.start()
        logger.info("Video writer thread started.")

    def stop(self) -> None:
        """Pushes stop sentinel, waits for writer to finish queue items, and releases outputs."""
        if self.thread and self.thread.is_alive():
            # Send stop sentinel
            try:
                self.queue.put(None, block=True, timeout=2.0)
            except queue.Full:
                pass
            self.thread.join(timeout=10.0)
            logger.info("Video writer thread joined.")

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Released VideoWriter resource.")

    def write_result(self, result: Any, count: int, frame_idx: int, timestamp_sec: float) -> None:
        """Enqueues inference results for plotting and writing.

        Args:
            result: YOLO Result object.
            count: Number of heads detected.
            frame_idx: Index of the current frame.
            timestamp_sec: Frame time offset from the start of the video.
        """
        try:
            self.queue.put((result, count, frame_idx, timestamp_sec), block=True, timeout=1.0)
        except queue.Full:
            logger.warning("Writer queue full. Dropping output frame index: %d", frame_idx)

    def _annotate_frame(self, result: Any, count: int, frame_idx: int, timestamp_sec: float) -> np.ndarray:
        """Draws bounding boxes, head counts, frame index, and timestamps on a copy of the frame.

        Args:
            result: YOLO Result object.
            count: Number of heads detected.
            frame_idx: Current frame index.
            timestamp_sec: Frame time stamp in seconds.

        Returns:
            A NumPy array of the annotated image.
        """
        overlay_cfg = self.config.output.overlay
        if not overlay_cfg.enabled:
            return result.orig_img.copy()

        # Plot boxes on the frame without labels/conf to maintain a clean display
        annotated = result.plot(
            labels=False,
            conf=False,
            boxes=True,
            line_width=overlay_cfg.thickness,
        )

        font_scale = overlay_cfg.font_scale
        thickness = overlay_cfg.thickness

        # Draw red head count text at top-left
        cv2.putText(
            annotated,
            f"Cabecas Detectadas: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.5,
            (0, 0, 255),
            thickness + 1,
            cv2.LINE_AA,
        )

        # Draw white frame index and timestamp at bottom-left
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
        """Background loop pulling tasks, running overlays, resizing, and writing frames/snapshots."""
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
                
                # Render annotations
                annotated = self._annotate_frame(result, count, frame_idx, timestamp_sec)

                # Resize to target resolution if needed
                if self._writer is not None:
                    if annotated.shape[1] != self.out_w or annotated.shape[0] != self.out_h:
                        annotated = cv2.resize(annotated, (self.out_w, self.out_h))
                    self._writer.write(annotated)

                # Save snapshot if matching interval
                if snapshot_every > 0 and frame_idx % snapshot_every == 0:
                    filename = snapshots_dir / f"snapshot_frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(filename), annotated)

        except Exception as e:
            logger.error(f"Error in video writer thread: {e}", exc_info=True)
