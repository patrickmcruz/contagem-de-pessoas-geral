"""
RGBT Image Equalization & Preprocessing Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides a decoupled, independent preprocessing class (`RGBTImageEqualizer`)
to equalize, crop, scale, and contrast-enhance dual-modality (RGB + Thermal) image pairs.

Key Capabilities:
- Spatial Resampling & Resolution Standardization (e.g., matching target 1280x1024).
- Aspect Ratio Preservation with Letterbox Padding.
- Thermal Dynamic Range & CLAHE Contrast Enhancement.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RGBTImageEqualizer:
    """Decoupled image preprocessor for dual-stream RGB and Thermal image pairs.

    Attributes:
        target_size: Target (width, height) tuple for equalized output matrices.
        keep_aspect_ratio: If True, uses letterboxing to avoid aspect distortion.
        thermal_clahe: If True, applies CLAHE contrast enhancement to the Thermal image.
        clahe_clip_limit: Threshold limit for contrast limiting in CLAHE.
        clahe_tile_grid: Grid size for histogram equalization (e.g. (8, 8)).
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (1280, 1024),
        keep_aspect_ratio: bool = True,
        thermal_clahe: bool = True,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
    ):
        """Initializes RGBTImageEqualizer settings.

        Args:
            target_size: Target (width, height) resolution tuple.
            keep_aspect_ratio: Preserves native aspect ratio using padding.
            thermal_clahe: Enables CLAHE contrast boost on thermal images.
            clahe_clip_limit: CLAHE clip limit.
            clahe_tile_grid: CLAHE tile grid dimensions tuple.
        """
        self.target_w, self.target_h = target_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.thermal_clahe = thermal_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid

        self._clahe = (
            cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
            if self.thermal_clahe
            else None
        )

    def _letterbox_resize(self, img: np.ndarray) -> np.ndarray:
        """Resizes an image preserving aspect ratio with zero-padding (letterbox).

        Args:
            img: Input BGR NumPy array image.

        Returns:
            Resized and padded BGR NumPy array matching (target_h, target_w, 3).
        """
        h, w = img.shape[:2]
        if (w, h) == (self.target_w, self.target_h):
            return img.copy()

        if not self.keep_aspect_ratio:
            return cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

        scale = min(self.target_w / w, self.target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        # Create canvas with neutral dark background
        canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)

        # Center resized image on canvas
        top = (self.target_h - new_h) // 2
        left = (self.target_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized

        return canvas

    def enhance_thermal(self, thermal_img: np.ndarray) -> np.ndarray:
        """Applies CLAHE dynamic range enhancement to a thermal image.

        Args:
            thermal_img: Input BGR NumPy array thermal image.

        Returns:
            Enhanced BGR NumPy array thermal image.
        """
        if not self.thermal_clahe or self._clahe is None:
            return thermal_img.copy()

        # Convert to LAB color space to equalize luminosity without altering tint
        lab = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def process_pair(
        self, rgb_img: np.ndarray, thermal_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Equalizes an RGB and Thermal image pair to matching target dimensions.

        Args:
            rgb_img: Raw RGB NumPy array image.
            thermal_img: Raw Thermal NumPy array image.

        Returns:
            A tuple (rgb_equalized, thermal_equalized) with identical shape (target_h, target_w, 3).
        """
        # 1. Enhance thermal contrast if enabled
        enhanced_thermal = self.enhance_thermal(thermal_img)

        # 2. Letterbox resize both images to exact target resolution
        rgb_eq = self._letterbox_resize(rgb_img)
        thermal_eq = self._letterbox_resize(enhanced_thermal)

        return rgb_eq, thermal_eq

    def process_files(
        self,
        rgb_path: str | Path,
        thermal_path: str | Path,
        output_dir: str | Path,
    ) -> Tuple[Path, Path]:
        """Loads RGB and Thermal files, equalizes them, and saves the output files to disk.

        Args:
            rgb_path: Path to the raw RGB input image file.
            thermal_path: Path to the raw Thermal input image file.
            output_dir: Output directory to save the equalized images.

        Returns:
            A tuple of Paths (out_rgb_path, out_thermal_path).
        """
        rgb_path = Path(rgb_path)
        thermal_path = Path(thermal_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            raise RuntimeError(f"Failed to read RGB image file: {rgb_path}")

        thermal_img = cv2.imread(str(thermal_path))
        if thermal_img is None:
            raise RuntimeError(f"Failed to read Thermal image file: {thermal_path}")

        rgb_eq, thermal_eq = self.process_pair(rgb_img, thermal_img)

        out_rgb_path = output_dir / f"{rgb_path.stem}_equalized.JPG"
        out_thermal_path = output_dir / f"{thermal_path.stem}_equalized.JPG"

        cv2.imwrite(str(out_rgb_path), rgb_eq)
        cv2.imwrite(str(out_thermal_path), thermal_eq)

        logger.info(f"Saved equalized RGB image: {out_rgb_path} ({self.target_w}x{self.target_h})")
        logger.info(f"Saved equalized Thermal image: {out_thermal_path} ({self.target_w}x{self.target_h})")

        return out_rgb_path, out_thermal_path
