"""
RGBT Image Equalization & Spatial Alignment Module (ADR 001)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides a decoupled, independent preprocessor (`RGBTImageEqualizer`)
to perform pixel-level layer alignment (co-registration) between RGB and Thermal image pairs.

Key Capabilities:
- Automatic Homography Alignment via Feature Matching (SIFT/ORB + RANSAC).
- Graceful Fallback to FOV Center Crop when features are insufficient.
- Spatial Resampling & Resolution Standardization (e.g., matching target 1280x1024).
- Aspect Ratio Preservation with Letterbox Padding.
- Thermal Dynamic Range & CLAHE Contrast Enhancement.
- Layer Blend Check Overlay Generation (50% RGB + 50% Thermal).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RGBTImageEqualizer:
    """Decoupled image preprocessor for dual-stream RGB and Thermal image layer alignment.

    Attributes:
        target_size: Target (width, height) tuple for equalized output matrices.
        keep_aspect_ratio: If True, uses letterboxing to avoid aspect distortion.
        thermal_clahe: If True, applies CLAHE contrast enhancement to the Thermal image.
        clahe_clip_limit: Threshold limit for contrast limiting in CLAHE.
        clahe_tile_grid: Grid size for histogram equalization (e.g. (8, 8)).
        mode: Alignment mode ("homography" or "crop").
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (1280, 1024),
        keep_aspect_ratio: bool = True,
        thermal_clahe: bool = True,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        mode: str = "homography",
    ):
        """Initializes RGBTImageEqualizer settings.

        Args:
            target_size: Target (width, height) resolution tuple.
            keep_aspect_ratio: Preserves native aspect ratio using padding.
            thermal_clahe: Enables CLAHE contrast boost on thermal images.
            clahe_clip_limit: CLAHE clip limit.
            clahe_tile_grid: CLAHE tile grid dimensions tuple.
            mode: Alignment mode ("homography" or "crop").
        """
        self.target_w, self.target_h = target_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.thermal_clahe = thermal_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.mode = mode.lower()

        self._clahe = (
            cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
            if self.thermal_clahe
            else None
        )

    def _fov_center_crop(self, rgb_img: np.ndarray, crop_ratio: float = 0.75) -> np.ndarray:
        """Crops the central Field of View of the RGB image matching the narrower Thermal sensor.

        Args:
            rgb_img: Input RGB BGR NumPy array image.
            crop_ratio: Center crop ratio (default 0.75 for ~61° vs 84° FOV).

        Returns:
            Center-cropped RGB BGR NumPy array image.
        """
        h, w = rgb_img.shape[:2]
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)

        left = (w - crop_w) // 2
        top = (h - crop_h) // 2

        return rgb_img[top : top + crop_h, left : left + crop_w].copy()

    def align_homography(
        self, rgb_img: np.ndarray, thermal_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Aligns RGB image to Thermal image plane using SIFT/ORB feature matching & RANSAC.

        Args:
            rgb_img: Input raw RGB BGR NumPy array image.
            thermal_img: Input raw Thermal BGR NumPy array image.

        Returns:
            A tuple of (warped_rgb, thermal_img, H_matrix).
        """
        # Enhance thermal image first to improve feature detection
        enhanced_thermal = self.enhance_thermal(thermal_img)

        # 1. Initialize feature detector (SIFT preferred, fallback to ORB)
        detector: Any = None
        is_sift = False
        try:
            detector = cv2.SIFT_create(nfeatures=3000)
            is_sift = True
        except Exception:
            detector = cv2.ORB_create(nfeatures=3000)

        # 2. Detect keypoints and descriptors
        gray_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        gray_th = cv2.cvtColor(enhanced_thermal, cv2.COLOR_BGR2GRAY)

        kp_rgb, des_rgb = detector.detectAndCompute(gray_rgb, None)
        kp_th, des_th = detector.detectAndCompute(gray_th, None)

        if des_rgb is None or des_th is None or len(kp_rgb) < 4 or len(kp_th) < 4:
            logger.warning("[HOMOGRAPHY] Insufficient keypoints detected. Falling back to FOV Center Crop.")
            cropped_rgb = self._fov_center_crop(rgb_img)
            return cropped_rgb, enhanced_thermal, None

        # 3. Match descriptors using KNN
        norm_type = cv2.NORM_L2 if is_sift else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)

        try:
            raw_matches = matcher.knnMatch(des_th, des_rgb, k=2)
        except Exception as e:
            logger.warning(f"[HOMOGRAPHY] Feature matching failed: {e}. Falling back to FOV Center Crop.")
            cropped_rgb = self._fov_center_crop(rgb_img)
            return cropped_rgb, enhanced_thermal, None

        # 4. Apply Lowe's Ratio Test
        good_matches = []
        for m_tuple in raw_matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        logger.info(f"[HOMOGRAPHY] Found {len(good_matches)} valid feature matches.")

        # 5. Estimate Homography Matrix if enough good matches exist
        if len(good_matches) >= 4:
            src_pts = np.float32([kp_th[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_rgb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if H is not None:
                th_h, th_w = thermal_img.shape[:2]
                warped_rgb = cv2.warpPerspective(rgb_img, H, (th_w, th_h))
                logger.info("[HOMOGRAPHY] Homography matrix H successfully estimated and applied!")
                return warped_rgb, enhanced_thermal, H

        logger.warning("[HOMOGRAPHY] RANSAC failed to estimate robust H matrix. Falling back to FOV Center Crop.")
        cropped_rgb = self._fov_center_crop(rgb_img)
        return cropped_rgb, enhanced_thermal, None

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

        canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
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

        lab = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def create_blend_overlay(
        self, rgb_img: np.ndarray, thermal_img: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Creates a 50%/50% alpha blended overlay of RGB and Thermal images for alignment verification.

        Args:
            rgb_img: Equalized RGB BGR NumPy array image.
            thermal_img: Equalized Thermal BGR NumPy array image.
            alpha: Alpha blending weight (default 0.5).

        Returns:
            Blended BGR NumPy array image.
        """
        if rgb_img.shape != thermal_img.shape:
            thermal_resized = cv2.resize(thermal_img, (rgb_img.shape[1], rgb_img.shape[0]))
        else:
            thermal_resized = thermal_img

        return cv2.addWeighted(rgb_img, 1.0 - alpha, thermal_resized, alpha, 0)

    def process_pair(
        self, rgb_img: np.ndarray, thermal_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Equalizes and aligns an RGB and Thermal image pair to matching target dimensions.

        Args:
            rgb_img: Raw RGB NumPy array image.
            thermal_img: Raw Thermal NumPy array image.

        Returns:
            A tuple (rgb_aligned, thermal_aligned) with identical shape (target_h, target_w, 3).
        """
        if self.mode == "homography":
            rgb_aligned, thermal_enhanced, _ = self.align_homography(rgb_img, thermal_img)
        else:
            rgb_cropped = self._fov_center_crop(rgb_img)
            thermal_enhanced = self.enhance_thermal(thermal_img)
            rgb_aligned = rgb_cropped

        # Resample both aligned streams to target uniform resolution
        rgb_eq = self._letterbox_resize(rgb_aligned)
        thermal_eq = self._letterbox_resize(thermal_enhanced)

        return rgb_eq, thermal_eq

    def process_files(
        self,
        rgb_path: str | Path,
        thermal_path: str | Path,
        output_dir: str | Path,
    ) -> Tuple[Path, Path, Path]:
        """Loads RGB and Thermal files, performs homography alignment & equalization, and saves outputs.

        Args:
            rgb_path: Path to the raw RGB input image file.
            thermal_path: Path to the raw Thermal input image file.
            output_dir: Output directory to save the equalized images and blend check.

        Returns:
            A tuple of Paths (out_rgb_path, out_thermal_path, out_blend_path).
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

        # Create Layer Blend Overlay Check
        blend_img = self.create_blend_overlay(rgb_eq, thermal_eq, alpha=0.5)

        out_rgb_path = output_dir / f"{rgb_path.stem}_equalized.JPG"
        out_thermal_path = output_dir / f"{thermal_path.stem}_equalized.JPG"
        out_blend_path = output_dir / "layer_blend_check.jpg"

        cv2.imwrite(str(out_rgb_path), rgb_eq)
        cv2.imwrite(str(out_thermal_path), thermal_eq)
        cv2.imwrite(str(out_blend_path), blend_img)

        logger.info(f"Saved equalized RGB layer: {out_rgb_path} ({self.target_w}x{self.target_h})")
        logger.info(f"Saved equalized Thermal layer: {out_thermal_path} ({self.target_w}x{self.target_h})")
        logger.info(f"Saved Layer Blend Check: {out_blend_path}")

        return out_rgb_path, out_thermal_path, out_blend_path
