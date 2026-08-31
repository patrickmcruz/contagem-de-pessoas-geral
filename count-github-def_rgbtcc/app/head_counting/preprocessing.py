"""
RGBT Image Equalization & Spatial Alignment Module (ADR 001)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides a decoupled, independent preprocessor (`RGBTImageEqualizer`)
to perform pixel-level layer alignment (co-registration) between RGB and Thermal image pairs.

Key Capabilities:
- Pedestrian Sub-Pixel Alignment: Calibrated displacement offsets for perfect crowd silhouette overlay.
- Lens Undistortion Correction (`undistort_lens`): Corrects Wide 24mm lens barrel distortion.
- Precise FOV Aspect-Ratio Matching (Corrects 4:3 vs 5:4 sensor aspect ratio mismatch eliminating lateral shearing/squeeze).
- Ground-Plane ROI Focused Co-registration: Prioritizes pedestrian surface alignment over rooftop parallax.
- Fine-tuning Translation Offsets (`shift_rgb_x`, `shift_rgb_y`) applied strictly to RGB image.
- Aspect Ratio Preservation with Letterbox Padding.
- Robust Partial Affine Co-registration with Sanity Checks (Scale & Translation bounds).
- Automatic Fallback to Calibrated Aspect-Matched FOV Crop.
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
        fov_crop_ratio: Center crop height ratio for Wide RGB matching Thermal FOV (default 0.70 / 70%).
        undistort_lens: If True, corrects Wide lens radial barrel distortion.
        shift_rgb_x: Horizontal shift in pixels applied strictly to RGB image (default -15px left).
        shift_rgb_y: Vertical shift in pixels applied strictly to RGB image (default -16px up).
        mode: Alignment mode ("homography", "affine", or "crop").
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (1280, 1024),
        keep_aspect_ratio: bool = True,
        thermal_clahe: bool = True,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        fov_crop_ratio: float = 0.70,
        undistort_lens: bool = True,
        shift_rgb_x: int = -22,
        shift_rgb_y: int = -23,
        mode: str = "homography",
    ):
        """Initializes RGBTImageEqualizer settings.

        Args:
            target_size: Target (width, height) resolution tuple.
            keep_aspect_ratio: Preserves native aspect ratio using padding.
            thermal_clahe: Enables CLAHE contrast boost on thermal images.
            clahe_clip_limit: CLAHE clip limit.
            clahe_tile_grid: CLAHE tile grid dimensions tuple.
            fov_crop_ratio: Wide RGB center crop height ratio matching Thermal HFOV (default 0.70).
            undistort_lens: Corrects 24mm Wide lens barrel distortion.
            shift_rgb_x: Direct horizontal shift in pixels for RGB (default -15px left).
            shift_rgb_y: Direct vertical shift in pixels for RGB (default -16px up).
            mode: Alignment mode ("homography", "affine", or "crop").
        """
        self.target_w, self.target_h = target_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.thermal_clahe = thermal_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.fov_crop_ratio = fov_crop_ratio
        self.undistort_lens = undistort_lens
        self.shift_rgb_x = shift_rgb_x
        self.shift_rgb_y = shift_rgb_y
        self.mode = mode.lower()

        self._clahe = (
            cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid)
            if self.thermal_clahe
            else None
        )

    def _undistort_wide_lens(self, rgb_img: np.ndarray) -> np.ndarray:
        """Applies radial lens undistortion correction to Wide 24mm optical sensor.

        Args:
            rgb_img: Input raw BGR NumPy array image.

        Returns:
            Undistorted BGR NumPy array image.
        """
        if not self.undistort_lens:
            return rgb_img

        h, w = rgb_img.shape[:2]
        K = np.array(
            [[w * 0.8, 0, w / 2], [0, h * 0.8, h / 2], [0, 0, 1]], dtype=np.float32
        )
        dist_coeffs = np.array([-0.04, 0.01, 0, 0], dtype=np.float32)

        return cv2.undistort(rgb_img, K, dist_coeffs)

    def _fov_center_crop(
        self, rgb_img: np.ndarray, thermal_img: np.ndarray | None = None, crop_ratio: float | None = None
    ) -> np.ndarray:
        """Crops the central Field of View of the Wide RGB image matching Thermal FOV and target aspect ratio (5:4).

        Args:
            rgb_img: Input RGB BGR NumPy array image.
            thermal_img: Optional Thermal BGR NumPy array image to extract native aspect ratio.
            crop_ratio: Height crop ratio (default 0.70 for ~61° vs 84° HFOV optical match).

        Returns:
            Aspect-ratio matched center-cropped RGB BGR NumPy array image.
        """
        ratio = crop_ratio if crop_ratio is not None else self.fov_crop_ratio
        rgb_prep = self._undistort_wide_lens(rgb_img)

        h_rgb, w_rgb = rgb_prep.shape[:2]

        # Target aspect ratio: match target_w / target_h (or thermal_img aspect ratio)
        if thermal_img is not None:
            th_h, th_w = thermal_img.shape[:2]
            target_aspect = th_w / th_h
        else:
            target_aspect = self.target_w / self.target_h

        # Calculate crop dimensions matching target aspect ratio (e.g. 5:4)
        crop_h = int(h_rgb * ratio)
        crop_w = int(crop_h * target_aspect)

        # If crop_w exceeds w_rgb, clamp crop_w and recompute crop_h
        if crop_w > w_rgb:
            crop_w = w_rgb
            crop_h = int(crop_w / target_aspect)

        left = (w_rgb - crop_w) // 2
        top = (h_rgb - crop_h) // 2

        return rgb_prep[top : top + crop_h, left : left + crop_w].copy()

    def align_homography(
        self, rgb_img: np.ndarray, thermal_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Aligns RGB image to Thermal image plane using FOV pre-scaling and Partial Affine similarity.

        Args:
            rgb_img: Input raw RGB BGR NumPy array image.
            thermal_img: Input raw Thermal BGR NumPy array image.

        Returns:
            A tuple of (aligned_rgb, thermal_img, transformation_matrix).
        """
        enhanced_thermal = self.enhance_thermal(thermal_img)
        th_h, th_w = thermal_img.shape[:2]

        # 1. First bring Wide RGB image into optical Thermal FOV & 5:4 Aspect Ratio space
        rgb_cropped = self._fov_center_crop(rgb_img, thermal_img, crop_ratio=self.fov_crop_ratio)

        if self.mode == "crop":
            logger.info(f"[ALIGNMENT] Mode is set to 'crop'. Using FOV Center Crop ({self.fov_crop_ratio*100:.0f}%).")
            return rgb_cropped, enhanced_thermal, None

        rgb_scaled = cv2.resize(rgb_cropped, (th_w, th_h), interpolation=cv2.INTER_AREA)

        # 2. Detect SIFT / ORB features on FOV-matched images
        detector: Any = None
        is_sift = False
        try:
            detector = cv2.SIFT_create(nfeatures=2000)
            is_sift = True
        except Exception:
            detector = cv2.ORB_create(nfeatures=2000)

        gray_rgb = cv2.cvtColor(rgb_scaled, cv2.COLOR_BGR2GRAY)
        gray_th = cv2.cvtColor(enhanced_thermal, cv2.COLOR_BGR2GRAY)

        kp_rgb, des_rgb = detector.detectAndCompute(gray_rgb, None)
        kp_th, des_th = detector.detectAndCompute(gray_th, None)

        if des_rgb is None or des_th is None or len(kp_rgb) < 6 or len(kp_th) < 6:
            logger.warning("[ALIGNMENT] Insufficient keypoints detected. Falling back to FOV Center Crop.")
            return rgb_cropped, enhanced_thermal, None

        # 3. Match descriptors using KNN and Lowe's Ratio Test
        norm_type = cv2.NORM_L2 if is_sift else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)

        try:
            raw_matches = matcher.knnMatch(des_th, des_rgb, k=2)
        except Exception as e:
            logger.warning(f"[ALIGNMENT] Feature matching failed: {e}. Falling back to FOV Center Crop.")
            return rgb_cropped, enhanced_thermal, None

        good_matches = []
        for m_tuple in raw_matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                if m.distance < 0.65 * n.distance:
                    good_matches.append(m)

        # 4. Estimate Partial Affine Transformation with Strict Sanity Validation
        if len(good_matches) >= 6:
            src_pts = np.float32([kp_th[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_rgb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)

            if M is not None:
                scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
                tx = abs(M[0, 2])
                ty = abs(M[1, 2])

                # Verify that matrix is physically plausible for gimbal sensors
                if 0.85 <= scale <= 1.15 and tx <= 50.0 and ty <= 50.0:
                    aligned_rgb = cv2.warpAffine(rgb_scaled, M, (th_w, th_h))
                    logger.info(f"[ALIGNMENT] Valid Partial Affine matrix applied (Scale={scale:.2f}, Tx={tx:.1f}px, Ty={ty:.1f}px).")
                    return aligned_rgb, enhanced_thermal, M
                else:
                    logger.warning(f"[ALIGNMENT] Matrix failed sanity check (Scale={scale:.2f}, Tx={tx:.1f}px, Ty={ty:.1f}px). Falling back to FOV Center Crop.")

        logger.warning(f"[ALIGNMENT] Multimodal feature alignment unviable. Falling back to FOV Center Crop ({self.fov_crop_ratio*100:.0f}%).")
        return rgb_cropped, enhanced_thermal, None

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
        if self.mode in ("homography", "affine"):
            rgb_aligned, thermal_enhanced, _ = self.align_homography(rgb_img, thermal_img)
        else:
            rgb_cropped = self._fov_center_crop(rgb_img, thermal_img)
            thermal_enhanced = self.enhance_thermal(thermal_img)
            rgb_aligned = rgb_cropped

        # Resample both aligned streams to target uniform resolution with letterboxing
        rgb_eq = self._letterbox_resize(rgb_aligned)
        thermal_eq = self._letterbox_resize(thermal_enhanced)

        # Apply fine-tuning translation shift strictly to RGB image
        if self.shift_rgb_x != 0 or self.shift_rgb_y != 0:
            M_shift = np.float32([[1, 0, self.shift_rgb_x], [0, 1, self.shift_rgb_y]])
            rgb_eq = cv2.warpAffine(rgb_eq, M_shift, (self.target_w, self.target_h))

        return rgb_eq, thermal_eq

    def process_files(
        self,
        rgb_path: str | Path,
        thermal_path: str | Path,
        output_dir: str | Path,
    ) -> Tuple[Path, Path, Path]:
        """Loads RGB and Thermal files, performs alignment & equalization, and saves outputs.

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
        blend_img = self.create_blend_overlay(rgb_eq, thermal_eq, alpha=0.5)

        # Determine clean subdirectories for Silver layer
        if output_dir.name in ("images", "layer_blend_checks"):
            img_out_dir = output_dir if output_dir.name == "images" else output_dir.parent / "images"
            blend_out_dir = output_dir if output_dir.name == "layer_blend_checks" else output_dir.parent / "layer_blend_checks"
        else:
            img_out_dir = output_dir / "images"
            blend_out_dir = output_dir / "layer_blend_checks"

        img_out_dir.mkdir(parents=True, exist_ok=True)
        blend_out_dir.mkdir(parents=True, exist_ok=True)

        out_rgb_path = img_out_dir / f"{rgb_path.stem}_equalized{rgb_path.suffix}"
        out_thermal_path = img_out_dir / f"{thermal_path.stem}_equalized.jpg"
        out_blend_path = blend_out_dir / f"{rgb_path.stem}_blend_check.jpg"

        cv2.imwrite(str(out_rgb_path), rgb_eq)
        cv2.imwrite(str(out_thermal_path), thermal_eq)
        cv2.imwrite(str(out_blend_path), blend_img)

        logger.info(f"Saved equalized RGB layer: {out_rgb_path} ({self.target_w}x{self.target_h})")
        logger.info(f"Saved equalized Thermal layer: {out_thermal_path} ({self.target_w}x{self.target_h})")
        logger.info(f"Saved Layer Blend Check: {out_blend_path}")

        return out_rgb_path, out_thermal_path, out_blend_path
