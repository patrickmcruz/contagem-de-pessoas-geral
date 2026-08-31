#!/usr/bin/env python3
"""
Standalone CLI Utility for RGBT Image Equalization & Spatial Alignment (ADR 001)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equalizes resolution, performs Homography Layer Alignment (SIFT/ORB + RANSAC),
preserves aspect ratio via letterbox, and generates 50%/50% Layer Blend Check overlays.

Usage:
    python preprocess_images.py --rgb input/images/DJI_0789_W.JPG \
                                --thermal input/images/DJI_0790_T.JPG \
                                --output input/images_equalized/ \
                                --mode homography
"""

import argparse
import sys
from pathlib import Path

# Add app directory to sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from head_counting.preprocessing import RGBTImageEqualizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone CLI Utility for RGBT Dual-Stream Image Homography Alignment (ADR 001)"
    )
    parser.add_argument(
        "--rgb",
        type=str,
        default=str(APP_DIR / "input/images/DJI_0789_W.JPG"),
        help="Path to raw RGB image file",
    )
    parser.add_argument(
        "--thermal",
        type=str,
        default=str(APP_DIR / "input/images/DJI_0790_T.JPG"),
        help="Path to raw Thermal image file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(APP_DIR / "input/images_equalized"),
        help="Target output directory for equalized image pair",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="homography",
        choices=["homography", "crop"],
        help="Alignment mode: 'homography' (ADR 001 SIFT/RANSAC) or 'crop' (FOV Center Crop)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target equalized width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Target equalized height in pixels (default: 1024)",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE thermal contrast enhancement",
    )
    parser.add_argument(
        "--no-letterbox",
        action="store_true",
        help="Disable aspect-ratio letterboxing (stretch resize)",
    )

    args = parser.parse_args()

    rgb_path = Path(args.rgb)
    thermal_path = Path(args.thermal)
    output_dir = Path(args.output)

    print("=" * 60)
    print("       RGBT LAYER HOMOGRAPHY EQUALIZER (ADR 001)")
    print("=" * 60)
    print(f"RGB Input:      {rgb_path}")
    print(f"Thermal Input:  {thermal_path}")
    print(f"Target Output:  {output_dir}")
    print(f"Target Size:    {args.width}x{args.height} px")
    print(f"Alignment Mode: {args.mode.upper()} (SIFT/ORB + RANSAC)")
    print(f"Thermal CLAHE:  {not args.no_clahe}")
    print(f"Letterboxing:   {not args.no_letterbox}")
    print("=" * 60)

    equalizer = RGBTImageEqualizer(
        target_size=(args.width, args.height),
        keep_aspect_ratio=not args.no_letterbox,
        thermal_clahe=not args.no_clahe,
        mode=args.mode,
    )

    out_rgb, out_thermal, out_blend = equalizer.process_files(
        rgb_path=rgb_path,
        thermal_path=thermal_path,
        output_dir=output_dir,
    )

    print("\n[SUCCESS] Homography layer equalization completed!")
    print(f" ├─ Equalized RGB Layer:     {out_rgb}")
    print(f" ├─ Equalized Thermal Layer: {out_thermal}")
    print(f" └─ Layer Blend Check:       {out_blend}\n")


if __name__ == "__main__":
    main()
