#!/usr/bin/env python3
"""
RGBT Image Equalization & Spatial Alignment CLI Tool (ADR 001)
=============================================================

Independent CLI tool to preprocess raw dual-stream RGB and Thermal images before
submitting them to the head_counting pipeline.

Usage:
    python preprocess_images.py --rgb data/bronze/images/DJI_0789_W.JPG \\
                                --thermal data/bronze/images/DJI_0790_T.JPG \\
                                --output data/silver/
"""

import sys
import argparse
import logging
from pathlib import Path

# Add app directory to sys.path for local module resolution
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from head_counting.preprocessing import RGBTImageEqualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_images")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RGBT Layer Equalizer & Spatial Alignment Tool (Medallion Architecture)"
    )
    parser.add_argument(
        "--rgb",
        type=str,
        default=str(APP_DIR / "data/bronze/images/DJI_0789_W.JPG"),
        help="Path to input raw RGB image file (Bronze Layer)",
    )
    parser.add_argument(
        "--thermal",
        type=str,
        default=str(APP_DIR / "data/bronze/images/DJI_0790_T.JPG"),
        help="Path to input raw Thermal image file (Bronze Layer)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(APP_DIR / "data/silver"),
        help="Path to output directory for equalized layers (Silver Layer)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["homography", "affine", "crop"],
        default="homography",
        help="Alignment algorithm mode (default: homography / affine similarity)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target resolution width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Target resolution height in pixels (default: 1024)",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE contrast boost on thermal images",
    )
    parser.add_argument(
        "--no-letterbox",
        action="store_true",
        help="Disable letterboxing padding (force resize stretch)",
    )

    args = parser.parse_args()

    rgb_path = Path(args.rgb)
    thermal_path = Path(args.thermal)
    output_dir = Path(args.output)

    print("============================================================")
    print("       RGBT LAYER EQUALIZER & ALIGNMENT (MEDALLION BRONZE -> SILVER)")
    print("============================================================")
    print(f"RGB Input (Bronze):    {rgb_path}")
    print(f"Thermal Input (Bronze):{thermal_path}")
    print(f"Target Output (Silver):{output_dir}")
    print(f"Target Size:           {args.width}x{args.height} px")
    print(f"Alignment Mode:        {args.mode.upper()}")
    print(f"Thermal CLAHE:         {not args.no_clahe}")
    print(f"Letterboxing:          {not args.no_letterbox}")
    print("============================================================")

    if not rgb_path.exists():
        logger.error(f"RGB input file not found: {rgb_path}")
        return 1
    if not thermal_path.exists():
        logger.error(f"Thermal input file not found: {thermal_path}")
        return 1

    equalizer = RGBTImageEqualizer(
        target_size=(args.width, args.height),
        keep_aspect_ratio=not args.no_letterbox,
        thermal_clahe=not args.no_clahe,
        mode=args.mode,
    )

    try:
        out_rgb, out_th, out_blend = equalizer.process_files(
            rgb_path=rgb_path,
            thermal_path=thermal_path,
            output_dir=output_dir,
        )
        print("\n[SUCCESS] Medallion Silver layer equalization completed!")
        print(f" |-- Equalized RGB Layer:     {out_rgb}")
        print(f" |-- Equalized Thermal Layer: {out_th}")
        print(f" \-- Layer Blend Check:       {out_blend}")
        return 0
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
