#!/usr/bin/env python3
"""
Standalone CLI Utility for RGBT Image Equalization & Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equalizes resolution, preserves aspect ratio via letterbox, and applies
thermal contrast enhancement (CLAHE) on RGB and Thermal image pairs.

Usage:
    python preprocess_images.py --rgb input/images/DJI_0789_W.JPG \
                                --thermal input/images/DJI_0790_T.JPG \
                                --output input/images_equalized/
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
        description="Standalone CLI Utility for RGBT Dual-Stream Image Equalization"
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
    print("       RGBT DUAL-STREAM IMAGE EQUALIZER (DECOUPLED)")
    print("=" * 60)
    print(f"RGB Input:      {rgb_path}")
    print(f"Thermal Input:  {thermal_path}")
    print(f"Target Output:  {output_dir}")
    print(f"Target Size:    {args.width}x{args.height} px")
    print(f"Thermal CLAHE:  {not args.no_clahe}")
    print(f"Letterboxing:   {not args.no_letterbox}")
    print("=" * 60)

    equalizer = RGBTImageEqualizer(
        target_size=(args.width, args.height),
        keep_aspect_ratio=not args.no_letterbox,
        thermal_clahe=not args.no_clahe,
    )

    out_rgb, out_thermal = equalizer.process_files(
        rgb_path=rgb_path,
        thermal_path=thermal_path,
        output_dir=output_dir,
    )

    print("\n[SUCCESS] Image equalization completed!")
    print(f" ├─ Equalized RGB:     {out_rgb}")
    print(f" └─ Equalized Thermal: {out_thermal}\n")


if __name__ == "__main__":
    main()
