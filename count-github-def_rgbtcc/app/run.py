#!/usr/bin/env python
"""
Command Line Interface to execute the Head Counting Pipeline.
Usage:
    python run.py --config data_day.yaml
    python run.py --config data_night.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

from head_counting import run_pipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("head_counting_runner")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26 Head Counting - High Performance Pipeline")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="data_day.yaml",
        help="Path to the YAML configuration file (default: data_day.yaml)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file does not exist: {config_path}")
        sys.exit(1)

    logger.info(f"Starting pipeline execution with config: {config_path.name}")
    try:
        summary = run_pipeline(config_path)
        logger.info("Pipeline executed successfully!")
        
        # Print a short report summary to console
        counts = summary.get("counts", {})
        print("\n" + "=" * 50)
        print("                 EXECUTION REPORT SUMMARY")
        print("=" * 50)
        print(f"App Name:      {summary.get('app')}")
        print(f"Processed:     {summary.get('processed_frames')} frames in {summary.get('elapsed_sec')}s")
        print(f"Average FPS:   {summary.get('fps_processed')} FPS")
        print("-" * 50)
        print("Crowd Statistics:")
        print(f"  ├─ Min Count: {counts.get('min_people_in_frame')}")
        print(f"  ├─ Max Count: {counts.get('max_people_in_frame')}")
        print(f"  ├─ Mean Count: {counts.get('mean_people_per_frame')}")
        print(f"  ├─ Median Count: {counts.get('median_people_per_frame')}")
        print(f"  └─ P95 Count: {counts.get('p95_people_per_frame')}")
        print("=" * 50 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
