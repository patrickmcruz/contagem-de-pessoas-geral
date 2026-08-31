#!/usr/bin/env python
"""
Command Line Interface to execute the DEF-rgbtcc Dual-Stream RGBT Crowd Counting Pipeline
with Medallion Data Architecture support (Bronze -> Silver -> Gold).

Usage:
    python run.py --config data_rgbt_images.yaml
    python run.py --config data_rgbt_images.yaml --medallion --layer all
    python run.py --config data_rgbt_images.yaml --layer silver
"""
import argparse
import logging
import sys
from pathlib import Path

from head_counting import PipelineConfig, CountingPipeline, MedallionPipelineRunner, run_pipeline

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
    parser = argparse.ArgumentParser(description="DEF-rgbtcc Dual-Stream RGBT Crowd Counting Pipeline")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="data_rgbt_day.yaml",
        help="Path to the YAML configuration file (default: data_rgbt_day.yaml)"
    )
    parser.add_argument(
        "-m", "--medallion",
        action="store_true",
        help="Enables Medallion Data Architecture (Bronze -> Silver -> Gold) execution"
    )
    parser.add_argument(
        "-l", "--layer",
        type=str,
        choices=["all", "silver", "gold"],
        default="all",
        help="Target Medallion layer to execute ('all', 'silver', 'gold')"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file does not exist: {config_path}")
        sys.exit(1)

    logger.info(f"Starting pipeline execution with config: {config_path.name}")
    try:
        config = PipelineConfig.from_yaml(config_path)

        if args.medallion or args.layer != "all":
            runner = MedallionPipelineRunner(config=config, config_path=config_path)
            if args.layer == "silver":
                silver_rgb, silver_thermal = runner.run_silver()
                logger.info(f"Silver layer processing completed: {silver_rgb}")
                return
            elif args.layer == "gold":
                summary = runner.run_gold()
            else:
                summary = runner.run_all()
        else:
            pipeline = CountingPipeline(config=config, config_path=config_path)
            summary = pipeline.run()

        logger.info("Pipeline executed successfully!")

        # Print executive summary report to console
        counts = summary.get("counts", {})
        print("\n" + "=" * 50)
        print("          RGBT CROWD COUNTING EXECUTION REPORT")
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
