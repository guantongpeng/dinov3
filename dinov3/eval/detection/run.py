# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Entry point for detection training/evaluation with DINOv3.

Usage:
    # Training
    python -m dinov3.eval.detection.run \
        output_dir=./output/detection \
        model=dinov3_vit7b16 \
        datasets.train_img_dir=/path/to/train2017 \
        datasets.train_ann_file=/path/to/instances_train2017.json \
        datasets.val_img_dir=/path/to/val2017 \
        datasets.val_ann_file=/path/to/instances_val2017.json \
        epochs=60 batch_size=2

    # With config file
    python -m dinov3.eval.detection.run \
        config=/path/to/config.yaml \
        output_dir=./output/detection

    # Resume from checkpoint
    python -m dinov3.eval.detection.run \
        config=/path/to/config.yaml \
        load_from=/path/to/checkpoint_best.pth \
        output_dir=./output/detection
"""
import logging
import os
import sys
from typing import Any

from omegaconf import OmegaConf

import dinov3.distributed as distributed
from dinov3.eval.detection.config import DetectionTrainConfig
from dinov3.eval.detection.train import train_detection
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser
from dinov3.run.init import job_context

logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-detection.csv"


def benchmark_launcher(eval_args: dict[str, Any]) -> dict[str, Any]:
    """Main launcher for detection training."""
    if "config" in eval_args:
        base_config_path = eval_args.pop("config")
        output_dir = eval_args["output_dir"]
        base_config = OmegaConf.load(base_config_path)
        structured_config = OmegaConf.structured(DetectionTrainConfig)
        config: DetectionTrainConfig = OmegaConf.to_object(
            OmegaConf.merge(
                structured_config,
                base_config,
                OmegaConf.create(eval_args),
            )
        )
    else:
        config, output_dir = args_dict_to_dataclass(
            eval_args=eval_args, config_dataclass=DetectionTrainConfig, save_config=False
        )

    # Save config
    config_path = os.path.join(output_dir, "detection_config.yaml")
    if distributed.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(config=config, f=config_path)
    logger.info(f"Detection Config:\n{OmegaConf.to_yaml(config)}")

    results = train_detection(config)
    return results


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    with job_context(output_dir=eval_args["output_dir"]):
        results = benchmark_launcher(eval_args)
    logger.info(f"Training results: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
