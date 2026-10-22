"""Standalone training entry point and fast training-session initialization."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .case import ProjectConfig
from .model import PINN
from .trainer import Trainer
from .validation import structural_validation


@dataclass(frozen=True)
class TrainingOptions:
    """Runtime overrides for one training run."""

    config_path: str | Path = "configs/pinn_vanila_config.yaml"
    output: str | Path = "trained/model"
    device: str = "auto"
    seed: int | None = None
    max_steps: int | None = None
    interior_batch: int | None = None
    boundary_batch: int | None = None
    enable_refinement: bool = True
    enable_polishing: bool = True


def resolve_device(value: str) -> str:
    """Resolve the CLI's portable ``auto`` device selection."""
    return "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)


def initialize_training(options: TrainingOptions) -> tuple[ProjectConfig, PINN, Trainer]:
    """Load the configuration and construct a ready-to-run model and trainer.

    This performs no optimizer steps and is useful for notebooks, launchers, and
    callers that need to inspect or modify a session before starting it.
    """
    config = ProjectConfig.from_yaml(options.config_path)
    if options.seed is not None:
        config.raw.setdefault("project", {})["random_seed"] = options.seed
    model = PINN.from_config(config)
    trainer = Trainer(model, config, resolve_device(options.device))
    return config, model, trainer


def run_training(options: TrainingOptions) -> dict[str, Any]:
    """Initialize, train, validate structural constraints, and save a checkpoint."""
    _, model, trainer = initialize_training(options)
    result = trainer.train(
        max_steps=options.max_steps,
        interior_batch=options.interior_batch,
        boundary_batch=options.boundary_batch,
        checkpoint_directory=Path(options.output) / "checkpoints",
        enable_refinement=options.enable_refinement,
        enable_polishing=options.enable_polishing,
    )
    model.validation_status = "trained_structurally_checked_no_cfd_reference"
    model.training_summary["structural_validation"] = structural_validation(model)
    model.save(options.output)
    return {
        "model": str(Path(options.output).resolve()),
        "steps": result.steps,
        "elapsed_seconds": result.elapsed_seconds,
        "final_losses": result.final_losses,
        "validation_status": model.validation_status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinn-vanila-train", description="Initialize and train a PINN-Vanila model"
    )
    parser.add_argument("--config", default="configs/pinn_vanila_config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, help="override project.random_seed")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--interior-batch", type=int)
    parser.add_argument("--boundary-batch", type=int)
    parser.add_argument("--no-rar", action="store_true", help="disable residual adaptive refinement")
    parser.add_argument("--skip-polishing", action="store_true", help="skip final fixed-point L-BFGS")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="initialize and execute one small CPU-friendly training step",
    )
    return parser


def options_from_args(args: argparse.Namespace) -> TrainingOptions:
    quick = bool(args.quick)
    return TrainingOptions(
        config_path=args.config,
        output=args.output,
        device=args.device,
        seed=args.seed,
        max_steps=args.max_steps if args.max_steps is not None else (1 if quick else None),
        interior_batch=(
            args.interior_batch if args.interior_batch is not None else (4 if quick else None)
        ),
        boundary_batch=(
            args.boundary_batch if args.boundary_batch is not None else (4 if quick else None)
        ),
        enable_refinement=not (args.no_rar or quick),
        enable_polishing=not (args.skip_polishing or quick),
    )


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_training(options_from_args(args)), indent=2))


if __name__ == "__main__":
    main()
