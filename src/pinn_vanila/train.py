"""Standalone training entry point and fast training-session initialization."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

MODULE_PATH = Path(__file__).resolve()
SOURCE_ROOT = MODULE_PATH.parents[1]
PROJECT_ROOT = MODULE_PATH.parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pinn_vanila_config.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "trained" / "rans-komega" / "seed-42"

# VS Code's "Run Python File" executes this source file directly, rather than
# through ``python -m``. Add the src root in that case so package imports work
# in a fresh checkout as well as in an editable installation.
if __package__ in {None, ""}:  # pragma: no cover - exercised by direct CLI launch
    sys.path.insert(0, str(SOURCE_ROOT))
    from pinn_vanila.case import ProjectConfig
    from pinn_vanila.model import PINN
    from pinn_vanila.reporting import TrainingReporter
    from pinn_vanila.trainer import Trainer
    from pinn_vanila.validation import structural_validation
else:
    from .case import ProjectConfig
    from .model import PINN
    from .reporting import TrainingReporter
    from .trainer import Trainer
    from .validation import structural_validation


@dataclass(frozen=True)
class TrainingOptions:
    """Runtime overrides for one training run."""

    config_path: str | Path = DEFAULT_CONFIG
    output: str | Path = DEFAULT_OUTPUT
    device: str = "cuda"
    seed: int | None = None
    max_steps: int | None = None
    interior_batch: int | None = None
    boundary_batch: int | None = None
    enable_refinement: bool = True
    enable_polishing: bool = True
    log_every: int | None = None
    field_every: int | None = None


def resolve_device(value: str) -> str:
    """Select an accelerator and reject a silent CPU fallback."""
    requested = "cuda" if value == "auto" else value
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for training but is unavailable. "
            "Run on a CUDA-enabled environment, or explicitly pass --device cpu only for debugging."
        )
    return requested


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
    config, model, trainer = initialize_training(options)
    reporting_config = config.raw.get("reporting", {}) or {}
    log_every = int(
        options.log_every
        if options.log_every is not None
        else reporting_config.get("console_and_metrics_every_steps", 10)
    )
    field_every = int(
        options.field_every
        if options.field_every is not None
        else reporting_config.get("field_images_and_diagnostics_every_steps", 200)
    )
    reporter = TrainingReporter(
        options.output,
        config,
        device=trainer.device,
        log_every=log_every,
        field_every=field_every,
    )
    reporter.start(model)
    try:
        result = trainer.train(
            max_steps=options.max_steps,
            interior_batch=options.interior_batch,
            boundary_batch=options.boundary_batch,
            checkpoint_directory=Path(options.output) / "checkpoints",
            enable_refinement=options.enable_refinement,
            enable_polishing=options.enable_polishing,
            log_every=log_every,
            progress_callbacks=[reporter.on_step],
        )
        model.validation_status = "trained_structurally_checked_no_cfd_reference"
        structural = structural_validation(model)
        model.training_summary["structural_validation"] = structural
        model.training_summary["reporting"] = {
            "log_every_steps": log_every,
            "field_every_steps": field_every,
        }
        model.save(options.output)
        reporter.finalize(result, model, structural)
    except BaseException as error:
        reporter.fail(error)
        raise
    return {
        "model": str(Path(options.output).resolve()),
        "steps": result.steps,
        "elapsed_seconds": result.elapsed_seconds,
        "final_losses": result.final_losses,
        "validation_status": model.validation_status,
        "artifacts": {
            "log": str(reporter.log_path),
            "metrics": str(reporter.metrics_json_path),
            "loss_plot": str(reporter.plot_path),
            "fields": str(reporter.fields_directory),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinn-vanila-train", description="Initialize and train a PINN-Vanila model"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"checkpoint directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device; CPU requires an explicit --device cpu")
    parser.add_argument("--seed", type=int, help="override project.random_seed")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--interior-batch", type=int)
    parser.add_argument("--boundary-batch", type=int)
    parser.add_argument("--log-every", type=int, help="override reporting.console_and_metrics_every_steps")
    parser.add_argument("--field-every", type=int, help="override reporting.field_images_and_diagnostics_every_steps")
    parser.add_argument("--no-rar", action="store_true", help="disable residual adaptive refinement")
    parser.add_argument("--skip-polishing", action="store_true", help="skip final fixed-point L-BFGS")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="initialize and execute one small training step",
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
        log_every=args.log_every,
        field_every=args.field_every,
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        print(json.dumps(run_training(options_from_args(args)), indent=2))
    except RuntimeError as error:
        print(f"Training error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
