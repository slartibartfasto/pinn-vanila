"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .case import BendGeometry, FlowCase, ProjectConfig
from .evaluation import evaluate_model, load_reference
from .export import export_prediction
from .model import PINN
from .train import TrainingOptions, run_training
from .validation import structural_validation


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _load_points(path: str | Path) -> np.ndarray:
    source = Path(path)
    delimiter = "," if source.suffix.lower() == ".csv" else None
    try:
        points = np.loadtxt(source, delimiter=delimiter, ndmin=2)
    except ValueError:
        points = np.loadtxt(source, delimiter=delimiter, skiprows=1, ndmin=2)
    if points.shape[1] < 2:
        raise ValueError("Points file must contain at least x and y columns")
    return points[:, :2]


def validate_config(args: argparse.Namespace) -> int:
    config = ProjectConfig.from_yaml(args.config)
    model = PINN.from_config(config).to(_device(args.device))
    report = {
        "configuration": str(config.path),
        "velocity_bounds_m_s": config.velocity_bounds,
        "reynolds_bounds": config.reynolds_bounds,
        "angle_bounds_deg": config.case.geometry.angle_bounds,
        "structural_validation": structural_validation(model),
    }
    print(json.dumps(report, indent=2))
    return 0


def train(args: argparse.Namespace) -> int:
    report = run_training(
        TrainingOptions(
            config_path=args.config,
            output=args.output,
            device=args.device,
            seed=args.seed,
            max_steps=args.max_steps,
            interior_batch=args.interior_batch,
            boundary_batch=args.boundary_batch,
            enable_refinement=not args.no_rar,
            enable_polishing=not args.skip_polishing,
            log_every=args.log_every,
            field_every=args.field_every,
        )
    )
    print(json.dumps(report, indent=2))
    return 0


def predict(args: argparse.Namespace) -> int:
    model = PINN.load(args.model, device=_device(args.device))
    geometry = BendGeometry.from_yaml(args.geometry)
    # Validate the supplied fluid file against model metadata as part of the API contract.
    case = FlowCase.from_yaml(args.geometry, args.fluid)
    if case.fluid != model.case.fluid:
        raise ValueError("Fluid properties differ from the trained model metadata")
    points = _load_points(args.points)
    result = model.predict(
        geometry,
        points,
        args.vmax,
        args.pressure,
        extrapolation_override=args.extrapolation_override,
        batch_size=args.batch_size,
    )
    export_prediction(args.output, points, result)
    print(
        json.dumps(
            {
                "points": len(result),
                "output": str(Path(args.output).resolve()),
                "validation_status": model.validation_status,
            },
            indent=2,
        )
    )
    return 0


def evaluate(args: argparse.Namespace) -> int:
    model = PINN.load(args.model, device=_device(args.device))
    geometry = BendGeometry.from_yaml(args.geometry)
    reference = load_reference(args.reference)
    report = evaluate_model(model, geometry, reference, args.vmax, args.pressure)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pinn-vanila")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate files and hard constraints")
    validate.add_argument("--config", default="configs/pinn_vanila_config.yaml")
    validate.add_argument("--device", default="auto")
    validate.set_defaults(handler=validate_config)

    training = subparsers.add_parser("train", help="run physics-only curriculum training")
    training.add_argument("--config", default="configs/pinn_vanila_config.yaml")
    training.add_argument("--output", required=True)
    training.add_argument("--device", default="cuda", help="CUDA device; CPU requires an explicit --device cpu")
    training.add_argument("--max-steps", type=int)
    training.add_argument("--seed", type=int, help="override project.random_seed for an independent run")
    training.add_argument("--interior-batch", type=int)
    training.add_argument("--boundary-batch", type=int)
    training.add_argument("--log-every", type=int)
    training.add_argument("--field-every", type=int)
    training.add_argument("--no-rar", action="store_true", help="disable residual adaptive refinement")
    training.add_argument("--skip-polishing", action="store_true", help="skip final fixed-point L-BFGS")
    training.set_defaults(handler=train)

    inference = subparsers.add_parser("predict", help="run a saved model")
    inference.add_argument("--model", required=True)
    inference.add_argument("--geometry", required=True)
    inference.add_argument("--fluid", required=True)
    inference.add_argument("--points", required=True, help="CSV or whitespace file with x,y columns")
    inference.add_argument("--vmax", required=True, type=float)
    inference.add_argument("--pressure", required=True, type=float)
    inference.add_argument("--output", required=True)
    inference.add_argument("--batch-size", type=int, default=65536)
    inference.add_argument("--device", default="auto")
    inference.add_argument("--extrapolation-override", action="store_true")
    inference.set_defaults(handler=predict)

    evaluation = subparsers.add_parser("evaluate", help="compare a model with CFD reference data")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--geometry", required=True)
    evaluation.add_argument("--reference", required=True, help="CSV, HDF5, VTU, or VTK reference")
    evaluation.add_argument("--vmax", required=True, type=float)
    evaluation.add_argument("--pressure", required=True, type=float)
    evaluation.add_argument("--output", help="optional JSON report path")
    evaluation.add_argument("--device", default="auto")
    evaluation.set_defaults(handler=evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
