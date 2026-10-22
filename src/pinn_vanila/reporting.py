"""Persistent training logs, metrics, and field visualizations."""

from __future__ import annotations

import csv
import json
import os
import platform
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Cloud runners often mount $HOME read-only. Keeping font and Matplotlib caches
# in /tmp avoids startup warnings and does not pollute a model directory.
_matplotlib_cache = Path(tempfile.gettempdir()) / "pinn-vanila-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_matplotlib_cache))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .case import ProjectConfig
from .evaluation import physics_metrics
from .model import PINN
from .trainer import TrainingResult


LOSS_COLUMNS = (
    "total",
    "continuity",
    "momentum_x",
    "momentum_y",
    "outlet_pressure_gradient",
    "outlet_velocity_gradient",
    "mass_flux",
)


class TrainingReporter:
    """Write live, restart-inspectable artifacts beside the active model."""

    def __init__(
        self,
        output_directory: str | Path,
        config: ProjectConfig,
        *,
        device: torch.device,
        log_every: int = 10,
        field_every: int = 200,
    ) -> None:
        if log_every <= 0 or field_every <= 0:
            raise ValueError("log_every and field_every must be positive")
        self.root = Path(output_directory)
        self.config = config
        self.device = device
        self.log_every = log_every
        self.field_every = field_every
        self.fields_directory = self.root / "fields"
        self.log_path = self.root / "training.log"
        self.metrics_csv_path = self.root / "metrics.csv"
        self.metrics_json_path = self.root / "metrics.json"
        self.plot_path = self.root / "training_losses.png"
        self.history: list[dict[str, float]] = []
        self.diagnostics: list[dict[str, Any]] = []
        self._last_entry: dict[str, float] | None = None
        self._last_field_step: int | None = None
        self._log_file = None
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None
        resolution = (config.raw.get("reporting", {}) or {}).get("field_plot_resolution", [181, 81])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("reporting.field_plot_resolution must contain [s_points, eta_points]")
        self.field_resolution = (int(resolution[0]), int(resolution[1]))
        if min(self.field_resolution) < 2:
            raise ValueError("reporting.field_plot_resolution values must be at least 2")

    def start(self, model: PINN) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.fields_directory.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf-8", buffering=1)
        self._csv_file = self.metrics_csv_path.open("w", newline="", encoding="utf-8", buffering=1)
        fieldnames = [
            "step",
            "stage",
            "elapsed_seconds",
            "lr",
            "reynolds_min",
            "reynolds_max",
            "angle_min_deg",
            "angle_max_deg",
            *LOSS_COLUMNS,
            "weight_continuity",
            "weight_momentum_x",
            "weight_momentum_y",
            "weight_outlet_pressure_gradient",
            "weight_outlet_velocity_gradient",
            "weight_mass_flux",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames, extrasaction="ignore")
        self._csv_writer.writeheader()
        run_metadata = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu": self._gpu_metadata(),
            "config": str(self.config.path),
            "log_every_steps": self.log_every,
            "field_every_steps": self.field_every,
            "field_plot_resolution": list(self.field_resolution),
            "model_metadata": model.metadata(),
        }
        self._write_json(self.root / "run_metadata.json", run_metadata)
        self._write_log(
            "Training started "
            f"device={self.device}; log_every={self.log_every}; field_every={self.field_every}"
        )

    def _gpu_metadata(self) -> dict[str, Any] | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        return {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_gib": round(properties.total_memory / 1024**3, 3),
            "capability": list(torch.cuda.get_device_capability(index)),
        }

    def _write_log(self, line: str) -> None:
        stamped = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {line}"
        print(stamped, flush=True)
        if self._log_file is not None:
            self._log_file.write(stamped + "\n")

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def on_step(self, entry: dict[str, float], model: PINN) -> None:
        step = int(entry["step"])
        self._last_entry = dict(entry)
        should_render = step % self.field_every == 0
        # A field snapshot must always have a matching scalar-loss row, even
        # when callers choose log and field intervals that do not divide evenly.
        should_log = step == 1 or step % self.log_every == 0 or should_render
        if not should_log and not should_render:
            return

        if should_log:
            self.history.append(dict(entry))
            if self._csv_writer is not None:
                self._csv_writer.writerow(entry)
                self._csv_file.flush()
            self._write_log(self._format_entry(entry))
            self._write_metrics_json()

        if should_render:
            diagnostic = self._collect_diagnostic(model, step)
            self.diagnostics.append(diagnostic)
            self._render_fields(model, step)
            self._render_loss_history()
            self._last_field_step = step
            self._write_metrics_json()
            self._write_log(f"Saved field visualization and diagnostics for step={step}")

    @staticmethod
    def _format_entry(entry: dict[str, float]) -> str:
        return (
            f"step={int(entry['step'])} stage={int(entry['stage'])} "
            f"elapsed={entry['elapsed_seconds']:.1f}s lr={entry['lr']:.3e} "
            f"total={entry['total']:.3e} cont={entry['continuity']:.3e} "
            f"mom_x={entry['momentum_x']:.3e} mom_y={entry['momentum_y']:.3e} "
            f"bc_p={entry['outlet_pressure_gradient']:.3e} "
            f"bc_uv={entry['outlet_velocity_gradient']:.3e} flux={entry['mass_flux']:.3e}"
        )

    def _collect_diagnostic(self, model: PINN, step: int) -> dict[str, Any]:
        was_training = model.network.training
        try:
            values = physics_metrics(model, samples=65)
        finally:
            model.network.train(was_training)
        return {"step": step, **values}

    def _render_fields(self, model: PINN, step: int) -> None:
        """Save u, v, relative pressure, and velocity-magnitude fields."""
        geometry = model.case.geometry
        angle = geometry.bend_angle
        reynolds = float(np.sqrt(np.prod(model.network.reynolds_bounds)))
        speed = reynolds * model.case.fluid.kinematic_viscosity / geometry.pipe_width
        s = np.linspace(0.0, geometry.total_length(angle), self.field_resolution[0])
        eta = np.linspace(-1.0, 1.0, self.field_resolution[1])
        s_grid, eta_grid = np.meshgrid(s, eta)
        points = geometry.map(s_grid.ravel(), eta_grid.ravel(), angle)
        was_training = model.network.training
        try:
            prediction = model._dimensionless_prediction(points, reynolds, angle, batch_size=65536)
        finally:
            model.network.train(was_training)
        u = (speed * prediction[:, 0]).reshape(s_grid.shape)
        v = (speed * prediction[:, 1]).reshape(s_grid.shape)
        relative_pressure = (
            model.case.fluid.density * speed**2 * prediction[:, 2]
        ).reshape(s_grid.shape)
        magnitude = np.hypot(u, v)
        xy = geometry.map(s_grid, eta_grid, angle)
        x, y = xy[..., 0], xy[..., 1]
        fields = (
            (u, "$u$ [m/s]", "RdBu_r"),
            (v, "$v$ [m/s]", "RdBu_r"),
            (relative_pressure, "relative $p$ [Pa]", "coolwarm"),
            (magnitude, "$|u|$ [m/s]", "viridis"),
        )
        figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        for axis, (values, title, cmap) in zip(axes.flat, fields, strict=True):
            image = axis.pcolormesh(x, y, values, shading="auto", cmap=cmap)
            figure.colorbar(image, ax=axis, label=title)
            axis.set_title(title)
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            axis.set_aspect("equal")
        stride_y, stride_x = 8, 14
        axes[1, 1].quiver(
            x[::stride_y, ::stride_x],
            y[::stride_y, ::stride_x],
            u[::stride_y, ::stride_x],
            v[::stride_y, ::stride_x],
            color="white",
            scale=None,
            width=0.003,
        )
        figure.suptitle(
            f"PINN fields: step {step}, Re={reynolds:.0f}, angle={angle:.1f}°",
            fontsize=14,
        )
        figure.savefig(self.fields_directory / f"fields_step_{step:07d}.png", dpi=160)
        plt.close(figure)

    def _render_loss_history(self) -> None:
        if not self.history:
            return
        steps = np.array([row["step"] for row in self.history])
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for name in ("total", "continuity", "momentum_x", "momentum_y"):
            values = np.maximum(np.array([row[name] for row in self.history]), 1e-30)
            axes[0].semilogy(steps, values, label=name)
        axes[0].set_title("PDE and total losses")
        axes[0].set_xlabel("optimization step")
        axes[0].set_ylabel("MSE loss")
        axes[0].legend()
        axes[0].grid(True, which="both", alpha=0.3)
        for name in ("outlet_pressure_gradient", "outlet_velocity_gradient", "mass_flux"):
            values = np.maximum(np.array([row[name] for row in self.history]), 1e-30)
            axes[1].semilogy(steps, values, label=name)
        axes[1].set_title("Boundary and mass-flux losses")
        axes[1].set_xlabel("optimization step")
        axes[1].set_ylabel("MSE loss")
        axes[1].legend()
        axes[1].grid(True, which="both", alpha=0.3)
        figure.savefig(self.plot_path, dpi=160)
        plt.close(figure)

    def _write_metrics_json(
        self,
        result: TrainingResult | None = None,
        structural_validation: dict[str, Any] | None = None,
    ) -> None:
        document: dict[str, Any] = {
            "latest": self._last_entry,
            "loss_history": self.history,
            "physics_diagnostics": self.diagnostics,
            "artifacts": {
                "log": self.log_path.name,
                "metrics_csv": self.metrics_csv_path.name,
                "loss_plot": self.plot_path.name,
                "fields_directory": self.fields_directory.name,
            },
        }
        if result is not None:
            document["result"] = asdict(result)
        if structural_validation is not None:
            document["structural_validation"] = structural_validation
        self._write_json(self.metrics_json_path, document)

    def finalize(
        self,
        result: TrainingResult,
        model: PINN,
        structural_validation: dict[str, Any],
    ) -> None:
        final_step = result.steps
        if final_step > 0 and self._last_field_step != final_step:
            diagnostic = self._collect_diagnostic(model, final_step)
            self.diagnostics.append(diagnostic)
            self._render_fields(model, final_step)
            self._last_field_step = final_step
            self._write_log(f"Saved final field visualization and diagnostics for step={final_step}")
        self._render_loss_history()
        self._write_metrics_json(result=result, structural_validation=structural_validation)
        self._write_log(
            f"Training complete steps={result.steps} elapsed={result.elapsed_seconds:.1f}s "
            f"total={result.final_losses.get('total', float('nan')):.3e}"
        )
        if self._csv_file is not None:
            self._csv_file.close()
        if self._log_file is not None:
            self._log_file.close()

    def fail(self, error: BaseException) -> None:
        self._write_log(f"Training failed: {type(error).__name__}: {error}")
        self._write_metrics_json()
        if self._csv_file is not None:
            self._csv_file.close()
        if self._log_file is not None:
            self._log_file.close()
