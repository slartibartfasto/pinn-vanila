"""High-level persistence and inference API."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .case import BendGeometry, FlowCase, FluidProperties, ProjectConfig
from .errors import ExtrapolationError, GeometryError
from .network import GeometryConditionedPINN, NetworkConfig


@dataclass(frozen=True)
class PredictionResult:
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    p_pa: np.ndarray

    @property
    def velocity_magnitude_m_s(self) -> np.ndarray:
        return np.hypot(self.u_m_s, self.v_m_s)

    def __len__(self) -> int:
        return len(self.u_m_s)


class PINN:
    """A trained or trainable PINN with physical-domain metadata."""

    METADATA_VERSION = 1

    def __init__(
        self,
        network: GeometryConditionedPINN,
        case: FlowCase,
        velocity_bounds: tuple[float, float],
        validation_status: str = "unvalidated",
        training_summary: dict[str, Any] | None = None,
    ):
        self.network = network
        self.case = case
        self.velocity_bounds = tuple(float(v) for v in velocity_bounds)
        self.validation_status = validation_status
        self.training_summary = training_summary or {}

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "PINN":
        raw = config.raw
        seed = int((raw.get("project", {}) or {}).get("random_seed", 42))
        torch.manual_seed(seed)
        network_config = NetworkConfig.from_spec(
            raw.get("network", {}), seed=seed
        )
        network = GeometryConditionedPINN(config.case.geometry, config.reynolds_bounds, network_config)
        return cls(network, config.case, config.velocity_bounds)

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.network.parameters()).dtype

    def to(self, device: str | torch.device) -> "PINN":
        self.network.to(device=device)
        return self

    def _check_geometry(self, geometry: BendGeometry) -> None:
        trained = self.case.geometry
        fields = ("pipe_width", "bend_radius", "inlet_length", "outlet_length")
        mismatches = [
            name
            for name in fields
            if not np.isclose(getattr(geometry, name), getattr(trained, name), rtol=1e-12, atol=1e-12)
        ]
        if mismatches:
            raise GeometryError(
                "Geometry differs from the trained family in: " + ", ".join(mismatches) + ". Retraining is required."
            )

    def _check_domain(
        self, geometry: BendGeometry, inlet_vmax_m_s: float, extrapolation_override: bool
    ) -> float:
        self._check_geometry(geometry)
        speed = float(inlet_vmax_m_s)
        low_v, high_v = self.velocity_bounds
        low_a, high_a = self.case.geometry.angle_bounds
        outside = not (low_v <= speed <= high_v and low_a <= geometry.bend_angle <= high_a)
        if outside and not extrapolation_override:
            raise ExtrapolationError(
                f"Request (Vmax={speed:g} m/s, angle={geometry.bend_angle:g} deg) is outside "
                f"the trained domain Vmax={self.velocity_bounds}, angle={self.case.geometry.angle_bounds}; "
                "set extrapolation_override=True to proceed explicitly"
            )
        if outside:
            warnings.warn("Inference is outside the trained parameter domain", RuntimeWarning, stacklevel=2)
        return self.case.reynolds(speed)

    def _chunks(self, values: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]

    def _dimensionless_prediction(
        self,
        points: np.ndarray,
        reynolds: float,
        angle: float,
        batch_size: int,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        self.network.eval()
        with torch.no_grad():
            for chunk in self._chunks(points, batch_size):
                xy = torch.as_tensor(chunk, dtype=self.dtype, device=self.device)
                re = torch.full((len(chunk),), reynolds, dtype=self.dtype, device=self.device)
                alpha = torch.full((len(chunk),), angle, dtype=self.dtype, device=self.device)
                outputs.append(self.network(xy, re, alpha).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict(
        self,
        geometry: BendGeometry,
        points_xy: np.ndarray,
        inlet_vmax_m_s: float,
        inlet_pressure_pa: float,
        *,
        extrapolation_override: bool = False,
        batch_size: int = 65536,
        gauge_quadrature_points: int = 64,
    ) -> PredictionResult:
        """Predict physical ``u``, ``v`` and pressure at valid channel points."""
        pressure = float(inlet_pressure_pa)
        if not np.isfinite(pressure):
            raise ValueError("inlet_pressure_pa must be finite")
        if batch_size <= 0 or gauge_quadrature_points < 2:
            raise ValueError("batch_size must be positive and gauge_quadrature_points must be at least 2")
        reynolds = self._check_domain(geometry, inlet_vmax_m_s, extrapolation_override)
        points = np.asarray(points_xy, dtype=float)
        _, _, inside = geometry.inverse(points)
        if not np.all(inside):
            indices = np.flatnonzero(~inside)
            preview = ", ".join(str(int(i)) for i in indices[:10])
            raise GeometryError(f"Query points outside the channel at indices: {preview}")

        dimensionless = self._dimensionless_prediction(
            points, reynolds, geometry.bend_angle, batch_size
        )
        eta, weights = np.polynomial.legendre.leggauss(gauge_quadrature_points)
        inlet_points = geometry.map(np.zeros_like(eta), eta)
        inlet_pi = self._dimensionless_prediction(
            inlet_points, reynolds, geometry.bend_angle, batch_size
        )[:, 2]
        mean_inlet_pi = 0.5 * np.dot(weights, inlet_pi)
        speed = float(inlet_vmax_m_s)
        physical_pressure = pressure + self.case.fluid.density * speed**2 * (
            dimensionless[:, 2] - mean_inlet_pi
        )
        return PredictionResult(
            u_m_s=speed * dimensionless[:, 0],
            v_m_s=speed * dimensionless[:, 1],
            p_pa=physical_pressure,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "metadata_version": self.METADATA_VERSION,
            "package": "pinn-vanila",
            "geometry": self.case.geometry.to_dict(),
            "fluid": self.case.fluid.to_dict(),
            "velocity_bounds_m_s": list(self.velocity_bounds),
            "reynolds_bounds": list(self.network.reynolds_bounds),
            "network": self.network.config.to_dict(),
            "validation_status": self.validation_status,
            "training_summary": self.training_summary,
            "pressure_gauge": "mean_inlet_static_pressure",
        }

    def save(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), destination / "model.pt")
        (destination / "metadata.json").write_text(
            json.dumps(self.metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, device: str | torch.device = "cpu") -> "PINN":
        source = Path(directory)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("metadata_version") != cls.METADATA_VERSION:
            raise ValueError("Unsupported model metadata version")
        geometry = BendGeometry.from_metadata(metadata["geometry"])
        fluid = FluidProperties(**metadata["fluid"])
        network_config = NetworkConfig.from_dict(metadata["network"])
        network = GeometryConditionedPINN(
            geometry, tuple(metadata["reynolds_bounds"]), network_config
        )
        state = torch.load(source / "model.pt", map_location=device, weights_only=True)
        network.load_state_dict(state)
        network.to(device=device)
        return cls(
            network,
            FlowCase(geometry, fluid),
            tuple(metadata["velocity_bounds_m_s"]),
            validation_status=metadata.get("validation_status", "unknown"),
            training_summary=metadata.get("training_summary", {}),
        )
