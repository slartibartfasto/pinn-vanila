"""Validated input configuration for PINN-Vanila."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .errors import ConfigurationError, GeometryError


def _read_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read YAML file {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"YAML root must be a mapping: {source}")
    return data


def _positive(data: Mapping[str, Any], key: str) -> float:
    try:
        value = float(data[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be a number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{key} must be finite and positive")
    return value


@dataclass(frozen=True)
class FluidProperties:
    density: float
    kinematic_viscosity: float
    dynamic_viscosity: float
    id: str = "fluid"
    name: str = "Fluid"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FluidProperties":
        fluid = cls(
            density=_positive(data, "density"),
            kinematic_viscosity=_positive(data, "kinematic_viscosity"),
            dynamic_viscosity=_positive(data, "dynamic_viscosity"),
            id=str(data.get("id", "fluid")),
            name=str(data.get("name", "Fluid")),
        )
        validation = data.get("validation", {}) or {}
        if validation.get("require_mu_equals_rho_nu", False):
            tolerance = float(validation.get("relative_tolerance", 1e-10))
            expected = fluid.density * fluid.kinematic_viscosity
            relative_error = abs(fluid.dynamic_viscosity - expected) / expected
            if relative_error > tolerance:
                raise ConfigurationError(
                    "dynamic_viscosity must equal density * kinematic_viscosity "
                    f"within relative tolerance {tolerance:g}"
                )
        return fluid

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FluidProperties":
        return cls.from_dict(_read_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BendGeometry:
    pipe_width: float
    bend_radius: float
    inlet_length: float
    outlet_length: float
    bend_angle: float
    angle_bounds: tuple[float, float] = (15.0, 90.0)
    id: str = "smooth_planar_bend"
    min_inner_radius: float = 0.001

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BendGeometry":
        bounds = (data.get("parameter_bounds", {}) or {}).get(
            "bend_angle_deg", [data.get("bend_angle", 90.0)] * 2
        )
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ConfigurationError("parameter_bounds.bend_angle_deg must contain two values")
        geometry = cls(
            pipe_width=_positive(data, "pipe_width"),
            bend_radius=_positive(data, "bend_radius"),
            inlet_length=_positive(data, "inlet_length"),
            outlet_length=_positive(data, "outlet_length"),
            bend_angle=float(data.get("bend_angle", bounds[1])),
            angle_bounds=(float(bounds[0]), float(bounds[1])),
            id=str(data.get("id", "smooth_planar_bend")),
            min_inner_radius=float(
                (data.get("validation", {}) or {}).get("min_inner_radius_m", 0.001)
            ),
        )
        geometry.validate()
        return geometry

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BendGeometry":
        return cls.from_dict(_read_yaml(path))

    def validate(self) -> None:
        low, high = self.angle_bounds
        if not (0 < low <= self.bend_angle <= high < 180):
            raise GeometryError("bend angle and bounds must satisfy 0 < min <= angle <= max < 180")
        if self.inner_radius < self.min_inner_radius:
            raise GeometryError(
                f"inner radius {self.inner_radius:g} m is below minimum "
                f"{self.min_inner_radius:g} m"
            )
        if self.minimum_jacobian_factor <= 0:
            raise GeometryError("centerline mapping has a non-positive Jacobian")

    @property
    def inner_radius(self) -> float:
        return self.bend_radius - 0.5 * self.pipe_width

    @property
    def minimum_jacobian_factor(self) -> float:
        return 1.0 - self.pipe_width / (2.0 * self.bend_radius)

    def total_length(self, angle_deg: float | None = None) -> float:
        angle = self.bend_angle if angle_deg is None else float(angle_deg)
        return self.inlet_length + self.bend_radius * np.deg2rad(angle) + self.outlet_length

    def map(
        self,
        s: np.ndarray | float,
        eta: np.ndarray | float,
        angle_deg: float | None = None,
    ) -> np.ndarray:
        """Map centerline coordinates ``(s, eta)`` to physical ``(x, y)``."""
        angle = self.bend_angle if angle_deg is None else float(angle_deg)
        if not self.angle_bounds[0] <= angle <= self.angle_bounds[1]:
            raise GeometryError(f"angle {angle:g} deg is outside geometry bounds {self.angle_bounds}")
        s_arr, eta_arr = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(eta, dtype=float))
        total = self.total_length(angle)
        if np.any((s_arr < 0) | (s_arr > total)):
            raise GeometryError("s is outside the channel centerline")
        if np.any(np.abs(eta_arr) > 1):
            raise GeometryError("eta must be in [-1, 1]")

        alpha = np.deg2rad(angle)
        arc_end = self.inlet_length + self.bend_radius * alpha
        theta = np.clip((s_arr - self.inlet_length) / self.bend_radius, 0.0, alpha)
        arc_x = self.inlet_length + self.bend_radius * np.sin(theta)
        arc_y = self.bend_radius * (1.0 - np.cos(theta))
        tangent_x, tangent_y = np.cos(theta), np.sin(theta)

        x = arc_x - 0.5 * self.pipe_width * eta_arr * tangent_y
        y = arc_y + 0.5 * self.pipe_width * eta_arr * tangent_x

        inlet = s_arr <= self.inlet_length
        x = np.where(inlet, s_arr, x)
        y = np.where(inlet, 0.5 * self.pipe_width * eta_arr, y)

        outlet = s_arr >= arc_end
        tau = s_arr - arc_end
        end_x = self.inlet_length + self.bend_radius * np.sin(alpha)
        end_y = self.bend_radius * (1.0 - np.cos(alpha))
        x_out = end_x + tau * np.cos(alpha) - 0.5 * self.pipe_width * eta_arr * np.sin(alpha)
        y_out = end_y + tau * np.sin(alpha) + 0.5 * self.pipe_width * eta_arr * np.cos(alpha)
        x = np.where(outlet, x_out, x)
        y = np.where(outlet, y_out, y)
        return np.stack((x, y), axis=-1)

    def inverse(
        self,
        points_xy: np.ndarray,
        angle_deg: float | None = None,
        tolerance: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``s``, ``eta`` and an inside mask for physical points."""
        points = np.asarray(points_xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise GeometryError("points_xy must be a finite array with shape [N, 2]")
        angle = self.bend_angle if angle_deg is None else float(angle_deg)
        alpha = np.deg2rad(angle)
        h = 0.5 * self.pipe_width
        x, y = points[:, 0], points[:, 1]

        s_in = x
        eta_in = y / h
        valid_in = (s_in >= -tolerance) & (s_in <= self.inlet_length + tolerance)

        dx, dy = x - self.inlet_length, y - self.bend_radius
        theta = np.arctan2(dx, -dy)
        radius = np.hypot(dx, dy)
        s_arc = self.inlet_length + self.bend_radius * theta
        eta_arc = (self.bend_radius - radius) / h
        valid_arc = (theta >= -tolerance) & (theta <= alpha + tolerance)

        arc_end = self.inlet_length + self.bend_radius * alpha
        end = np.array(
            [
                self.inlet_length + self.bend_radius * np.sin(alpha),
                self.bend_radius * (1.0 - np.cos(alpha)),
            ]
        )
        tangent = np.array([np.cos(alpha), np.sin(alpha)])
        normal = np.array([-np.sin(alpha), np.cos(alpha)])
        relative = points - end
        tau = relative @ tangent
        s_out = arc_end + tau
        eta_out = (relative @ normal) / h
        valid_out = (tau >= -tolerance) & (tau <= self.outlet_length + tolerance)

        candidate_s = np.stack((s_in, s_arc, s_out), axis=1)
        candidate_eta = np.stack((eta_in, eta_arc, eta_out), axis=1)
        valid_segment = np.stack((valid_in, valid_arc, valid_out), axis=1)
        score = np.where(valid_segment, np.abs(candidate_eta), np.inf)
        index = np.argmin(score, axis=1)
        rows = np.arange(len(points))
        s_result = candidate_s[rows, index]
        eta_result = candidate_eta[rows, index]
        has_segment = np.any(valid_segment, axis=1)
        inside = has_segment & (np.abs(eta_result) <= 1.0 + tolerance)
        return s_result, eta_result, inside

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["angle_bounds"] = list(self.angle_bounds)
        return result

    @classmethod
    def from_metadata(cls, data: Mapping[str, Any]) -> "BendGeometry":
        values = dict(data)
        values["angle_bounds"] = tuple(values["angle_bounds"])
        return cls(**values)


@dataclass(frozen=True)
class FlowCase:
    geometry: BendGeometry
    fluid: FluidProperties

    @classmethod
    def from_yaml(cls, geometry_path: str | Path, fluid_path: str | Path) -> "FlowCase":
        return cls(BendGeometry.from_yaml(geometry_path), FluidProperties.from_yaml(fluid_path))

    def reynolds(self, inlet_vmax_m_s: float) -> float:
        speed = float(inlet_vmax_m_s)
        if not np.isfinite(speed) or speed <= 0:
            raise ConfigurationError("inlet_vmax_m_s must be finite and positive")
        return speed * self.geometry.pipe_width / self.fluid.kinematic_viscosity


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    raw: dict[str, Any]
    case: FlowCase

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        config_path = Path(path).resolve()
        raw = _read_yaml(config_path)
        project = raw.get("project", {}) or {}
        try:
            geometry_path = config_path.parent / project["geometry_file"]
            fluid_path = config_path.parent / project["fluid_file"]
        except KeyError as exc:
            raise ConfigurationError(f"Missing project path: {exc.args[0]}") from exc
        case = FlowCase.from_yaml(geometry_path, fluid_path)
        mode = project.get("training_mode", "physics_only")
        reference_weight = float((raw.get("loss", {}) or {}).get("reference_data", 0.0))
        if mode == "physics_only" and reference_weight != 0:
            raise ConfigurationError("loss.reference_data must be zero in physics_only mode")
        config = cls(config_path, raw, case)
        config.validate_supported_contract()
        return config

    def validate_supported_contract(self) -> None:
        """Fail early when a config requests behavior this solver does not implement."""
        project = self.raw.get("project", {}) or {}
        physical = self.raw.get("physical_problem", {}) or {}
        boundary = self.raw.get("boundary_conditions", {}) or {}
        network = self.raw.get("network", {}) or {}
        required = {
            "project.training_mode": (project.get("training_mode"), "physics_only"),
            "physical_problem.equations": (
                physical.get("equations"),
                "steady_incompressible_navier_stokes_2d",
            ),
            "physical_problem.reynolds_definition": (
                physical.get("reynolds_definition"),
                "vmax_times_width_div_nu",
            ),
            "network.architecture": (
                network.get("architecture"),
                "geometry_conditioned_residual_mlp",
            ),
            "network.formulation": (network.get("formulation"), "velocity_pressure"),
            "network.differentiation": (
                network.get("differentiation"),
                "autograd_physical_coordinates",
            ),
            "network.inlet_boundary_enforcement": (
                network.get("inlet_boundary_enforcement"),
                "hard",
            ),
            "network.wall_boundary_enforcement": (
                network.get("wall_boundary_enforcement"),
                "hard",
            ),
        }
        for path, (actual, expected) in required.items():
            if actual != expected:
                raise ConfigurationError(f"{path} must be {expected!r}; got {actual!r}")
        inlet = boundary.get("inlet", {}) or {}
        outlet = boundary.get("outlet", {}) or {}
        if inlet.get("velocity") != "fully_developed_parabolic":
            raise ConfigurationError("Only a fully_developed_parabolic inlet is supported")
        if ((outlet.get("pressure", {}) or {}).get("type")) != "zero_normal_gradient":
            raise ConfigurationError("Configured outlet pressure must use zero_normal_gradient")
        if ((outlet.get("velocity", {}) or {}).get("type")) != "zero_normal_gradient":
            raise ConfigurationError("Configured outlet velocity must use zero_normal_gradient")
        low_v, high_v = self.velocity_bounds
        if not (0 < low_v < high_v):
            raise ConfigurationError("inlet_velocity_bounds_m_s must be positive and increasing")

    @property
    def velocity_bounds(self) -> tuple[float, float]:
        values = self.raw["physical_problem"]["inlet_velocity_bounds_m_s"]
        return float(values[0]), float(values[1])

    @property
    def reynolds_bounds(self) -> tuple[float, float]:
        return tuple(self.case.reynolds(value) for value in self.velocity_bounds)
