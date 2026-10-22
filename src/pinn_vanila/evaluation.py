"""Reference comparison and physics diagnostics for trained checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .case import BendGeometry
from .model import PINN, PredictionResult
from .physics import navier_stokes_residuals, outlet_normal_gradients
from .sampling import PhysicsSampler


@dataclass(frozen=True)
class ReferenceData:
    points_xy_m: np.ndarray
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    p_pa: np.ndarray


def load_reference(path: str | Path) -> ReferenceData:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        table = np.genfromtxt(source, delimiter=",", names=True)
        names = table.dtype.names or ()
        required = ("x_m", "y_m", "u_m_s", "v_m_s", "p_pa")
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError("Reference CSV is missing columns: " + ", ".join(missing))
        return ReferenceData(
            np.column_stack((table["x_m"], table["y_m"])),
            np.atleast_1d(table["u_m_s"]),
            np.atleast_1d(table["v_m_s"]),
            np.atleast_1d(table["p_pa"]),
        )
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("HDF5 reference loading requires 'h5py'") from exc
        with h5py.File(source, "r") as handle:
            return ReferenceData(
                np.asarray(handle["points_xy_m"]),
                np.asarray(handle["u_m_s"]),
                np.asarray(handle["v_m_s"]),
                np.asarray(handle["p_pa"]),
            )
    if suffix in {".vtu", ".vtk"}:
        try:
            import meshio
        except ImportError as exc:
            raise RuntimeError("VTK reference loading requires 'meshio'") from exc
        mesh = meshio.read(source)
        velocity = np.asarray(mesh.point_data["velocity_m_s"])
        return ReferenceData(
            np.asarray(mesh.points[:, :2]),
            velocity[:, 0],
            velocity[:, 1],
            np.asarray(mesh.point_data["pressure_pa"]),
        )
    raise ValueError("Reference format must be CSV, HDF5, VTU, or VTK")


def reference_metrics(
    geometry: BendGeometry,
    prediction: PredictionResult,
    reference: ReferenceData,
) -> dict[str, float | None]:
    predicted_velocity = np.column_stack((prediction.u_m_s, prediction.v_m_s))
    reference_velocity = np.column_stack((reference.u_m_s, reference.v_m_s))
    velocity_denominator = np.linalg.norm(reference_velocity)
    relative_velocity = np.linalg.norm(predicted_velocity - reference_velocity) / max(
        velocity_denominator, np.finfo(float).eps
    )

    # Pressure is compared only after optimal constant gauge alignment.
    gauge_shift = float(np.mean(reference.p_pa - prediction.p_pa))
    aligned_pressure = prediction.p_pa + gauge_shift
    pressure_fluctuation = reference.p_pa - np.mean(reference.p_pa)
    pressure_scale = max(np.linalg.norm(pressure_fluctuation), np.finfo(float).eps)
    normalized_pressure = np.linalg.norm(aligned_pressure - reference.p_pa) / pressure_scale

    s, _, inside = geometry.inverse(reference.points_xy_m)
    if not np.all(inside):
        raise ValueError("Reference contains points outside the configured geometry")
    section_tolerance = max(1e-8, geometry.total_length() * 1e-6)
    inlet = s <= section_tolerance
    outlet = s >= geometry.total_length() - section_tolerance
    pressure_drop_error: float | None = None
    if np.any(inlet) and np.any(outlet):
        reference_drop = float(np.mean(reference.p_pa[inlet]) - np.mean(reference.p_pa[outlet]))
        predicted_drop = float(np.mean(aligned_pressure[inlet]) - np.mean(aligned_pressure[outlet]))
        pressure_drop_error = abs(predicted_drop - reference_drop) / max(
            abs(reference_drop), np.finfo(float).eps
        )
    return {
        "relative_l2_velocity": float(relative_velocity),
        "normalized_pressure_error_after_gauge_alignment": float(normalized_pressure),
        "pressure_gauge_shift_pa": gauge_shift,
        "relative_pressure_drop_error": pressure_drop_error,
    }


def physics_metrics(
    model: PINN, geometry: BendGeometry | None = None, samples: int = 129
) -> dict[str, float]:
    """Compute inexpensive diagnostics independent of labeled reference data."""
    geometry = geometry or model.case.geometry
    angle = geometry.bend_angle
    reynolds = float(np.sqrt(np.prod(model.network.reynolds_bounds)))
    speed = reynolds * model.case.fluid.kinematic_viscosity / geometry.pipe_width
    eta, weights = np.polynomial.legendre.leggauss(samples)
    total = geometry.total_length()
    section_s = np.linspace(0.0, total, 7)
    fluxes = []
    for s_value in section_s:
        points = geometry.map(np.full(samples, s_value), eta)
        output = model._dimensionless_prediction(points, reynolds, angle, samples)
        theta = np.clip((s_value - geometry.inlet_length) / geometry.bend_radius, 0, np.deg2rad(angle))
        tangent = np.array([np.cos(theta), np.sin(theta)])
        axial = output[:, :2] @ tangent
        fluxes.append(0.5 * np.dot(weights, axial))
    inlet_flux = fluxes[0]
    mass_imbalance = max(abs(value / inlet_flux - 1.0) for value in fluxes[1:])

    wall_s = np.linspace(0.0, total, samples)
    wall_points = np.concatenate(
        (geometry.map(wall_s, np.ones(samples)), geometry.map(wall_s, -np.ones(samples)))
    )
    wall = model._dimensionless_prediction(wall_points, reynolds, angle, samples)
    wall_slip = float(np.max(np.linalg.norm(wall[:, :2], axis=1)))

    outlet_points = geometry.map(np.full(samples, total), eta)
    outlet = model._dimensionless_prediction(outlet_points, reynolds, angle, samples)
    tangent = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    outlet_axial = outlet[:, :2] @ tangent
    backflow_fraction = float(np.mean(outlet_axial < 0.0))

    sampler = PhysicsSampler(
        geometry, device=model.device, dtype=model.dtype, seed=model.network.config.seed + 1000
    )
    sample_batch = sampler.interior(256, (reynolds, reynolds), (angle, angle))
    xy = sample_batch.xy.detach().requires_grad_(True)
    residual = navier_stokes_residuals(
        model.network, xy, sample_batch.reynolds, sample_batch.angle_deg
    )
    outlet_batch = sampler.outlet(samples, (reynolds, reynolds), (angle, angle))
    outlet_xy = outlet_batch.xy.detach().requires_grad_(True)
    grad_u, grad_v, grad_p = outlet_normal_gradients(
        model.network, outlet_xy, outlet_batch.reynolds, outlet_batch.angle_deg
    )
    return {
        "mass_imbalance": float(mass_imbalance),
        "relative_wall_slip": wall_slip,
        "outlet_backflow_fraction": backflow_fraction,
        "pde_continuity_rms": float(residual.continuity.square().mean().sqrt().detach().cpu()),
        "pde_momentum_x_rms": float(residual.momentum_x.square().mean().sqrt().detach().cpu()),
        "pde_momentum_y_rms": float(residual.momentum_y.square().mean().sqrt().detach().cpu()),
        "outlet_normal_pressure_gradient_rms": float(grad_p.square().mean().sqrt().detach().cpu()),
        "outlet_normal_velocity_gradient_rms": float(
            (grad_u.square() + grad_v.square()).mean().sqrt().detach().cpu()
        ),
        "diagnostic_vmax_m_s": speed,
        "diagnostic_reynolds": reynolds,
    }


def evaluate_model(
    model: PINN,
    geometry: BendGeometry,
    reference: ReferenceData,
    inlet_vmax_m_s: float,
    inlet_pressure_pa: float,
) -> dict[str, Any]:
    prediction = model.predict(
        geometry, reference.points_xy_m, inlet_vmax_m_s, inlet_pressure_pa
    )
    return {
        "validation_status": model.validation_status,
        "reference_points": len(reference.points_xy_m),
        "reference_metrics": reference_metrics(geometry, prediction, reference),
        "physics_metrics": physics_metrics(model, geometry),
    }
