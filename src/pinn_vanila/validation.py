"""Fast structural validations that do not require a CFD reference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .model import PINN
from .physics import poiseuille_pressure_gradient_pa_m


def structural_validation(model: PINN, samples: int = 33) -> dict[str, Any]:
    geometry = model.case.geometry
    angle = geometry.bend_angle
    total = geometry.total_length(angle)
    s = np.linspace(0.0, total, samples)
    eta = np.linspace(-1.0, 1.0, samples)
    points = geometry.map(s, eta)
    s_back, eta_back, inside = geometry.inverse(points)

    device, dtype = model.device, model.dtype
    re_mid = np.sqrt(np.prod(model.network.reynolds_bounds))
    with torch.no_grad():
        inlet_eta = torch.linspace(-1.0, 1.0, samples, device=device, dtype=dtype)
        inlet_xy = torch.as_tensor(
            geometry.map(np.zeros(samples), inlet_eta.cpu().numpy()), device=device, dtype=dtype
        )
        reynolds = torch.full((samples,), re_mid, device=device, dtype=dtype)
        angles = torch.full((samples,), angle, device=device, dtype=dtype)
        inlet = model.network(inlet_xy, reynolds, angles)[:, :2]
        expected_u = 1.0 - inlet_eta.square()

        wall_s = torch.linspace(0.0, total, samples, device=device, dtype=dtype)
        wall_angle = torch.full((samples,), angle, device=device, dtype=dtype)
        wall_plus = model.network(
            model.network.bend_map.map(wall_s, torch.ones_like(wall_s), wall_angle),
            reynolds,
            wall_angle,
        )[:, :2]
        wall_minus = model.network(
            model.network.bend_map.map(wall_s, -torch.ones_like(wall_s), wall_angle),
            reynolds,
            wall_angle,
        )[:, :2]

    checks = {
        "geometry_roundtrip_max_s_m": float(np.max(np.abs(s_back - s))),
        "geometry_roundtrip_max_eta": float(np.max(np.abs(eta_back - eta))),
        "geometry_all_inside": bool(np.all(inside)),
        "minimum_jacobian_factor": geometry.minimum_jacobian_factor,
        "inlet_profile_max_error": float(
            torch.max(torch.abs(inlet[:, 0] - expected_u)).cpu()
        ),
        "inlet_cross_velocity_max": float(torch.max(torch.abs(inlet[:, 1])).cpu()),
        "wall_slip_dimensionless_max": float(
            torch.max(torch.cat((wall_plus.abs().flatten(), wall_minus.abs().flatten()))).cpu()
        ),
        "poiseuille_compatible_dpdx_pa_m_at_vmax_1": poiseuille_pressure_gradient_pa_m(
            model.case.fluid.dynamic_viscosity, 1.0, geometry.pipe_width
        ),
        "note": "Poiseuille uses its compatible nonzero outlet pressure gradient; it is not a zero-gradient outlet benchmark.",
    }
    if model.network.config.turbulence.enabled:
        closure = model.network.config.turbulence
        with torch.no_grad():
            full_inlet = model.network(inlet_xy, reynolds, angles)
            expected_k, expected_omega = closure.inlet_targets(inlet_eta, reynolds)
            checks["inlet_k_relative_max_error"] = float(
                ((full_inlet[:, 3] - expected_k).abs() / closure.k_reference).max().cpu()
            )
            checks["inlet_omega_relative_max_error"] = float(
                ((full_inlet[:, 4] - expected_omega).abs() / expected_omega).max().cpu()
            )
            wall_k_errors, wall_omega_errors = [], []
            for sign in (-1.0, 1.0):
                wall_eta = sign * torch.ones_like(wall_s)
                wall_points = model.network.bend_map.map(wall_s, wall_eta, wall_angle)
                fields = model.network(wall_points, reynolds, wall_angle)
                _, target_omega = closure.inlet_targets(wall_eta, reynolds)
                wall_k_errors.append(fields[:, 3].abs().max())
                wall_omega_errors.append(((fields[:, 4] - target_omega).abs() / target_omega).max())
            checks["wall_k_dimensionless_max"] = float(torch.stack(wall_k_errors).max().cpu())
            checks["wall_omega_relative_max_error"] = float(torch.stack(wall_omega_errors).max().cpu())
        checks["turbulence_model"] = closure.model
        checks["wall_omega_distance_over_width"] = closure.wall_omega_distance_over_width
    return checks
