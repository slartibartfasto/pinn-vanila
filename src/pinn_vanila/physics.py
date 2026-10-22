"""Strong-form residuals and analytical consistency checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .network import GeometryConditionedPINN


@dataclass(frozen=True)
class NavierStokesResiduals:
    continuity: torch.Tensor
    momentum_x: torch.Tensor
    momentum_y: torch.Tensor

    def squared_score(self) -> torch.Tensor:
        return self.continuity.square() + self.momentum_x.square() + self.momentum_y.square()


def _gradient(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def navier_stokes_residuals(
    network: GeometryConditionedPINN,
    xy_m: torch.Tensor,
    reynolds: torch.Tensor,
    angle_deg: torch.Tensor,
) -> NavierStokesResiduals:
    """Evaluate dimensionless steady incompressible Navier--Stokes residuals.

    Derivatives originate from physical ``x,y`` tensors and are scaled to
    ``x*=x/H, y*=y/H``. This avoids treating centerline coordinates as Cartesian.
    """
    if not xy_m.requires_grad:
        raise ValueError("xy_m must require gradients")
    prediction = network(xy_m, reynolds, angle_deg)
    u, v, pressure = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    length = network.geometry.pipe_width

    grad_u_physical = _gradient(u, xy_m)
    grad_v_physical = _gradient(v, xy_m)
    grad_p_physical = _gradient(pressure, xy_m)
    grad_u = length * grad_u_physical
    grad_v = length * grad_v_physical
    grad_p = length * grad_p_physical

    u_xx = length**2 * _gradient(grad_u_physical[:, 0], xy_m)[:, 0]
    u_yy = length**2 * _gradient(grad_u_physical[:, 1], xy_m)[:, 1]
    v_xx = length**2 * _gradient(grad_v_physical[:, 0], xy_m)[:, 0]
    v_yy = length**2 * _gradient(grad_v_physical[:, 1], xy_m)[:, 1]

    inverse_re = reynolds.reshape(-1).reciprocal()
    continuity = grad_u[:, 0] + grad_v[:, 1]
    momentum_x = u * grad_u[:, 0] + v * grad_u[:, 1] + grad_p[:, 0] - inverse_re * (u_xx + u_yy)
    momentum_y = u * grad_v[:, 0] + v * grad_v[:, 1] + grad_p[:, 1] - inverse_re * (v_xx + v_yy)
    return NavierStokesResiduals(continuity, momentum_x, momentum_y)


def outlet_normal_gradients(
    network: GeometryConditionedPINN,
    xy_m: torch.Tensor,
    reynolds: torch.Tensor,
    angle_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dimensionless outward-normal derivatives of u, v and pi."""
    prediction = network(xy_m, reynolds, angle_deg)
    length = network.geometry.pipe_width
    tangent = torch.stack(
        (torch.cos(torch.deg2rad(angle_deg)), torch.sin(torch.deg2rad(angle_deg))), dim=1
    )
    values = []
    for component in range(3):
        gradient = length * _gradient(prediction[:, component], xy_m)
        values.append((gradient * tangent).sum(dim=1))
    return values[0], values[1], values[2]


def poiseuille_solution(
    x_star: np.ndarray, eta: np.ndarray, reynolds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatible straight-channel analytical solution in dimensionless units."""
    x_star, eta = np.broadcast_arrays(np.asarray(x_star, float), np.asarray(eta, float))
    u = 1.0 - eta**2
    v = np.zeros_like(u)
    pi = -8.0 * x_star / float(reynolds)
    return u, v, pi


def poiseuille_pressure_gradient_pa_m(
    dynamic_viscosity_pa_s: float, inlet_vmax_m_s: float, width_m: float
) -> float:
    """Analytical pressure gradient; it is deliberately nonzero at the outlet."""
    return -8.0 * dynamic_viscosity_pa_s * inlet_vmax_m_s / width_m**2
