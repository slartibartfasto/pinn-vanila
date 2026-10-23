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
        return sum(value.square() for value in self.as_dict().values())

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "continuity": self.continuity,
            "momentum_x": self.momentum_x,
            "momentum_y": self.momentum_y,
        }


@dataclass(frozen=True)
class RANSResiduals(NavierStokesResiduals):
    turbulent_k: torch.Tensor
    turbulent_omega: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            **super().as_dict(),
            "turbulent_k": self.turbulent_k,
            "turbulent_omega": self.turbulent_omega,
        }


def _gradient(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if not output.requires_grad:
        return inputs * 0.0
    gradient = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return inputs * 0.0 if gradient is None else gradient


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
    if network.config.turbulence.enabled:
        raise ValueError("Use rans_residuals or flow_residuals for a RANS network")
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


def rans_residuals(
    network: GeometryConditionedPINN,
    xy_m: torch.Tensor,
    reynolds: torch.Tensor,
    angle_deg: torch.Tensor,
) -> RANSResiduals:
    """Steady incompressible RANS, Boussinesq stress and Wilcox-1988 k-omega.

    Pressure is static mean p/(rho Umax^2), NOT p+2 rho k/3. The isotropic
    Reynolds stress therefore appears explicitly as +(2/3) grad(k).
    All diffusion terms are in divergence form, including gradients of nu_t.
    P_k = 2 nu_t S:S; alpha*(omega/k)*P_k simplifies to alpha*2 S:S.
    """
    if not xy_m.requires_grad:
        raise ValueError("xy_m must require gradients")
    closure = network.config.turbulence
    if not closure.enabled:
        raise ValueError("RANS residuals require turbulence.model=k_omega_1988")
    prediction = network(xy_m, reynolds, angle_deg)
    if prediction.shape[1] != 5:
        raise ValueError("RANS network must output u, v, static p, k and omega")
    u, v, pressure, k, omega = prediction.unbind(dim=1)
    length = network.geometry.pipe_width

    def gradient(value: torch.Tensor) -> torch.Tensor:
        return length * _gradient(value, xy_m)

    def divergence(flux_x: torch.Tensor, flux_y: torch.Tensor) -> torch.Tensor:
        return gradient(flux_x)[:, 0] + gradient(flux_y)[:, 1]

    grad_u, grad_v, grad_p, grad_k, grad_omega = (
        gradient(value) for value in (u, v, pressure, k, omega)
    )
    # Network transforms enforce omega>0. A machine-tiny floor only protects
    # pathological underflow; no turbulent-viscosity clipping or detach is used.
    nu_t = k / omega.clamp_min(torch.finfo(omega.dtype).tiny)
    nu = reynolds.reshape(-1).reciprocal()
    nu_eff = nu + nu_t
    strain_xx = 2.0 * grad_u[:, 0]
    strain_xy = grad_u[:, 1] + grad_v[:, 0]
    strain_yy = 2.0 * grad_v[:, 1]
    stress_div_x = divergence(nu_eff * strain_xx, nu_eff * strain_xy)
    stress_div_y = divergence(nu_eff * strain_xy, nu_eff * strain_yy)
    strain_contraction = (
        2.0 * grad_u[:, 0].square()
        + 2.0 * grad_v[:, 1].square()
        + strain_xy.square()
    )
    production = nu_t * strain_contraction
    momentum_x = (
        u * grad_u[:, 0] + v * grad_u[:, 1]
        + grad_p[:, 0] + (2.0 / 3.0) * grad_k[:, 0] - stress_div_x
    )
    momentum_y = (
        u * grad_v[:, 0] + v * grad_v[:, 1]
        + grad_p[:, 1] + (2.0 / 3.0) * grad_k[:, 1] - stress_div_y
    )
    diffusion_k = divergence(
        (nu + closure.sigma_k * nu_t) * grad_k[:, 0],
        (nu + closure.sigma_k * nu_t) * grad_k[:, 1],
    )
    diffusion_omega = divergence(
        (nu + closure.sigma_omega * nu_t) * grad_omega[:, 0],
        (nu + closure.sigma_omega * nu_t) * grad_omega[:, 1],
    )
    residual_k = (
        u * grad_k[:, 0] + v * grad_k[:, 1] - diffusion_k
        - production + closure.beta_star * k * omega
    )
    residual_omega = (
        u * grad_omega[:, 0] + v * grad_omega[:, 1] - diffusion_omega
        - closure.alpha * strain_contraction + closure.beta * omega.square()
    )
    return RANSResiduals(
        grad_u[:, 0] + grad_v[:, 1], momentum_x, momentum_y,
        residual_k, residual_omega,
    )


def flow_residuals(
    network: GeometryConditionedPINN,
    xy_m: torch.Tensor,
    reynolds: torch.Tensor,
    angle_deg: torch.Tensor,
) -> NavierStokesResiduals:
    evaluate = rans_residuals if network.config.turbulence.enabled else navier_stokes_residuals
    return evaluate(network, xy_m, reynolds, angle_deg)


def all_outlet_normal_gradients(
    network: GeometryConditionedPINN,
    xy_m: torch.Tensor,
    reynolds: torch.Tensor,
    angle_deg: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction = network(xy_m, reynolds, angle_deg)
    tangent = torch.stack(
        (torch.cos(torch.deg2rad(angle_deg)), torch.sin(torch.deg2rad(angle_deg))), dim=1
    )
    fields = ("u", "v", "p", "k", "omega")[: prediction.shape[1]]
    return {
        name: (network.geometry.pipe_width * _gradient(prediction[:, index], xy_m) * tangent).sum(dim=1)
        for index, name in enumerate(fields)
    }


def outlet_normal_gradients(
    network: GeometryConditionedPINN,
    xy_m: torch.Tensor,
    reynolds: torch.Tensor,
    angle_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dimensionless outward-normal derivatives of u, v and pi."""
    values = all_outlet_normal_gradients(network, xy_m, reynolds, angle_deg)
    return values["u"], values["v"], values["p"]


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
