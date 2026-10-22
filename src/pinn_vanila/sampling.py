"""Sobol sampling in centerline coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .case import BendGeometry
from .network import TorchBendMap


@dataclass
class SampleBatch:
    xy: torch.Tensor
    reynolds: torch.Tensor
    angle_deg: torch.Tensor


class PhysicsSampler:
    def __init__(
        self,
        geometry: BendGeometry,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        seed: int = 42,
    ):
        self.geometry = geometry
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = int(seed)
        self._counter = 0
        self.bend_map = TorchBendMap(geometry)

    def _draw(self, n: int, dimensions: int) -> torch.Tensor:
        # A fresh deterministic scramble avoids dimension coupling between samplers.
        engine = torch.quasirandom.SobolEngine(
            dimensions, scramble=True, seed=self.seed + self._counter
        )
        self._counter += 1
        return engine.draw(n).to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _parameters(
        unit_re: torch.Tensor,
        unit_angle: torch.Tensor,
        reynolds_bounds: tuple[float, float],
        angle_bounds: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_re, high_re = reynolds_bounds
        log_low = unit_re.new_tensor(low_re).log()
        reynolds = torch.exp(log_low + unit_re * (unit_re.new_tensor(high_re).log() - log_low))
        low_angle, high_angle = angle_bounds
        angle = low_angle + unit_angle * (high_angle - low_angle)
        return reynolds, angle

    def interior(
        self,
        n: int,
        reynolds_bounds: tuple[float, float],
        angle_bounds: tuple[float, float],
    ) -> SampleBatch:
        unit = self._draw(n, 4)
        reynolds, angle = self._parameters(
            unit[:, 2], unit[:, 3], reynolds_bounds, angle_bounds
        )
        alpha = torch.deg2rad(angle)
        total = self.geometry.inlet_length + self.geometry.bend_radius * alpha + self.geometry.outlet_length
        s = (1e-6 + (1.0 - 2e-6) * unit[:, 0]) * total
        eta = -1.0 + 2.0 * unit[:, 1]
        return SampleBatch(self.bend_map.map(s, eta, angle), reynolds, angle)

    def outlet(
        self,
        n: int,
        reynolds_bounds: tuple[float, float],
        angle_bounds: tuple[float, float],
    ) -> SampleBatch:
        unit = self._draw(n, 3)
        reynolds, angle = self._parameters(
            unit[:, 1], unit[:, 2], reynolds_bounds, angle_bounds
        )
        alpha = torch.deg2rad(angle)
        s = self.geometry.inlet_length + self.geometry.bend_radius * alpha + self.geometry.outlet_length
        eta = -1.0 + 2.0 * unit[:, 0]
        return SampleBatch(self.bend_map.map(s, eta, angle), reynolds, angle)

    def flux_sections(
        self,
        cases: int,
        points_per_case: int,
        reynolds_bounds: tuple[float, float],
        angle_bounds: tuple[float, float],
    ) -> tuple[SampleBatch, torch.Tensor]:
        unit_cases = self._draw(cases, 3)
        re_case, angle_case = self._parameters(
            unit_cases[:, 1], unit_cases[:, 2], reynolds_bounds, angle_bounds
        )
        alpha = torch.deg2rad(angle_case)
        total = self.geometry.inlet_length + self.geometry.bend_radius * alpha + self.geometry.outlet_length
        # Avoid interfaces and boundaries for stable inverse differentiation.
        s_case = (0.05 + 0.90 * unit_cases[:, 0]) * total
        eta = torch.linspace(
            -1.0, 1.0, points_per_case, device=self.device, dtype=self.dtype
        ).repeat(cases)
        s = s_case.repeat_interleave(points_per_case)
        angle = angle_case.repeat_interleave(points_per_case)
        reynolds = re_case.repeat_interleave(points_per_case)
        xy = self.bend_map.map(s, eta, angle)
        local_theta = (s - self.geometry.inlet_length) / self.geometry.bend_radius
        local_theta = torch.minimum(torch.clamp(local_theta, min=0.0), torch.deg2rad(angle))
        tangent = torch.stack((torch.cos(local_theta), torch.sin(local_theta)), dim=1)
        return SampleBatch(xy, reynolds, angle), tangent
