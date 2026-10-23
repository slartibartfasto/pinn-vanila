"""Configuration of the standard incompressible Wilcox (1988) k-omega model.

All quantities here are dimensionless: k*=k/Umax^2, omega*=omega*H/Umax,
and nu_t*=nu_t/(Umax*H). This is not an SST or transition model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping

import torch

from .errors import ConfigurationError


@dataclass(frozen=True)
class TurbulenceConfig:
    model: str = "none"
    alpha: float = 5.0 / 9.0
    beta: float = 0.075
    beta_star: float = 0.09
    sigma_k: float = 0.5
    sigma_omega: float = 0.5
    inlet_intensity: float = 0.05
    inlet_length_scale_over_width: float = 0.07
    wall_omega_distance_over_width: float = 0.01

    def __post_init__(self) -> None:
        if self.model not in {"none", "k_omega_1988"}:
            raise ConfigurationError("turbulence.model must be none or k_omega_1988")
        for item in fields(self):
            if item.name == "model":
                continue
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"turbulence.{item.name} must be finite and positive")
        if self.inlet_intensity >= 1.0:
            raise ConfigurationError("turbulence.inlet_intensity must be a fraction in (0, 1)")
        if self.wall_omega_distance_over_width >= 0.5:
            raise ConfigurationError("wall_omega_distance_over_width must lie in (0, 0.5)")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TurbulenceConfig":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ConfigurationError("turbulence must be a mapping")
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise ConfigurationError("Unsupported turbulence options: " + ", ".join(sorted(unknown)))
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def enabled(self) -> bool:
        return self.model != "none"

    @property
    def k_reference(self) -> float:
        """Inlet centerline k*, with intensity defined relative to Vmax."""
        return 1.5 * self.inlet_intensity**2

    @property
    def omega_reference(self) -> float:
        return math.sqrt(self.k_reference) / (
            self.beta_star**0.25 * self.inlet_length_scale_over_width
        )

    def inlet_targets(
        self, eta: torch.Tensor, reynolds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatible inlet/wall values with a finite two-wall omega profile.

        omega has the 6 nu/(beta d^2) near-wall form regularized by d_ref.
        Distances to the two walls are smooth across the centerline. d_ref is
        an explicit modeling approximation, NOT a computed y+ or a wall law.
        """
        envelope = (1.0 - eta.square()).clamp_min(0.0)
        k = self.k_reference * envelope.square()
        delta = self.wall_omega_distance_over_width
        lower = 0.5 * (1.0 + eta)
        upper = 0.5 * (1.0 - eta)
        wall_profile = (
            (lower + delta).reciprocal().square()
            + (upper + delta).reciprocal().square()
            - 2.0 / (0.5 + delta)**2
        )
        omega = self.omega_reference + 6.0 / (self.beta * reynolds.reshape(-1)) * wall_profile
        return k, omega
