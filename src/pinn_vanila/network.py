"""Geometry-conditioned residual PINN network."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .case import BendGeometry
from .turbulence import TurbulenceConfig


def _positive_scale(value: torch.Tensor) -> torch.Tensor:
    """Strictly positive smooth factor equal to one at zero."""
    return (F.softplus(value) + 1e-12) / (math.log(2.0) + 1e-12)


@dataclass(frozen=True)
class NetworkConfig:
    hidden_layers: int = 8
    hidden_width: int = 128
    activation: str = "tanh"
    residual_skip_interval: int = 2
    parameter_conditioning: str = "film"
    fourier_enabled: bool = True
    fourier_features_per_scale: int = 16
    fourier_scales: tuple[float, ...] = (1.0, 4.0)
    precision: str = "float64"
    seed: int = 42
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)

    @classmethod
    def from_spec(
        cls, data: Mapping[str, Any], seed: int = 42,
        turbulence: TurbulenceConfig | None = None,
    ) -> "NetworkConfig":
        fourier = data.get("fourier_embedding", {}) or {}
        return cls(
            hidden_layers=int(data.get("hidden_layers", 8)),
            hidden_width=int(data.get("hidden_width", 128)),
            activation=str(data.get("activation", "tanh")),
            residual_skip_interval=int(data.get("residual_skip_interval", 2)),
            parameter_conditioning=str(data.get("parameter_conditioning", "film")),
            fourier_enabled=bool(fourier.get("enabled", True)),
            fourier_features_per_scale=int(fourier.get("spatial_features_per_scale", 16)),
            fourier_scales=tuple(float(x) for x in fourier.get("scales", [1.0, 4.0])),
            precision=str(data.get("precision", "float64")),
            seed=int(seed),
            turbulence=turbulence or TurbulenceConfig(),
        )

    @property
    def dtype(self) -> torch.dtype:
        if self.precision == "float64":
            return torch.float64
        if self.precision == "float32":
            return torch.float32
        raise ValueError(f"Unsupported precision: {self.precision}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fourier_scales"] = list(self.fourier_scales)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NetworkConfig":
        values = dict(data)
        values["fourier_scales"] = tuple(values["fourier_scales"])
        values["turbulence"] = TurbulenceConfig.from_dict(values.get("turbulence"))
        return cls(**values)


class TorchBendMap:
    """Differentiable physical-coordinate features for the supported bend."""

    def __init__(self, geometry: BendGeometry):
        self.geometry = geometry

    def map(
        self, s: torch.Tensor, eta: torch.Tensor, angle_deg: torch.Tensor
    ) -> torch.Tensor:
        g = self.geometry
        alpha = torch.deg2rad(angle_deg)
        arc_end = g.inlet_length + g.bend_radius * alpha
        theta = (s - g.inlet_length) / g.bend_radius
        theta = torch.minimum(torch.clamp(theta, min=0.0), alpha)
        arc_x = g.inlet_length + g.bend_radius * torch.sin(theta)
        arc_y = g.bend_radius * (1.0 - torch.cos(theta))
        x = arc_x - 0.5 * g.pipe_width * eta * torch.sin(theta)
        y = arc_y + 0.5 * g.pipe_width * eta * torch.cos(theta)
        inlet = s <= g.inlet_length
        x = torch.where(inlet, s, x)
        y = torch.where(inlet, 0.5 * g.pipe_width * eta, y)
        outlet = s >= arc_end
        tau = s - arc_end
        end_x = g.inlet_length + g.bend_radius * torch.sin(alpha)
        end_y = g.bend_radius * (1.0 - torch.cos(alpha))
        x_out = end_x + tau * torch.cos(alpha) - 0.5 * g.pipe_width * eta * torch.sin(alpha)
        y_out = end_y + tau * torch.sin(alpha) + 0.5 * g.pipe_width * eta * torch.cos(alpha)
        return torch.stack((torch.where(outlet, x_out, x), torch.where(outlet, y_out, y)), dim=1)

    def inverse_features(
        self, xy_m: torch.Tensor, angle_deg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return s, eta, wall distance/H, tangent and normalized s."""
        g = self.geometry
        angle_deg = angle_deg.reshape(-1)
        x, y = xy_m[:, 0], xy_m[:, 1]
        alpha = torch.deg2rad(angle_deg)
        half_width = 0.5 * g.pipe_width

        s_in = x
        eta_in = y / half_width
        tolerance = 1e-9
        valid_in = (s_in >= -tolerance) & (s_in <= g.inlet_length + tolerance)

        dx, dy = x - g.inlet_length, y - g.bend_radius
        theta = torch.atan2(dx, -dy)
        radius = torch.sqrt(dx.square() + dy.square())
        s_arc = g.inlet_length + g.bend_radius * theta
        eta_arc = (g.bend_radius - radius) / half_width
        valid_arc = (theta >= -tolerance) & (theta <= alpha + tolerance)

        arc_end = g.inlet_length + g.bend_radius * alpha
        end_x = g.inlet_length + g.bend_radius * torch.sin(alpha)
        end_y = g.bend_radius * (1.0 - torch.cos(alpha))
        rel_x, rel_y = x - end_x, y - end_y
        tau = rel_x * torch.cos(alpha) + rel_y * torch.sin(alpha)
        eta_out = (-rel_x * torch.sin(alpha) + rel_y * torch.cos(alpha)) / half_width
        s_out = arc_end + tau
        valid_out = (tau >= -tolerance) & (tau <= g.outlet_length + tolerance)

        candidates_s = torch.stack((s_in, s_arc, s_out), dim=1)
        candidates_eta = torch.stack((eta_in, eta_arc, eta_out), dim=1)
        valid = torch.stack((valid_in, valid_arc, valid_out), dim=1)
        scores = torch.where(valid, candidates_eta.abs(), torch.full_like(candidates_eta, 1e30))
        index = scores.argmin(dim=1, keepdim=True)
        s = candidates_s.gather(1, index).squeeze(1)
        eta = candidates_eta.gather(1, index).squeeze(1)

        local_theta = (s - g.inlet_length) / g.bend_radius
        local_theta = torch.minimum(torch.clamp(local_theta, min=0.0), alpha)
        tangent = torch.stack((torch.cos(local_theta), torch.sin(local_theta)), dim=1)
        wall_distance_over_h = 0.5 * (1.0 - eta.abs())
        total = g.inlet_length + g.bend_radius * alpha + g.outlet_length
        return s, eta, wall_distance_over_h, tangent, s / total


class GeometryConditionedPINN(nn.Module):
    """Residual MLP with FiLM, Fourier features, and exact velocity BCs."""

    def __init__(
        self,
        geometry: BendGeometry,
        reynolds_bounds: tuple[float, float],
        config: NetworkConfig | None = None,
    ):
        super().__init__()
        self.geometry = geometry
        self.reynolds_bounds = tuple(float(v) for v in reynolds_bounds)
        self.config = config or NetworkConfig()
        if self.config.activation != "tanh":
            raise ValueError("Only tanh activation is supported for second-order PINN derivatives")
        if self.config.parameter_conditioning != "film":
            raise ValueError("Only FiLM parameter conditioning is supported")

        fourier_dim = 0
        if self.config.fourier_enabled:
            generator = torch.Generator().manual_seed(self.config.seed)
            for index, scale in enumerate(self.config.fourier_scales):
                matrix = torch.randn(
                    2, self.config.fourier_features_per_scale, generator=generator, dtype=self.config.dtype
                ) * scale
                self.register_buffer(f"fourier_matrix_{index}", matrix)
                fourier_dim += 2 * self.config.fourier_features_per_scale
        input_dim = 7 + fourier_dim
        width = self.config.hidden_width
        self.encoder = nn.Linear(input_dim, width)
        self.hidden = nn.ModuleList(
            nn.Linear(width, width) for _ in range(self.config.hidden_layers)
        )
        self.film = nn.ModuleList(
            nn.Linear(2, 2 * width) for _ in range(self.config.hidden_layers + 1)
        )
        self.output = nn.Linear(width, 5 if self.config.turbulence.enabled else 3)
        self._reset_parameters()
        self.to(dtype=self.config.dtype)
        self.bend_map = TorchBendMap(geometry)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        # FiLM begins as an identity transform: gamma=1 and beta=0.
        for layer in self.film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            with torch.no_grad():
                layer.bias[: self.config.hidden_width].fill_(1.0)

    def normalized_parameters(
        self, reynolds: torch.Tensor, angle_deg: torch.Tensor
    ) -> torch.Tensor:
        log_re = torch.log(reynolds.reshape(-1))
        low_re, high_re = self.reynolds_bounds
        re_norm = 2.0 * (log_re - torch.log(log_re.new_tensor(low_re))) / (
            torch.log(log_re.new_tensor(high_re)) - torch.log(log_re.new_tensor(low_re))
        ) - 1.0
        low_angle, high_angle = self.geometry.angle_bounds
        angle_norm = 2.0 * (angle_deg.reshape(-1) - low_angle) / (high_angle - low_angle) - 1.0
        return torch.stack((re_norm, angle_norm), dim=1)

    def _film(self, values: torch.Tensor, params: torch.Tensor, index: int) -> torch.Tensor:
        gamma, beta = self.film[index](params).chunk(2, dim=1)
        return gamma * values + beta

    def forward(
        self, xy_m: torch.Tensor, reynolds: torch.Tensor, angle_deg: torch.Tensor
    ) -> torch.Tensor:
        if reynolds.ndim == 0:
            reynolds = reynolds.expand(len(xy_m))
        if angle_deg.ndim == 0:
            angle_deg = angle_deg.expand(len(xy_m))
        s, eta, wall_distance, tangent, s_normalized = self.bend_map.inverse_features(
            xy_m, angle_deg
        )
        if self.config.turbulence.enabled:
            # Smooth wall proximity for second derivatives across the centerline.
            # Exact distances to BOTH walls are used by the omega boundary profile.
            wall_distance = 0.25 * (1.0 - eta.square())
        params = self.normalized_parameters(reynolds, angle_deg)
        base = [
            xy_m[:, 0] / self.geometry.pipe_width,
            xy_m[:, 1] / self.geometry.pipe_width,
            params[:, 0],
            params[:, 1],
            s / self.geometry.pipe_width,
            eta,
            wall_distance,
        ]
        encoded = [torch.stack(base, dim=1)]
        if self.config.fourier_enabled:
            spatial = torch.stack((s_normalized, eta), dim=1)
            for index in range(len(self.config.fourier_scales)):
                projection = 2.0 * torch.pi * spatial @ getattr(self, f"fourier_matrix_{index}")
                encoded.extend((torch.sin(projection), torch.cos(projection)))
        features = torch.cat(encoded, dim=1)

        hidden = torch.tanh(self._film(self.encoder(features), params, 0))
        interval = self.config.residual_skip_interval
        residual = hidden
        for index, layer in enumerate(self.hidden):
            hidden = torch.tanh(self._film(layer(hidden), params, index + 1))
            if interval > 0 and (index + 1) % interval == 0:
                hidden = hidden + residual
                residual = hidden
        raw = self.output(hidden)

        wall_envelope = torch.clamp(1.0 - eta.square(), min=0.0)
        inlet_ramp = s / (s + self.geometry.pipe_width)
        correction = wall_envelope[:, None] * inlet_ramp[:, None] * raw[:, :2]
        inlet_profile = wall_envelope[:, None] * tangent
        velocity = inlet_profile + correction
        outputs = [velocity, raw[:, 2:3]]
        if self.config.turbulence.enabled:
            k_inlet, omega_base = self.config.turbulence.inlet_targets(eta, reynolds)
            # Smooth positive transforms, identity at inlet. k=0 at walls and
            # omega equals the configured regularized wall value exactly.
            k = k_inlet * _positive_scale(inlet_ramp * raw[:, 3])
            omega = omega_base * _positive_scale(inlet_ramp * wall_envelope * raw[:, 4])
            outputs.extend((k[:, None], omega[:, None]))
        return torch.cat(outputs, dim=1)
