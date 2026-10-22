"""Physics-only curriculum trainer."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .case import ProjectConfig
from .model import PINN
from .physics import navier_stokes_residuals, outlet_normal_gradients
from .sampling import PhysicsSampler, SampleBatch


@dataclass
class TrainingResult:
    steps: int
    elapsed_seconds: float
    final_losses: dict[str, float]
    history: list[dict[str, float]]


class Trainer:
    """Train a parameterized model using PDE and boundary residuals only."""

    def __init__(self, model: PINN, config: ProjectConfig, device: str | torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        seed = int((config.raw.get("project", {}) or {}).get("random_seed", 42))
        torch.manual_seed(seed)
        self.sampler = PhysicsSampler(
            config.case.geometry, device=self.device, dtype=model.dtype, seed=seed
        )
        loss_config = config.raw.get("loss", {}) or {}
        self.weights = {
            "continuity": float(loss_config.get("continuity", 1.0)),
            "momentum_x": float(loss_config.get("momentum_x", 1.0)),
            "momentum_y": float(loss_config.get("momentum_y", 1.0)),
            "outlet_pressure_gradient": float(loss_config.get("outlet_pressure_gradient", 1.0)),
            "outlet_velocity_gradient": float(loss_config.get("outlet_velocity_gradient", 1.0)),
            "mass_flux": float(loss_config.get("mass_flux", 5.0)),
        }
        self.adaptive_interval = int(loss_config.get("weighting_update_every_steps", 200))
        clip = loss_config.get("weighting_clip_range", [0.1, 100.0])
        self.weight_clip = (float(clip[0]), float(clip[1]))
        self.adaptive_points: SampleBatch | None = None

    def _curriculum(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        training = self.config.raw.get("training", {}) or {}
        re_stages = [tuple(map(float, values)) for values in training.get("reynolds_stages", [self.config.reynolds_bounds])]
        angle_stages = [
            tuple(map(float, values))
            for values in training.get("angle_stages_deg", [self.config.case.geometry.angle_bounds])
        ]
        count = max(len(re_stages), len(angle_stages))
        return [
            (re_stages[min(i, len(re_stages) - 1)], angle_stages[min(i, len(angle_stages) - 1)])
            for i in range(count)
        ]

    def _interior(
        self,
        n: int,
        reynolds_bounds: tuple[float, float],
        angle_bounds: tuple[float, float],
    ) -> SampleBatch:
        if self.adaptive_points is None:
            return self.sampler.interior(n, reynolds_bounds, angle_bounds)
        refinement = (self.config.raw.get("collocation", {}) or {}).get(
            "residual_adaptive_refinement", {}
        ) or {}
        uniform_fraction = float(refinement.get("preserve_uniform_sampling_fraction", 0.4))
        uniform_n = max(1, math.ceil(n * uniform_fraction))
        uniform = self.sampler.interior(uniform_n, reynolds_bounds, angle_bounds)
        adaptive_n = n - uniform_n
        index = torch.randint(
            len(self.adaptive_points.xy), (adaptive_n,), device=self.device
        )
        return SampleBatch(
            torch.cat((uniform.xy, self.adaptive_points.xy[index]), dim=0),
            torch.cat((uniform.reynolds, self.adaptive_points.reynolds[index]), dim=0),
            torch.cat((uniform.angle_deg, self.adaptive_points.angle_deg[index]), dim=0),
        )

    def _losses(
        self,
        reynolds_bounds: tuple[float, float],
        angle_bounds: tuple[float, float],
        interior_batch: int,
        boundary_batch: int,
        parameter_cases: int,
    ) -> dict[str, torch.Tensor]:
        interior = self._interior(interior_batch, reynolds_bounds, angle_bounds)
        xy = interior.xy.detach().requires_grad_(True)
        residuals = navier_stokes_residuals(
            self.model.network, xy, interior.reynolds, interior.angle_deg
        )
        losses = {
            "continuity": residuals.continuity.square().mean(),
            "momentum_x": residuals.momentum_x.square().mean(),
            "momentum_y": residuals.momentum_y.square().mean(),
        }

        outlet = self.sampler.outlet(boundary_batch, reynolds_bounds, angle_bounds)
        outlet_xy = outlet.xy.detach().requires_grad_(True)
        grad_u, grad_v, grad_p = outlet_normal_gradients(
            self.model.network, outlet_xy, outlet.reynolds, outlet.angle_deg
        )
        losses["outlet_pressure_gradient"] = grad_p.square().mean()
        losses["outlet_velocity_gradient"] = grad_u.square().mean() + grad_v.square().mean()

        points_per_case = max(8, boundary_batch // max(1, parameter_cases))
        flux_batch, tangent = self.sampler.flux_sections(
            parameter_cases, points_per_case, reynolds_bounds, angle_bounds
        )
        velocity = self.model.network(
            flux_batch.xy, flux_batch.reynolds, flux_batch.angle_deg
        )[:, :2]
        axial = (velocity * tangent).sum(dim=1).reshape(parameter_cases, points_per_case)
        # Trapezoidal mean over eta in [-1,1]; expected integral/2 is 2/3.
        mean_flux = torch.trapezoid(axial, dx=2.0 / (points_per_case - 1), dim=1) / 2.0
        losses["mass_flux"] = (mean_flux - 2.0 / 3.0).square().mean()
        return losses

    def _update_adaptive_weights(self, losses: dict[str, torch.Tensor]) -> None:
        parameter = self.model.network.output.weight
        norms = {}
        for name, loss in losses.items():
            gradient = torch.autograd.grad(
                loss, parameter, retain_graph=True, allow_unused=True
            )[0]
            norms[name] = float(gradient.norm().detach()) if gradient is not None else 0.0
        positive = [value for value in norms.values() if value > 0]
        if not positive:
            return
        target = sum(positive) / len(positive)
        low, high = self.weight_clip
        for name, norm in norms.items():
            if norm > 0:
                proposed = min(high, max(low, target / norm))
                self.weights[name] = 0.5 * self.weights[name] + 0.5 * proposed

    def _refine(
        self,
        reynolds_bounds: tuple[float, float],
        angle_bounds: tuple[float, float],
    ) -> None:
        collocation = self.config.raw.get("collocation", {}) or {}
        refinement = collocation.get("residual_adaptive_refinement", {}) or {}
        candidates = int(collocation.get("interior_candidate_pool", 100000))
        keep = min(int(refinement.get("added_points_each_round", 2048)), candidates)
        best_scores: torch.Tensor | None = None
        best_batch: SampleBatch | None = None
        chunk_size = min(4096, candidates)
        for start in range(0, candidates, chunk_size):
            batch = self.sampler.interior(
                min(chunk_size, candidates - start), reynolds_bounds, angle_bounds
            )
            xy = batch.xy.detach().requires_grad_(True)
            residuals = navier_stokes_residuals(
                self.model.network, xy, batch.reynolds, batch.angle_deg
            )
            score = residuals.squared_score().detach()
            count = min(keep, len(score))
            values, index = torch.topk(score, count)
            selected = SampleBatch(
                batch.xy[index].detach(), batch.reynolds[index].detach(), batch.angle_deg[index].detach()
            )
            if best_scores is None:
                best_scores, best_batch = values, selected
            else:
                combined_scores = torch.cat((best_scores, values))
                combined = SampleBatch(
                    torch.cat((best_batch.xy, selected.xy)),
                    torch.cat((best_batch.reynolds, selected.reynolds)),
                    torch.cat((best_batch.angle_deg, selected.angle_deg)),
                )
                best_scores, chosen = torch.topk(combined_scores, min(keep, len(combined_scores)))
                best_batch = SampleBatch(
                    combined.xy[chosen], combined.reynolds[chosen], combined.angle_deg[chosen]
                )
        self.adaptive_points = best_batch

    def train(
        self,
        *,
        max_steps: int | None = None,
        interior_batch: int | None = None,
        boundary_batch: int | None = None,
        checkpoint_directory: str | Path | None = None,
        enable_refinement: bool = True,
        enable_polishing: bool = True,
    ) -> TrainingResult:
        training = self.config.raw.get("training", {}) or {}
        adam = training.get("optimizer_adam", {}) or {}
        collocation = self.config.raw.get("collocation", {}) or {}
        per_stage = int(adam.get("max_steps_per_stage", 50000))
        stages = self._curriculum()
        requested_total = per_stage * len(stages) if max_steps is None else int(max_steps)
        if requested_total <= 0:
            raise ValueError("max_steps must be positive")
        interior_n = int(interior_batch or collocation.get("adam_interior_batch", 4096))
        boundary_n = int(boundary_batch or collocation.get("adam_boundary_batch", 1024))
        parameter_cases = int(collocation.get("parameter_samples_per_batch", 16))
        optimizer = torch.optim.Adam(
            self.model.network.parameters(), lr=float(adam.get("initial_lr", 5e-4))
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=requested_total,
            eta_min=float(adam.get("min_lr", 1e-5)),
        )
        clip_norm = float(adam.get("gradient_clip_norm", 1.0))
        checkpoint_every = int(training.get("checkpoint_every_steps", 2000))
        refinement = collocation.get("residual_adaptive_refinement", {}) or {}
        refinement_interval = int(refinement.get("interval_steps", 2000))
        refinement_enabled = bool(refinement.get("enabled", True)) and enable_refinement
        history: list[dict[str, float]] = []
        started = time.monotonic()
        global_step = 0
        latest: dict[str, torch.Tensor] = {}
        self.model.network.train()

        for reynolds_bounds, angle_bounds in stages:
            stage_steps = min(per_stage, requested_total - global_step)
            for _ in range(stage_steps):
                optimizer.zero_grad(set_to_none=True)
                latest = self._losses(
                    reynolds_bounds, angle_bounds, interior_n, boundary_n, parameter_cases
                )
                if self.adaptive_interval > 0 and (global_step + 1) % self.adaptive_interval == 0:
                    self._update_adaptive_weights(latest)
                total = sum(self.weights[name] * value for name, value in latest.items())
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), clip_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1

                if global_step == 1 or global_step % 100 == 0 or global_step == requested_total:
                    entry = {name: float(value.detach()) for name, value in latest.items()}
                    entry.update(step=float(global_step), total=float(total.detach()), lr=scheduler.get_last_lr()[0])
                    history.append(entry)
                if refinement_enabled and refinement_interval > 0 and global_step % refinement_interval == 0:
                    self._refine(reynolds_bounds, angle_bounds)
                if checkpoint_directory and checkpoint_every > 0 and global_step % checkpoint_every == 0:
                    self.model.training_summary = {"completed_steps": global_step, "history": history}
                    self.model.save(Path(checkpoint_directory) / f"step-{global_step}")
                if global_step >= requested_total:
                    break
            if global_step >= requested_total:
                break

        polishing_summary: dict[str, Any] = {"enabled": False}
        polishing = training.get("optimizer_polishing", {}) or {}
        polishing_iterations = int(polishing.get("max_iterations", 0))
        if enable_polishing and polishing_iterations > 0:
            maximum_points = int(polishing.get("maximum_polishing_points", 8192))
            fixed_interior = min(interior_n, maximum_points)
            fixed_boundary = min(boundary_n, max(8, maximum_points // 4))
            final_reynolds, final_angles = stages[-1]
            # Resetting both the Sobol counter and RNG inside the closure makes the
            # objective fixed across every line-search evaluation.
            sobol_counter = self.sampler._counter
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            )
            saved_adaptive = self.adaptive_points
            self.adaptive_points = None
            optimizer_lbfgs = torch.optim.LBFGS(
                self.model.network.parameters(),
                max_iter=polishing_iterations,
                line_search_fn=str(polishing.get("line_search_fn", "strong_wolfe")),
            )
            closure_calls = 0
            polishing_total = float("nan")

            def closure() -> torch.Tensor:
                nonlocal closure_calls, latest, polishing_total
                self.sampler._counter = sobol_counter
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, self.device)
                optimizer_lbfgs.zero_grad(set_to_none=True)
                latest = self._losses(
                    final_reynolds,
                    final_angles,
                    fixed_interior,
                    fixed_boundary,
                    parameter_cases,
                )
                total_loss = sum(self.weights[name] * value for name, value in latest.items())
                total_loss.backward()
                closure_calls += 1
                polishing_total = float(total_loss.detach())
                return total_loss

            optimizer_lbfgs.step(closure)
            self.adaptive_points = saved_adaptive
            polishing_summary = {
                "enabled": True,
                "configured_max_iterations": polishing_iterations,
                "closure_calls": closure_calls,
                "fixed_interior_points": fixed_interior,
                "fixed_boundary_points": fixed_boundary,
                "final_total": polishing_total,
            }

        final = {name: float(value.detach()) for name, value in latest.items()}
        elapsed = time.monotonic() - started
        self.model.training_summary = {
            "random_seed": int(
                (self.config.raw.get("project", {}) or {}).get("random_seed", 42)
            ),
            "completed_steps": global_step,
            "elapsed_seconds": elapsed,
            "final_losses": final,
            "loss_weights": self.weights,
            "history": history,
            "physics_only": True,
            "lbfgs_polishing": polishing_summary,
        }
        return TrainingResult(global_step, elapsed, final, history)
