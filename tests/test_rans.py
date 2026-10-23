"""Analytical operator checks and end-to-end tests for the k-omega closure."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pinn_vanila import PINN, ProjectConfig
from pinn_vanila.errors import ConfigurationError
from pinn_vanila.export import export_prediction
from pinn_vanila.network import GeometryConditionedPINN, NetworkConfig
from pinn_vanila.physics import flow_residuals, navier_stokes_residuals, rans_residuals
from pinn_vanila.trainer import Trainer
from pinn_vanila.turbulence import TurbulenceConfig
from pinn_vanila.validation import structural_validation


ROOT = Path(__file__).resolve().parents[1]


class ManufacturedField(nn.Module):
    """Analytical fields in x*=x/H and y*=y/H, independent of PINN transforms."""

    def __init__(self, kind, turbulence=True, width=0.5):
        super().__init__()
        self.kind = kind
        self.geometry = SimpleNamespace(pipe_width=width)
        self.config = SimpleNamespace(turbulence=TurbulenceConfig(
            model="k_omega_1988" if turbulence else "none"
        ))
        self.amplitude = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, xy, reynolds, angle):
        x, y = (xy / self.geometry.pipe_width).unbind(dim=1)
        zero = x * 0.0
        if self.kind == "shear":
            u, v = self.amplitude * 1.7 * y, zero
            k = 0.3 + 0.2 * y + 0.1 * y.square()
            omega = zero + 2.3
            pressure = -(2.0 / 3.0) * k
        elif self.kind == "omega_diffusion":
            u, v, pressure = zero, zero, zero
            k = zero + 0.3
            omega = 2.3 + 0.4 * y + 0.2 * y.square()
        elif self.kind == "extension":
            u, v = 1.7 * x, -1.7 * y
            pressure = -0.5 * 1.7**2 * (x.square() + y.square())
            k, omega = zero + 0.3, zero + 2.3
        elif self.kind == "poiseuille":
            u, v = 1.0 - 4.0 * y.square(), zero
            pressure = -8.0 * x / reynolds
            k, omega = zero, zero + 2.3
        elif self.kind == "transport":
            u, v, pressure = zero + 1.2, zero - 0.4, zero
            k = 0.3 + 0.1*x + 0.2*y
            omega = 2.3 + 0.3*x + 0.4*y
        else:
            raise AssertionError(self.kind)
        values = [u, v, pressure]
        if self.config.turbulence.enabled:
            values.extend((k, omega))
        return torch.stack(values, dim=1)


class RANSOperatorTests(unittest.TestCase):
    def setUp(self):
        self.xy = torch.tensor(
            [[0.2, -0.13], [0.4, 0.04], [0.8, 0.19]],
            dtype=torch.float64, requires_grad=True,
        )
        self.re = torch.tensor([500.0, 2000.0, 10000.0], dtype=torch.float64)
        self.angles = torch.full((3,), 90.0, dtype=torch.float64)

    def test_variable_viscosity_stress_production_and_k_diffusion(self):
        field = ManufacturedField("shear")
        residual = rans_residuals(field, self.xy, self.re, self.angles)
        y = self.xy[:, 1] / field.geometry.pipe_width
        a, omega = 1.7, 2.3
        k, ky, kyy = 0.3 + 0.2*y + 0.1*y**2, 0.2 + 0.2*y, 0.2
        diffusion = (1/self.re + 0.5*k/omega)*kyy + 0.5*ky**2/omega
        torch.testing.assert_close(residual.continuity, torch.zeros_like(y))
        torch.testing.assert_close(residual.momentum_x, -a*ky/omega)
        # Static pressure balances the isotropic Reynolds stress, not an
        # implicitly redefined pressure that absorbs 2k/3.
        torch.testing.assert_close(residual.momentum_y, torch.zeros_like(y), atol=1e-14, rtol=0)
        torch.testing.assert_close(residual.turbulent_k, -diffusion-k/omega*a**2+0.09*k*omega)
        torch.testing.assert_close(residual.turbulent_omega, torch.full_like(y, 0.075*omega**2-(5/9)*a**2))
        residual.squared_score().mean().backward()
        self.assertTrue(torch.isfinite(field.amplitude.grad))
        self.assertGreater(abs(float(field.amplitude.grad)), 0.0)

    def test_variable_omega_diffusion_uses_coefficient_gradient(self):
        field = ManufacturedField("omega_diffusion")
        residual = rans_residuals(field, self.xy, self.re, self.angles)
        y = self.xy[:, 1] / field.geometry.pipe_width
        omega, wy, wyy = 2.3+0.4*y+0.2*y**2, 0.4+0.4*y, 0.4
        diffusion = (1/self.re+0.5*0.3/omega)*wyy-0.5*0.3*wy**2/omega**2
        torch.testing.assert_close(residual.turbulent_omega, -diffusion+0.075*omega**2)
        torch.testing.assert_close(residual.turbulent_k, 0.09*0.3*omega)

    def test_extensional_strain_production_factor(self):
        residual = rans_residuals(ManufacturedField("extension"), self.xy, self.re, self.angles)
        torch.testing.assert_close(residual.momentum_x, torch.zeros_like(self.re), atol=1e-14, rtol=0)
        torch.testing.assert_close(residual.momentum_y, torch.zeros_like(self.re), atol=1e-14, rtol=0)
        expected = -4*0.3/2.3*1.7**2+0.09*0.3*2.3
        torch.testing.assert_close(residual.turbulent_k, torch.full_like(self.re, expected))

    def test_zero_eddy_viscosity_recovers_laminar_poiseuille_momentum(self):
        rans = flow_residuals(ManufacturedField("poiseuille"), self.xy, self.re, self.angles)
        laminar = navier_stokes_residuals(
            ManufacturedField("poiseuille", turbulence=False), self.xy, self.re, self.angles
        )
        for name in ("continuity", "momentum_x", "momentum_y"):
            torch.testing.assert_close(getattr(rans, name), getattr(laminar, name), atol=1e-14, rtol=0)
            torch.testing.assert_close(getattr(rans, name), torch.zeros_like(self.re), atol=1e-14, rtol=0)

    def test_transport_advection_and_diffusion_in_both_coordinates(self):
        field = ManufacturedField("transport")
        residual = rans_residuals(field, self.xy, self.re, self.angles)
        x, y = (self.xy / field.geometry.pipe_width).unbind(dim=1)
        k, omega = 0.3+0.1*x+0.2*y, 2.3+0.3*x+0.4*y
        viscosity_x = (0.1*omega-0.3*k) / omega**2
        viscosity_y = (0.2*omega-0.4*k) / omega**2
        expected_k = 1.2*0.1-0.4*0.2-0.5*(0.1*viscosity_x+0.2*viscosity_y)+0.09*k*omega
        expected_omega = 1.2*0.3-0.4*0.4-0.5*(0.3*viscosity_x+0.4*viscosity_y)+0.075*omega**2
        torch.testing.assert_close(residual.turbulent_k, expected_k)
        torch.testing.assert_close(residual.turbulent_omega, expected_omega)
        torch.testing.assert_close(residual.momentum_x, torch.full_like(self.re, (2/3)*0.1))
        torch.testing.assert_close(residual.momentum_y, torch.full_like(self.re, (2/3)*0.2))


def small_rans_model():
    config = ProjectConfig.from_yaml(ROOT / "configs/pinn_vanila_config.yaml")
    network = GeometryConditionedPINN(
        config.case.geometry, config.reynolds_bounds,
        NetworkConfig(hidden_layers=2, hidden_width=12, fourier_enabled=False, turbulence=config.turbulence),
    )
    return PINN(network, config.case, config.velocity_bounds), config


class RANSWorkflowTests(unittest.TestCase):
    def test_hard_boundary_conditions_and_positive_outputs_across_family(self):
        model, config = small_rans_model()
        for angle in (15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
            model.case = replace(model.case, geometry=replace(model.case.geometry, bend_angle=angle))
            checks = structural_validation(model, samples=17)
            for name in (
                "inlet_k_relative_max_error", "inlet_omega_relative_max_error",
                "wall_omega_relative_max_error", "wall_k_dimensionless_max",
            ):
                self.assertLess(checks[name], 1e-10, name)
            points = model.case.geometry.map(np.linspace(0.1, 3.9, 15), np.linspace(-0.8, 0.8, 15))
            for speed in (1.0, 20.0):
                prediction = model.predict(model.case.geometry, points, speed, 100000.0)
                self.assertTrue(np.all(prediction.k_m2_s2 > 0))
                self.assertTrue(np.all(prediction.omega_s_inv > 0))
                np.testing.assert_allclose(prediction.nu_t_m2_s, prediction.k_m2_s2/prediction.omega_s_inv)

    def test_units_gauge_persistence_and_csv_export(self):
        model, config = small_rans_model()
        geometry = config.case.geometry
        points = geometry.map(np.array([0.0, 0.4, 3.0]), np.array([0.0, 0.1, -0.3]))
        prediction = model.predict(geometry, points, 4.0, 180000.0)
        self.assertAlmostEqual(prediction.k_m2_s2[0], 4.0**2*config.turbulence.k_reference)
        self.assertAlmostEqual(prediction.omega_s_inv[0], 4.0/geometry.pipe_width*config.turbulence.omega_reference)
        shifted = model.predict(geometry, points, 4.0, 200000.0)
        np.testing.assert_allclose(shifted.p_pa-prediction.p_pa, 20000.0)
        np.testing.assert_array_equal(shifted.k_m2_s2, prediction.k_m2_s2)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            loaded = PINN.load(directory)
            self.assertEqual(loaded.network.config.turbulence, config.turbulence)
            actual = loaded.predict(geometry, points, 4.0, 180000.0)
            np.testing.assert_allclose(actual.omega_s_inv, prediction.omega_s_inv)
            np.testing.assert_allclose(actual.p_pa, prediction.p_pa)
            output = Path(directory) / "prediction.csv"
            export_prediction(output, points, actual)
            data = np.genfromtxt(output, delimiter=",", names=True)
            self.assertIn("k_m2_s2", data.dtype.names)
            np.testing.assert_allclose(data["nu_t_m2_s"], actual.nu_t_m2_s)

    def test_laminar_config_and_legacy_checkpoint_remain_supported(self):
        config = ProjectConfig.from_yaml(ROOT / "configs/pinn_laminar_config.yaml")
        self.assertFalse(config.turbulence.enabled)
        model = PINN.from_config(config)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            path = Path(directory) / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["network"].pop("turbulence")
            metadata.pop("turbulence")
            # Recreate the metadata shape written before RANS support.
            path.write_text(json.dumps(metadata))
            loaded = PINN.load(directory)
            self.assertFalse(loaded.network.config.turbulence.enabled)
            self.assertEqual(loaded.network.output.out_features, 3)

    def test_training_rar_adaptive_weights_and_lbfgs_use_rans(self):
        model, config = small_rans_model()
        config.raw["training"]["optimizer_polishing"]["max_iterations"] = 1
        config.raw["collocation"]["interior_candidate_pool"] = 8
        config.raw["collocation"]["residual_adaptive_refinement"].update(
            interval_steps=1, added_points_each_round=4
        )
        trainer = Trainer(model, config, "cpu")
        trainer.adaptive_interval = 1
        result = trainer.train(max_steps=2, interior_batch=4, boundary_batch=4)
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(trainer.adaptive_points.xy), 4)
        self.assertTrue(model.training_summary["lbfgs_polishing"]["enabled"])
        for name in ("turbulent_k", "turbulent_omega", "outlet_k_gradient", "outlet_omega_gradient"):
            self.assertTrue(np.isfinite(result.final_losses[name]), name)

    def test_configuration_rejects_inconsistent_equations_and_unknown_closures(self):
        model, config = small_rans_model()
        config.raw["physical_problem"]["equations"] = "steady_incompressible_navier_stokes_2d"
        with self.assertRaises(ConfigurationError):
            config.validate_supported_contract()
        for values in ({"model": "sst"}, {"inlet_intensity": -0.1}, {"beta": 0}, {"typo": 1}):
            with self.assertRaises(ConfigurationError):
                TurbulenceConfig.from_dict(values)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is not attached")
    def test_rans_second_derivatives_and_optimizer_on_cuda(self):
        model, config = small_rans_model()
        trainer = Trainer(model, config, "cuda")
        result = trainer.train(
            max_steps=1, interior_batch=4, boundary_batch=4,
            enable_refinement=False, enable_polishing=False,
        )
        self.assertEqual(model.device.type, "cuda")
        self.assertTrue(all(np.isfinite(value) for value in result.final_losses.values()))


if __name__ == "__main__":
    unittest.main()
