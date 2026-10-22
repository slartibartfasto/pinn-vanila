import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinn_vanila import PINN, ProjectConfig
from pinn_vanila.errors import ExtrapolationError, GeometryError
from pinn_vanila.evaluation import ReferenceData, reference_metrics
from pinn_vanila.model import PredictionResult
from pinn_vanila.network import GeometryConditionedPINN, NetworkConfig
from pinn_vanila.physics import poiseuille_pressure_gradient_pa_m, poiseuille_solution


ROOT = Path(__file__).resolve().parents[1]


def small_model():
    config = ProjectConfig.from_yaml(ROOT / "configs/pinn_vanila_config.yaml")
    network_config = NetworkConfig(
        hidden_layers=2,
        hidden_width=16,
        fourier_enabled=True,
        fourier_features_per_scale=2,
        fourier_scales=(1.0,),
        precision="float64",
    )
    network = GeometryConditionedPINN(config.case.geometry, config.reynolds_bounds, network_config)
    return PINN(network, config.case, config.velocity_bounds), config


class ModelTests(unittest.TestCase):
    def test_hard_inlet_and_wall_boundaries(self):
        model, config = small_model()
        geometry = config.case.geometry
        dtype = model.dtype
        eta = torch.linspace(-1.0, 1.0, 21, dtype=dtype)
        angle = torch.full((21,), 90.0, dtype=dtype)
        reynolds = torch.full((21,), 500.0, dtype=dtype)
        inlet = model.network(
            torch.as_tensor(geometry.map(np.zeros(21), eta.numpy()), dtype=dtype), reynolds, angle
        )
        torch.testing.assert_close(inlet[:, 0], 1.0 - eta.square(), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(inlet[:, 1], torch.zeros_like(eta), atol=1e-14, rtol=0)

        s = torch.linspace(0.0, geometry.total_length(), 21, dtype=dtype)
        for wall_eta in (-1.0, 1.0):
            wall = model.network.bend_map.map(s, torch.full_like(s, wall_eta), angle)
            velocity = model.network(wall, reynolds, angle)[:, :2]
            torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=1e-14, rtol=0)

    def test_pressure_gauge_shift_does_not_change_velocity(self):
        model, config = small_model()
        geometry = config.case.geometry
        points = geometry.map(np.array([0.5, 2.5, 4.0]), np.array([0.0, 0.2, -0.3]))
        first = model.predict(geometry, points, 2.0, 100000.0)
        second = model.predict(geometry, points, 2.0, 180000.0)
        np.testing.assert_array_equal(first.u_m_s, second.u_m_s)
        np.testing.assert_array_equal(first.v_m_s, second.v_m_s)
        np.testing.assert_allclose(second.p_pa - first.p_pa, 80000.0, atol=1e-9)

    def test_domain_checks(self):
        model, config = small_model()
        geometry = config.case.geometry
        points = np.array([[0.5, 0.0]])
        with self.assertRaises(ExtrapolationError):
            model.predict(geometry, points, 25.0, 100000.0)
        with self.assertRaises(GeometryError):
            model.predict(geometry, np.array([[0.5, 2.0]]), 2.0, 100000.0)

    def test_save_and_load_preserves_prediction(self):
        model, config = small_model()
        points = config.case.geometry.map(np.array([0.2, 3.0]), np.array([0.1, -0.1]))
        expected = model.predict(config.case.geometry, points, 3.0, 120000.0)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            loaded = PINN.load(directory)
            actual = loaded.predict(config.case.geometry, points, 3.0, 120000.0)
        np.testing.assert_allclose(actual.u_m_s, expected.u_m_s, rtol=1e-13)
        np.testing.assert_allclose(actual.v_m_s, expected.v_m_s, rtol=1e-13)
        np.testing.assert_allclose(actual.p_pa, expected.p_pa, rtol=1e-13)

    def test_poiseuille_uses_compatible_nonzero_gradient(self):
        u, v, pressure = poiseuille_solution(np.array([0.0, 1.0]), np.array([0.0, 0.0]), 500.0)
        np.testing.assert_allclose(u, 1.0)
        np.testing.assert_allclose(v, 0.0)
        self.assertAlmostEqual(pressure[1] - pressure[0], -8.0 / 500.0)
        self.assertAlmostEqual(poiseuille_pressure_gradient_pa_m(0.998, 1.0, 0.5), -31.936)

    def test_reference_metrics_align_pressure_gauge(self):
        _, config = small_model()
        geometry = config.case.geometry
        points = geometry.map(
            np.array([0.0, geometry.total_length()]), np.array([0.0, 0.0])
        )
        reference = ReferenceData(
            points, np.array([1.0, 1.0]), np.zeros(2), np.array([10.0, 5.0])
        )
        prediction = PredictionResult(
            np.array([1.0, 1.0]), np.zeros(2), np.array([110.0, 105.0])
        )
        metrics = reference_metrics(geometry, prediction, reference)
        self.assertAlmostEqual(metrics["relative_l2_velocity"], 0.0)
        self.assertAlmostEqual(metrics["normalized_pressure_error_after_gauge_alignment"], 0.0)
        self.assertAlmostEqual(metrics["relative_pressure_drop_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
