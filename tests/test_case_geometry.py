import unittest
from pathlib import Path

import numpy as np

from pinn_vanila import FlowCase, ProjectConfig
from pinn_vanila.errors import ConfigurationError, GeometryError


ROOT = Path(__file__).resolve().parents[1]


class CaseGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ProjectConfig.from_yaml(ROOT / "configs/pinn_vanila_config.yaml")
        cls.geometry = cls.config.case.geometry

    def test_supplied_physics(self):
        self.assertEqual(self.config.velocity_bounds, (1.0, 20.0))
        self.assertEqual(self.config.reynolds_bounds, (500.0, 10000.0))
        self.assertAlmostEqual(self.config.case.fluid.dynamic_viscosity, 0.998)
        self.assertAlmostEqual(self.geometry.minimum_jacobian_factor, 0.75)

    def test_map_inverse_roundtrip_for_angle_family(self):
        rng = np.random.default_rng(42)
        for angle in (15.0, 30.0, 60.0, 90.0):
            geometry = type(self.geometry)(
                **{**self.geometry.__dict__, "bend_angle": angle}
            )
            s = rng.uniform(0.0, geometry.total_length(), 200)
            eta = rng.uniform(-1.0, 1.0, 200)
            points = geometry.map(s, eta)
            actual_s, actual_eta, inside = geometry.inverse(points)
            self.assertTrue(np.all(inside))
            np.testing.assert_allclose(actual_s, s, atol=2e-12)
            np.testing.assert_allclose(actual_eta, eta, atol=2e-12)

    def test_rejects_invalid_points(self):
        _, _, inside = self.geometry.inverse(np.array([[0.0, 1.0], [-1.0, 0.0]]))
        self.assertFalse(np.any(inside))

    def test_fluid_consistency_validation(self):
        with self.assertRaises(ConfigurationError):
            FlowCase.from_yaml(
                ROOT / "configs/geometry_parametric.yaml",
                ROOT / "tests/fixtures/invalid_fluid.yaml",
            )


if __name__ == "__main__":
    unittest.main()
