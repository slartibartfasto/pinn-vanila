import unittest
from pathlib import Path

import torch

from pinn_vanila import PINN, ProjectConfig
from pinn_vanila.network import GeometryConditionedPINN, NetworkConfig
from pinn_vanila.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]


class TrainingSmokeTests(unittest.TestCase):
    def test_one_physics_step(self):
        config = ProjectConfig.from_yaml(ROOT / "configs/pinn_vanila_config.yaml")
        network = GeometryConditionedPINN(
            config.case.geometry,
            config.reynolds_bounds,
            NetworkConfig(
                hidden_layers=2,
                hidden_width=12,
                fourier_enabled=False,
                precision="float64",
                turbulence=config.turbulence,
            ),
        )
        model = PINN(network, config.case, config.velocity_bounds)
        trainer = Trainer(model, config, "cpu")
        result = trainer.train(
            max_steps=1,
            interior_batch=4,
            boundary_batch=4,
            enable_refinement=False,
            enable_polishing=False,
        )
        self.assertEqual(result.steps, 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.final_losses.values()))


if __name__ == "__main__":
    unittest.main()
