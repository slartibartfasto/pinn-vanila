import unittest
from pathlib import Path

from pinn_vanila.train import TrainingOptions, initialize_training, options_from_args, build_parser


ROOT = Path(__file__).resolve().parents[1]


class TrainingEntryTests(unittest.TestCase):
    def test_initialization_constructs_ready_session_without_training(self):
        config, model, trainer = initialize_training(
            TrainingOptions(
                config_path=ROOT / "configs/pinn_vanila_config.yaml", output="unused", seed=137
            )
        )
        self.assertEqual(config.raw["project"]["random_seed"], 137)
        self.assertEqual(model.network.config.seed, 137)
        self.assertEqual(trainer.device.type, "cpu")

    def test_quick_flag_applies_small_safe_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--output", "/tmp/model", "--quick"])
        options = options_from_args(args)
        self.assertEqual(options.max_steps, 1)
        self.assertEqual(options.interior_batch, 4)
        self.assertEqual(options.boundary_batch, 4)
        self.assertFalse(options.enable_refinement)
        self.assertFalse(options.enable_polishing)


if __name__ == "__main__":
    unittest.main()
