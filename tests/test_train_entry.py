import unittest
import json
import tempfile
from pathlib import Path

from pinn_vanila.train import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    TrainingOptions,
    build_parser,
    initialize_training,
    options_from_args,
    run_training,
)


ROOT = Path(__file__).resolve().parents[1]


class TrainingEntryTests(unittest.TestCase):
    def test_initialization_constructs_ready_session_without_training(self):
        config, model, trainer = initialize_training(
            TrainingOptions(
                config_path=ROOT / "configs/pinn_vanila_config.yaml",
                output="unused",
                seed=137,
                device="cpu",
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
        self.assertIsNone(options.log_every)
        self.assertIsNone(options.field_every)

    def test_direct_run_has_a_default_output_directory(self):
        args = build_parser().parse_args([])
        self.assertEqual(Path(args.output), DEFAULT_OUTPUT)
        self.assertEqual(Path(args.config), DEFAULT_CONFIG)

    def test_cuda_is_required_unless_cpu_is_explicit(self):
        from pinn_vanila.train import resolve_device

        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_cpu_debug_run_writes_metrics_logs_and_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            report = run_training(
                TrainingOptions(
                    config_path=ROOT / "configs/pinn_vanila_config.yaml",
                    output=output,
                    device="cpu",
                    max_steps=1,
                    interior_batch=4,
                    boundary_batch=4,
                    enable_refinement=False,
                    enable_polishing=False,
                    log_every=1,
                    field_every=1,
                )
            )
            self.assertEqual(report["steps"], 1)
            self.assertTrue((output / "training.log").is_file())
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "training_losses.png").is_file())
            self.assertTrue((output / "fields" / "fields_step_0000001.png").is_file())
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertIn("continuity", metrics["latest"])
            self.assertTrue(metrics["physics_diagnostics"])


if __name__ == "__main__":
    unittest.main()
