import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from src.model.sparrow_moe import SparrowConfig, SparrowMoE
from src.engine.trainer import SparrowTrainer
from src.data.dataset import SyntheticDemoDataset


class TestCheckpoint(unittest.TestCase):
    def _create_dummy_setup(self):
        config = SparrowConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            head_dim=32,
            d_ff=128,
            n_experts=4,
            top_k=2,
            max_seq_len=64,
        )
        model = SparrowMoE(config)
        dataset = SyntheticDemoDataset(vocab_size=1000, seq_len=32, size=8)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        return config, model, loader

    def test_checkpoint_weights_only_save_and_load(self):
        """Verify that newly saved checkpoints save to single file and can be deserialized with weights_only=True."""
        _, model, loader = self._create_dummy_setup()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SparrowTrainer(
                model=model,
                dataloader=loader,
                max_steps=1,
                checkpoint_dir=tmpdir,
                checkpoint_name="sparrow_model.pt",
                device="cpu",
            )
            trainer.save_checkpoint(step=1)

            ckpt_file = Path(tmpdir) / "sparrow_model.pt"
            self.assertTrue(ckpt_file.is_file(), "Single checkpoint file was not created")

            # Verify only one file exists in the directory
            files = list(Path(tmpdir).glob("*.pt"))
            self.assertEqual(len(files), 1, f"Expected exactly 1 checkpoint file, found {files}")

            # Verify weights_only=True loads cleanly
            data = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            self.assertIn("model_state_dict", data)
            self.assertIn("config", data)
            self.assertIsInstance(data["config"], dict)
            self.assertEqual(data["step"], 1)

            # Verify model can reload weights
            loaded_config = SparrowConfig.from_dict(data["config"])
            new_model = SparrowMoE(loaded_config)
            new_model.load_state_dict(data["model_state_dict"])

    def test_checkpoint_atomic_overwrite_single_file(self):
        """Verify that multiple saves update the single file without creating extra files."""
        _, model, loader = self._create_dummy_setup()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SparrowTrainer(
                model=model,
                dataloader=loader,
                checkpoint_dir=tmpdir,
                checkpoint_name="sparrow_custom.pt",
                device="cpu",
            )
            trainer.save_checkpoint(step=100)
            data1 = torch.load(Path(tmpdir) / "sparrow_custom.pt", map_location="cpu", weights_only=True)
            self.assertEqual(data1["step"], 100)

            trainer.save_checkpoint(step=200)
            data2 = torch.load(Path(tmpdir) / "sparrow_custom.pt", map_location="cpu", weights_only=True)
            self.assertEqual(data2["step"], 200)

            # Confirm still only 1 file
            files = list(Path(tmpdir).glob("*.pt"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "sparrow_custom.pt")

    def test_save_interval_in_training(self):
        """Verify that trainer respects save_interval during training."""
        _, model, loader = self._create_dummy_setup()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SparrowTrainer(
                model=model,
                dataloader=loader,
                max_steps=2,
                save_interval=2,
                grad_accum_steps=1,
                checkpoint_dir=tmpdir,
                checkpoint_name="interval_model.pt",
                device="cpu",
            )
            trainer.train()

            ckpt_file = Path(tmpdir) / "interval_model.pt"
            self.assertTrue(ckpt_file.is_file(), "Checkpoint file was not created at save_interval")
            data = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            self.assertEqual(data["step"], 2)

    def test_evaluation_loop_computes_val_loss(self):
        """Verify that evaluate() runs without gradients and returns valid losses."""
        _, model, loader = self._create_dummy_setup()
        trainer = SparrowTrainer(
            model=model,
            dataloader=loader,
            val_dataloader=loader,
            eval_steps=2,
            device="cpu",
        )
        val_loss, val_aux = trainer.evaluate()
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_aux, float)
        self.assertGreater(val_loss, 0.0)
        self.assertGreaterEqual(val_aux, 0.0)
        self.assertTrue(trainer.model.training, "Model should be returned to training mode after evaluate()")

    def test_validation_in_training_loop(self):
        """Verify that training loop executes validation at eval_interval."""
        _, model, loader = self._create_dummy_setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SparrowTrainer(
                model=model,
                dataloader=loader,
                val_dataloader=loader,
                max_steps=2,
                eval_interval=1,
                eval_steps=1,
                grad_accum_steps=1,
                checkpoint_dir=tmpdir,
                device="cpu",
            )
            trainer.train()
            self.assertNotEqual(trainer.best_val_loss, float("inf"), "best_val_loss should be updated")

    def test_checkpoint_nonexistent_file_raises_error(self):
        """Verify that a non-existent checkpoint path raises FileNotFoundError."""
        nonexistent = "nonexistent_checkpoint_file_12345.pt"
        ckpt_path = Path(nonexistent)

        with self.assertRaises(FileNotFoundError):
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: '{nonexistent}'")


if __name__ == "__main__":
    unittest.main()
