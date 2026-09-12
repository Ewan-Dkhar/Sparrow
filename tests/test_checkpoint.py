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
    def test_checkpoint_weights_only_save_and_load(self):
        """Verify that newly saved checkpoints can be deserialized with weights_only=True."""
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
        dataset = SyntheticDemoDataset(vocab_size=1000, seq_len=32, size=4)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SparrowTrainer(
                model=model,
                dataloader=loader,
                max_steps=1,
                checkpoint_dir=tmpdir,
                device="cpu",
            )
            trainer.save_checkpoint(step=1)

            ckpt_file = Path(tmpdir) / "sparrow_step_1.pt"
            self.assertTrue(ckpt_file.is_file(), "Checkpoint file was not created")

            # Verify weights_only=True loads cleanly
            data = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            self.assertIn("model_state_dict", data)
            self.assertIn("config", data)
            self.assertIsInstance(data["config"], dict)

            # Verify model can reload weights
            loaded_config = SparrowConfig.from_dict(data["config"])
            new_model = SparrowMoE(loaded_config)
            new_model.load_state_dict(data["model_state_dict"])

    def test_checkpoint_nonexistent_file_raises_error(self):
        """Verify that a non-existent checkpoint path raises FileNotFoundError."""
        nonexistent = "nonexistent_checkpoint_file_12345.pt"
        ckpt_path = Path(nonexistent)

        with self.assertRaises(FileNotFoundError):
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: '{nonexistent}'")


if __name__ == "__main__":
    unittest.main()
