"""Tests for centralized application configuration.

Author: Miguel Medina Cantos
"""

from pathlib import Path
import tempfile
import unittest

from config.settings import ConfigurationError, Settings


class SettingsTests(unittest.TestCase):
    """Verify path and vector configuration validation."""

    def test_missing_input_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                app_root=root,
                input_dir=root / "missing",
                output_dir=root / "output",
            )
            with self.assertRaisesRegex(ConfigurationError, "does not exist"):
                settings.validate()

    def test_firestore_dimension_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            settings = Settings(
                app_root=root,
                input_dir=input_dir,
                output_dir=root / "output",
                embedding_dimension=2049,
            )
            with self.assertRaisesRegex(ConfigurationError, "2048"):
                settings.validate()

    def test_dot_product_requires_normalized_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            settings = Settings(
                app_root=root,
                input_dir=input_dir,
                output_dir=root / "output",
                distance_measure="DOT_PRODUCT",
                normalize_embeddings=False,
            )
            with self.assertRaisesRegex(ConfigurationError, "unit-normalized"):
                settings.validate()

    def test_relative_environment_paths_are_rooted_in_application(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings.from_env(
                {
                    "AGENT_LEDGER_ROOT": str(root),
                    "AGENT_LEDGER_INPUT_DIR": "dataset",
                    "AGENT_LEDGER_OUTPUT_DIR": "results",
                }
            )
            self.assertEqual(settings.input_dir, (root / "dataset").resolve())
            self.assertEqual(settings.output_dir, (root / "results").resolve())


if __name__ == "__main__":
    unittest.main()
