"""Tests for image discovery, validation, and structured invoice data.

Author: Miguel Medina Cantos
"""

import base64
from pathlib import Path
import tempfile
import unittest

from src.agents.extractor.agent import (
    ImageValidationError,
    InvoiceData,
    discover_images,
    sha256_file,
    validate_image,
)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


class ExtractorTests(unittest.TestCase):
    """Verify scalable discovery and early invalid-file rejection."""

    def test_discovery_returns_only_supported_images_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            (input_dir / "b.JPG").write_bytes(PNG_BYTES)
            (input_dir / "a.png").write_bytes(PNG_BYTES)
            (input_dir / "notes.txt").write_text("ignored", encoding="utf-8")
            self.assertEqual(
                [path.name for path in discover_images(input_dir)],
                ["a.png", "b.JPG"],
            )

    def test_missing_discovery_directory_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                list(discover_images(Path(directory) / "missing"))

    def test_invalid_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "invalid.jpg"
            image_path.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ImageValidationError, "Invalid image"):
                validate_image(image_path, max_bytes=1024)

    def test_valid_image_is_hashed_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "invoice.png"
            image_path.write_bytes(PNG_BYTES)
            self.assertEqual(validate_image(image_path, max_bytes=1024), "image/png")
            self.assertEqual(len(sha256_file(image_path)), 64)

    def test_invoice_search_text_contains_only_available_fields(self) -> None:
        invoice = InvoiceData(supplier_name="Acme", total="12.30", currency="EUR")
        self.assertEqual(
            invoice.search_text,
            "Supplier: Acme\nCurrency: EUR\nTotal: 12.30",
        )


if __name__ == "__main__":
    unittest.main()
