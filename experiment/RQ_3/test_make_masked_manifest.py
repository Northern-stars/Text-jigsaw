from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiment.RQ_3.make_masked_manifest import mask_panel


class MaskedManifestTests(unittest.TestCase):
    def test_dialog_box_page_coordinates_map_to_padded_panel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            Image.new("RGB", (40, 40), "white").save(source)
            panel = {
                "padded_path": "source.png",
                "bbox": [10, 20, 30, 40],
                "raw_size": [20, 20],
                "padded_size": [40, 40],
                "pad": [10, 10, 10, 10],
                "dialog_bboxes": [[15, 25, 20, 30]],
            }
            output = root / "masked.png"

            mask_panel(panel, root, output, "black")

            image = Image.open(output).convert("RGB")
            self.assertEqual(image.getpixel((17, 17)), (0, 0, 0))
            self.assertEqual(image.getpixel((22, 22)), (255, 255, 255))
            self.assertEqual(image.getpixel((32, 22)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
