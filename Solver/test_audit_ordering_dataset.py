"""Check displayed pixels, dialogue and labels through the real loader/shuffle."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from Solver.audit_ordering_dataset import export_audit, make_record, parse_args
from Solver.data.panel_ordering_dataset import MangaZeroPanelOrderingDataset


class OrderingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Raw input slots deliberately disagree with canonical reading ranks.
        self.ranks = [3, 0, 5, 2, 1, 4]
        panels = []
        for slot, rank in enumerate(self.ranks):
            path = self.root / f"panel_{slot}.png"
            Image.new("RGB", (40, 28), (rank * 40, 0, 0)).save(path)
            panels.append({"padded_path": path.name, "panel_index": rank,
                           "dialog_text": f"台词 {rank}", "source_text": f"原文 {rank}",
                           "translated_text": f"translation {rank}",
                           "description": "</script><img src=x onerror=alert(1)>"})
        sample = {"sequence_id": "test_puzzle", "manga_id": "test_work", "panel_count": 6,
                  "target_order": [self.ranks.index(i) for i in range(6)], "panels": panels}
        (self.root / "train.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
        other = dict(sample, sequence_id="validation_only")
        (self.root / "val.jsonl").write_text(json.dumps(other) + "\n", encoding="utf-8")
        self.dataset = MangaZeroPanelOrderingDataset(self.root, image_size=(16, 12),
                                                     use_layout=False, given_endpoints=True)

    def record(self, mode="train", originals=True):
        return make_record(self.dataset, 0, seed=42, shuffle_mode=mode,
                           include_original_images=originals)

    def test_pixels_dialogue_provenance_and_gt_stay_aligned(self):
        record = self.record()
        for panel in record["panels"]:
            rank = self.ranks[panel["source_index"]]
            image = Image.open(io.BytesIO(base64.b64decode(panel["image_uri"].split(",")[1])))
            self.assertEqual(image.size, (16, 12))
            self.assertEqual(image.getpixel((0, 0)), (rank * 40, 0, 0))
            self.assertEqual(panel["model_dialog_text"], f"台词 {rank}")
            self.assertEqual(panel["texts"]["source_text"], f"原文 {rank}")
            self.assertEqual(panel["metadata"]["panel_index"], rank)
            original = Image.open(io.BytesIO(base64.b64decode(panel["original_uri"].split(",")[1])))
            self.assertEqual(original.size, (40, 28))
        canonical = [record["panels"][slot]["metadata"]["panel_index"] for slot in record["target_order"]]
        self.assertEqual(canonical, list(range(6)))
        self.assertEqual(record["panels"][0]["metadata"]["panel_index"], 0)
        self.assertEqual(record["panels"][-1]["metadata"]["panel_index"], 5)

    def test_deterministic_shuffle_preserves_external_rng_and_dataset_mode(self):
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        first = self.record()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(first, self.record())
        unshuffled = self.record(mode="dataset", originals=False)
        self.assertEqual(unshuffled["target_order"], self.dataset[0]["target_order"].tolist())
        self.assertTrue(all(p["original_uri"] is None for p in unshuffled["panels"]))

    def test_export_reads_requested_split_embeds_safe_text_and_records_mapping(self):
        args = parse_args(["--dataset-dir", str(self.root), "--split", "val",
                           "--sample-indices", "0", "--image-width", "16", "--image-height", "12",
                           "--output-dir", str(self.root / "report")])
        payload = export_audit(args)
        self.assertEqual(payload["samples"][0]["sequence_id"], "validation_only")
        html = (args.output_dir / "audit.html").read_text(encoding="utf-8")
        self.assertNotIn("__AUDIT_PAYLOAD__", html)
        self.assertNotIn("</script><img src=x", html)
        embedded = html.split('<script id="audit-data" type="application/json">')[1].split("</script>")[0]
        self.assertEqual(json.loads(embedded), payload)
        manifest = json.loads((args.output_dir / "audit_samples.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["samples"][0]["target_order"], payload["samples"][0]["target_order"])
        self.assertNotIn("image_uri", manifest["samples"][0]["panels"][0])
        self.assertTrue((args.output_dir / "startup_config.json").is_file())

    def test_selection_rejects_duplicate_and_out_of_range_indices(self):
        for selection in (["0", "0"], ["1"], ["-1"]):
            args = parse_args(["--dataset-dir", str(self.root), "--sample-indices", *selection])
            with self.assertRaisesRegex(ValueError, "unique indices"):
                export_audit(args)


if __name__ == "__main__":
    unittest.main()
