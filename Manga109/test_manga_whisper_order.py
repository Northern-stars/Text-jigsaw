"""Regression cases evaluated against the pinned, original Magi v1 code."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from PIL import Image

from Manga109.create_ordering import create_manga109_ordering, sort_frames_reading_order
from Manga109.manga_whisper_order import (
    READING_ORDER_METHOD, READING_ORDER_VERSION, UPSTREAM_REVISION,
    sort_manga_whisper_reading_order, sort_panels,
)
from Manga109.reading_order_utils import ReadingOrderError

FIXTURES = json.loads(
    (Path(__file__).parent / "tests/fixtures/magi_v1_panel_orders.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURES.values(), ids=FIXTURES.keys())
def test_panel_orders_match_pinned_upstream(case):
    assert sort_panels(case["boxes"]) == case["order"]


def test_overlap_resolution_is_deterministic_and_ltr_is_a_mirror():
    items = [{"id": str(i), "bbox": bbox} for i, bbox in enumerate(FIXTURES["overlap"]["boxes"])]
    for permutation in itertools.permutations(items):
        ordered = sort_manga_whisper_reading_order(list(permutation))
        assert [item["id"] for item in ordered] == ["1", "0", "3", "2"]
        mirrored = [{**item, "bbox": [-item["bbox"][2], item["bbox"][1],
                                      -item["bbox"][0], item["bbox"][3]]} for item in permutation]
        assert [item["id"] for item in sort_manga_whisper_reading_order(mirrored, rtl=False)] == ["1", "0", "3", "2"]


def test_complete_physical_pages_are_read_before_the_next_page():
    frames = [dict(id=side + str(row), xmin=x, ymin=row * 100, xmax=x + 80, ymax=row * 100 + 90)
              for row in range(2) for side, x in (("L", 10), ("R", 210))]
    for rtl, expected in ((True, ["R0", "R1", "L0", "L1"]),
                          (False, ["L0", "L1", "R0", "R1"])):
        for permutation in itertools.permutations(frames):
            assert [f["id"] for f in sort_frames_reading_order(list(permutation), 200, rtl, 200)] == expected


def test_seam_crossing_and_invalid_boxes_still_require_review():
    with pytest.raises(ReadingOrderError, match="physical page boundary"):
        sort_manga_whisper_reading_order([{"bbox": [90, 0, 110, 50]}], page_split_x=100)
    with pytest.raises(ReadingOrderError, match="positive width"):
        sort_manga_whisper_reading_order([{"bbox": [0, 0, 0, 50]}])
    assert sort_manga_whisper_reading_order([]) == []


def test_generated_windows_preserve_source_identity_and_record_magi_provenance(tmp_path):
    source = tmp_path / "source"
    (source / "annotations").mkdir(parents=True)
    (source / "images" / "Book").mkdir(parents=True)
    # Each page has three overlapping rows. The former XY-cut rejected them.
    frame_tags = []
    for side, x in (("L", 10), ("R", 210)):
        for row in range(3):
            frame_tags.append(f'<frame id="{side}{row}" xmin="{x}" ymin="{row*60}" '
                              f'xmax="{x+80}" ymax="{row*60+70}"/>')
    (source / "annotations/Book.xml").write_text(
        '<book title="Book"><pages><page index="0" width="400" height="200">'
        + "".join(frame_tags) + '</page></pages></book>'
    )
    Image.new("RGB", (400, 200), "white").save(source / "images/Book/000.jpg")
    output = tmp_path / "output"
    stats = create_manga109_ordering(
        source, output, panel_count=4, train_puzzles=3, stride=1,
        target_width=32, target_height=32,
    )
    assert stats["puzzle_count"] == 3
    assert stats["skipped_reading_order_pages"] == 0
    expected_ids = ["R0", "R1", "R2", "L0", "L1", "L2"]
    seen = {}
    for start in range(3):
        sample = json.loads((output / f"{start:06d}/sample.json").read_text())
        ordered = [sample["panels"][slot] for slot in sample["target_order"]]
        assert [p["source_panel_id"].rsplit("_frame_", 1)[1] for p in ordered] == expected_ids[start:start+4]
        assert sample["reading_order_method"] == READING_ORDER_METHOD
        assert sample["reading_order_version"] == READING_ORDER_VERSION
        assert sample["reading_order_source_revision"] == UPSTREAM_REVISION
        for panel in ordered:
            assert (output / panel["padded_path"]).is_file()
            signature = (panel["source_panel_index"], panel["global_order"], panel["reading_rank_in_page"])
            assert seen.setdefault(panel["source_panel_id"], signature) == signature
            assert panel["reading_order_version"] == READING_ORDER_VERSION
    assert json.loads((output / "meta.json").read_text())["reading_order_method"] == READING_ORDER_METHOD
