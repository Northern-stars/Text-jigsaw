from __future__ import annotations

import itertools

import pytest

from Manga109.reading_order_utils import ReadingOrderError, sort_bboxes_reading_order


def test_spread_order_respects_direction_and_is_input_order_invariant() -> None:
    items = [
        {"id": "left-top", "bbox": [0, 0, 100, 100]},
        {"id": "left-bottom", "bbox": [0, 200, 100, 300]},
        {"id": "right-top", "bbox": [200, 0, 300, 100]},
        {"id": "right-bottom", "bbox": [200, 200, 300, 300]},
    ]
    for rtl, expected in (
        (True, ["right-top", "right-bottom", "left-top", "left-bottom"]),
        (False, ["left-top", "left-bottom", "right-top", "right-bottom"]),
    ):
        outputs = {
            tuple(item["id"] for item in sort_bboxes_reading_order(list(order), 300, rtl, page_split_x=150))
            for order in itertools.permutations(items)
        }
        assert outputs == {tuple(expected)}


def test_crossing_physical_page_boundary_is_rejected() -> None:
    with pytest.raises(ReadingOrderError, match="physical page boundary"):
        sort_bboxes_reading_order([{"bbox": [90, 0, 210, 100]}], 100, page_split_x=150)


def test_inseparable_overlap_is_rejected() -> None:
    items = [{"bbox": [0, 0, 100, 100]}, {"bbox": [20, 20, 120, 120]}]
    with pytest.raises(ReadingOrderError, match="inseparable"):
        sort_bboxes_reading_order(items, 120)
