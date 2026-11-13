"""Deterministic, page-aware reading order shared by the comic builders.

Geometry is a heuristic, not human annotation. Unseparable/overlapping layouts
raise ReadingOrderError so builders can exclude them instead of inventing GT.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

READING_ORDER_VERSION = "2"
READING_ORDER_METHOD = "physical_pages_xy_cut"
WINDOW_POLICY = "continuous_source_image"


class ReadingOrderError(ValueError):
    """The geometry cannot establish an unambiguous supported reading order."""


def bbox_of(item: Any, bbox_key: str = "bbox") -> list[float]:
    bbox = item.get(bbox_key) if isinstance(item, dict) else getattr(item, bbox_key, None)
    if bbox is None and hasattr(item, "record"):
        bbox = getattr(item.record, bbox_key, None)
    if bbox is None or len(bbox) != 4:
        raise ReadingOrderError("each item must have a four-coordinate bbox")
    values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in values):
        raise ReadingOrderError("bbox coordinates must be finite")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ReadingOrderError("bbox must have positive width and height")
    return values


def physical_page_side(bbox: list[float], page_split_x: float | None) -> str:
    if page_split_x is None:
        return "single"
    if not math.isfinite(page_split_x) or page_split_x <= 0:
        raise ReadingOrderError("page_split_x must be positive and finite")
    if bbox[2] <= page_split_x:
        return "left"
    if bbox[0] >= page_split_x:
        return "right"
    raise ReadingOrderError("a panel crosses the physical page boundary")


def sort_bboxes_reading_order(
    items: list[Any],
    page_height: int,
    rtl: bool = True,
    bbox_key: str = "bbox",
    page_split_x: float | None = None,
) -> list[Any]:
    """Read physical pages first, then recursively split rows and columns.

    Horizontal gutters take precedence over vertical gutters, so ordinary
    grids read row by row. A tall panel beside stacked panels instead splits
    into columns. No mutable row centers or input-order-dependent clustering
    are used. Touching edges are accepted; actual overlapping layouts without
    a separating cut are rejected. page_split_x is explicit source layout
    metadata, never guessed from panel dimensions.
    """
    if not items:
        return []
    boxes = [bbox_of(item, bbox_key) for item in items]

    def components(indices: list[int], axis: int) -> list[list[int]]:
        ordered = sorted(indices, key=lambda i: (boxes[i][axis], tuple(boxes[i])))
        groups: list[list[int]] = []
        end = -math.inf
        for i in ordered:
            start, stop = boxes[i][axis], boxes[i][axis + 2]
            if not groups or start >= end:
                groups.append([i])
                end = stop
            else:
                groups[-1].append(i)
                end = max(end, stop)
        return groups

    def read(indices: list[int]) -> list[int]:
        if len(indices) <= 1:
            return indices
        rows = components(indices, 1)
        if len(rows) > 1:
            return [i for row in rows for i in read(row)]
        columns = components(indices, 0)
        if len(columns) > 1:
            if rtl:
                columns.reverse()
            return [i for column in columns for i in read(column)]
        raise ReadingOrderError("overlapping or inseparable panel layout needs review")

    pages: dict[str, list[int]] = defaultdict(list)
    for i, bbox in enumerate(boxes):
        pages[physical_page_side(bbox, page_split_x)].append(i)
    sides = ("single",) if page_split_x is None else (("right", "left") if rtl else ("left", "right"))
    return [items[i] for side in sides for i in read(pages[side])]


def describe_panel_sequence(
    items: list[Any],
    *,
    page_id: str,
    source_image_id: str,
    page_split_x: float | None,
    rtl: bool,
    sorted_geometrically: bool = True,
) -> list[dict[str, Any]]:
    """Keep physical page rank distinct from source index and window rank."""
    ranks: dict[str, int] = defaultdict(int)
    result = []
    for rank, item in enumerate(items):
        try:
            side = physical_page_side(bbox_of(item), page_split_x)
        except ReadingOrderError:
            if sorted_geometrically:
                raise
            side = "unresolved"
        result.append({
            "source_image_id": source_image_id,
            "spread_id": page_id,
            "physical_page_id": f"{page_id}:{side}",
            "reading_rank_in_page": ranks[side] if sorted_geometrically else None,
            "reading_rank_in_source": rank if sorted_geometrically else None,
            "reading_direction": "rtl" if rtl else "ltr",
            "page_split_x": page_split_x,
            "page_layout": "spread" if page_split_x is not None else "single",
            "reading_order_method": READING_ORDER_METHOD if sorted_geometrically else "source_order",
            "reading_order_version": READING_ORDER_VERSION,
            "reading_order_status": "heuristic" if sorted_geometrically else "unverified",
            "window_policy": WINDOW_POLICY,
        })
        ranks[side] += 1
    return result


def sort_prepared_panels_reading_order(
    prepared_panels: list[Any],
    rtl: bool = True,
    page_split_x: float | None = None,
    sort_panels: bool = True,
) -> list[Any]:
    """Order PreparedPanels and attach provenance without changing source IDs."""
    if not prepared_panels:
        return []
    first = prepared_panels[0].record
    ordered = sort_bboxes_reading_order(
        list(prepared_panels), first.page_size[1], rtl=rtl, page_split_x=page_split_x,
    ) if sort_panels else list(prepared_panels)
    metadata = describe_panel_sequence(
        ordered, page_id=first.page_id, source_image_id=first.source_image_path,
        page_split_x=page_split_x, rtl=rtl, sorted_geometrically=sort_panels,
    )
    for panel, fields in zip(ordered, metadata):
        record = panel.record
        fields["source_panel_id"] = getattr(
            record, "panel_id", f"{record.page_id}_panel_{record.panel_index_in_page:03d}"
        )
        fields["source_panel_index"] = record.panel_index_in_page
        record.reading_order_metadata = fields
        # Source indices remain the unchanged annotation/detection indices.
        if sort_panels:
            record.global_order = record.page_index * 10000 + fields["reading_rank_in_source"]
    return ordered


def sequence_provenance(panels: list[dict[str, Any]]) -> dict[str, Any]:
    """Sample metadata for a window, which may contain both physical pages."""
    first = panels[0]
    keys = (
        "source_image_id", "spread_id", "reading_direction", "page_split_x", "page_layout",
        "reading_order_method", "reading_order_version", "reading_order_status", "window_policy",
    )
    return {
        **{key: first[key] for key in keys if key in first},
        "physical_page_ids": sorted({p["physical_page_id"] for p in panels if "physical_page_id" in p}),
    }
