"""Shared manga reading-order geometry helpers for panel-ordering datasets."""

from __future__ import annotations

from typing import Any, TypeVar


def sort_bboxes_reading_order(
    items: list[Any],
    page_height: int,
    rtl: bool = True,
    bbox_key: str = "bbox",
) -> list[Any]:
    """Sort panel-like objects by manga reading order using bbox geometry.

    Groups items into horizontal bands by y-center, sorts rows top-to-bottom,
    then sorts within each row right-to-left for manga (rtl=True).
    """
    if not items:
        return []

    def bbox_of(item: Any) -> list[float]:
        bbox = getattr(item, bbox_key, None)
        if bbox is None and isinstance(item, dict):
            bbox = item.get(bbox_key)
        if bbox is None and hasattr(item, "record"):
            bbox = getattr(item.record, "bbox", None)
        if bbox is None:
            raise ValueError(f"item has no bbox field: {item!r}")
        return [float(value) for value in bbox]

    bboxes = [bbox_of(item) for item in items]
    y_centers = [(b[1] + b[3]) / 2.0 for b in bboxes]
    xmins = [b[0] for b in bboxes]
    heights = [b[3] - b[1] for b in bboxes]
    avg_h = sum(heights) / len(heights) if heights else page_height / 4
    band_threshold = max(avg_h * 0.5, 10.0)

    rows: list[list[int]] = []
    row_centers: list[float] = []
    for index, y_center in enumerate(y_centers):
        assigned = False
        for row_idx, row_center in enumerate(row_centers):
            if abs(y_center - row_center) <= band_threshold:
                rows[row_idx].append(index)
                row_centers[row_idx] = sum(
                    y_centers[j] for j in rows[row_idx]
                ) / len(rows[row_idx])
                assigned = True
                break
        if not assigned:
            rows.append([index])
            row_centers.append(y_center)

    row_order = sorted(range(len(rows)), key=lambda r: row_centers[r])

    sorted_items: list[Any] = []
    for row_idx in row_order:
        row_indices = rows[row_idx]
        if rtl:
            row_indices.sort(key=lambda i: -xmins[i])
        else:
            row_indices.sort(key=lambda i: xmins[i])
        sorted_items.extend(items[i] for i in row_indices)
    return sorted_items


def sort_prepared_panels_reading_order(
    prepared_panels: list[Any],
    rtl: bool = True,
) -> list[Any]:
    """Sort PreparedPanel objects by manga reading order using record.bbox."""
    if not prepared_panels:
        return []
    page_height = 0
    for panel in prepared_panels:
        page_size = getattr(panel, "record", None)
        if page_size is not None and getattr(page_size, "page_size", None):
            page_height = int(page_size.page_size[1])
            break
    if page_height <= 0:
        page_height = 1000
    return sort_bboxes_reading_order(
        list(prepared_panels),
        page_height=page_height,
        rtl=rtl,
        bbox_key="bbox",
    )
