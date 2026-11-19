"""Panel ordering adapted from Magi v1 (The Manga Whisperer, CVPR 2024).

Source: ragavsachdeva/magi, Hugging Face utils.py, revision
15ad498ad45093ad3b833d625728220c6309ad99 (sort_panels and geometry helpers).
https://huggingface.co/ragavsachdeva/magi/blob/15ad498ad45093ad3b833d625728220c6309ad99/utils.py
Authors: Ragav Sachdeva and Andrew Zisserman. Upstream specifies academic
research use; see docs/manga109_manga_whisper_order.md for attribution.

Preserves the upstream relation graph, erosion, cycle removal and topological
sort. Axis-aligned rectangle operations replace Shapely; bounded erosion
raises ReadingOrderError on nonconvergence instead of hanging indefinitely.
The adapter adds explicit physical-page ordering, an LTR mirror option and
canonical input ordering to make ambiguous tie resolution reproducible.
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any

import networkx as nx
import numpy as np

from Manga109.reading_order_utils import ReadingOrderError, bbox_of, physical_page_side

READING_ORDER_METHOD = "manga_whisper_v1"
READING_ORDER_VERSION = "3"
UPSTREAM_REVISION = "15ad498ad45093ad3b833d625728220c6309ad99"
UPSTREAM_SOURCE = f"https://huggingface.co/ragavsachdeva/magi/blob/{UPSTREAM_REVISION}/utils.py"
MAX_EROSIONS = 2048


def erode_rectangle(bbox: list[float], erosion_factor: float) -> list[float]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        raise ReadingOrderError("MangaWhisper erosion collapsed a panel")
    cx, cy = x1 + w / 2, y1 + h / 2
    if w < h:
        erosion_factor_width = erosion_factor * (w / h)
        erosion_factor_height = erosion_factor
    else:
        erosion_factor_width = erosion_factor
        erosion_factor_height = erosion_factor * (h / w)
    w = w - w * erosion_factor_width
    h = h - h * erosion_factor_height
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def merge_overlapping_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges, key=lambda interval: interval[0]):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def intersects(a: list[float], b: list[float]) -> bool:
    # Shapely's intersects includes touching boundaries.
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def get_distance(a: list[float], b: list[float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.sqrt(dx * dx + dy * dy)


def use_cuts_to_determine_edge_from_a_to_b(a: int, b: int, rects: list[list[float]]) -> bool:
    rects = deepcopy(rects)
    for _ in range(MAX_EROSIONS):
        region = [min(rects[a][0], rects[b][0]), min(rects[a][1], rects[b][1]),
                  max(rects[a][2], rects[b][2]), max(rects[a][3], rects[b][3])]
        indices = [i for i, rect in enumerate(rects) if intersects(rect, region)]
        # Upstream tries horizontal cuts before right-to-left vertical cuts.
        for axis in (1, 0):
            ranges = merge_overlapping_ranges([(rects[i][axis], rects[i][axis + 2]) for i in indices])
            if axis == 0:
                ranges.reverse()
            split = {}
            for rank, (start, end) in enumerate(ranges):
                for i in indices:
                    if start <= rects[i][axis] <= rects[i][axis + 2] <= end:
                        split[i] = rank
            if a not in split or b not in split:
                raise ReadingOrderError("MangaWhisper cut erosion lost a panel")
            if split[a] != split[b]:
                return split[a] < split[b]
        rects = [erode_rectangle(rect, 0.05) for rect in rects]
    raise ReadingOrderError("MangaWhisper cuts did not converge")


def is_there_a_directed_edge(a: int, b: int, rects: list[list[float]]) -> bool:
    rect_a, rect_b = rects[a], rects[b]
    center_a = [rect_a[0] + (rect_a[2] - rect_a[0]) / 2, rect_a[1] + (rect_a[3] - rect_a[1]) / 2]
    center_b = [rect_b[0] + (rect_b[2] - rect_b[0]) / 2, rect_b[1] + (rect_b[3] - rect_b[1]) / 2]
    if np.allclose(np.array(center_a), np.array(center_b)):
        return (rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1]) > (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1])
    copy_a, copy_b = rect_a[:], rect_b[:]
    for _ in range(MAX_EROSIONS):
        above_ab = copy_a[3] < copy_b[1]
        above_ba = copy_b[3] < copy_a[1]
        left_ab = copy_a[2] < copy_b[0]
        left_ba = copy_b[2] < copy_a[0]
        if above_ab and not left_ab:
            return True
        if above_ba and not left_ba:
            return False
        if left_ba and not above_ba:
            return True
        if left_ab and not above_ab:
            return False
        if (above_ba and left_ba) or (above_ab and left_ab):
            return use_cuts_to_determine_edge_from_a_to_b(a, b, rects)
        copy_a = erode_rectangle(copy_a, 0.05)
        copy_b = erode_rectangle(copy_b, 0.05)
    raise ReadingOrderError("MangaWhisper pairwise erosion did not converge")


def sort_panels(rects: list[list[float]]) -> list[int]:
    """Return RTL reading-order input indices, following Magi v1 sort_panels."""
    rects = [erode_rectangle(rect, 0.05) for rect in rects]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(rects)))
    for i in range(len(rects)):
        for j in range(len(rects)):
            if i == j:
                continue
            if is_there_a_directed_edge(i, j, rects):
                graph.add_edge(i, j, weight=get_distance(rects[i], rects[j]))
            else:
                graph.add_edge(j, i, weight=get_distance(rects[i], rects[j]))
    while True:
        cycles = sorted(nx.simple_cycles(graph))
        cycles = [cycle for cycle in cycles if len(cycle) > 1]
        if not cycles:
            break
        cycle = cycles[0]
        edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        graph.remove_edge(*max(edges, key=lambda edge: graph.edges[edge]["weight"]))
    return list(nx.topological_sort(graph))


def sort_manga_whisper_reading_order(
    items: list[Any], *, rtl: bool = True, page_split_x: float | None = None,
) -> list[Any]:
    """Order complete physical pages with the MangaWhisper panel algorithm.

    A panel crossing the spread seam still requires review. Overlap within a
    physical page is handled by Magi erosion and no longer rejected by XY-cut.
    LTR is a project extension implemented by mirroring x coordinates.
    """
    pages: dict[str, list[tuple[Any, list[float]]]] = defaultdict(list)
    for item in items:
        bbox = bbox_of(item)
        side = physical_page_side(bbox, page_split_x)
        box = bbox if rtl else [-bbox[2], bbox[1], -bbox[0], bbox[3]]
        pages[side].append((item, box))
    sides = ("single",) if page_split_x is None else (("right", "left") if rtl else ("left", "right"))
    result = []
    for side in sides:
        # Deterministic ties/cycle removal irrespective of XML annotation order.
        def canonical_key(pair: tuple[Any, list[float]]) -> tuple:
            item, box = pair
            frame = getattr(item, "frame", item)
            source_id = str(frame.get("id", "")) if isinstance(frame, dict) else ""
            return tuple(box), source_id

        panels = sorted(pages[side], key=canonical_key)
        result.extend(panels[i][0] for i in sort_panels([box for _, box in panels]))
    return result
