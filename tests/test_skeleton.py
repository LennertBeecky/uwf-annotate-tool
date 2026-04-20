"""Tests for skeleton tracing — especially the no-double-counting property."""

from __future__ import annotations

import numpy as np

from uwf_zonal_extraction.skeleton import trace_segments


def _canonical_edge_set(segments):
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for seg in segments:
        pts = [tuple(p) for p in seg]
        for a, b in zip(pts[:-1], pts[1:]):
            e = (a, b) if a <= b else (b, a)
            assert e not in edges, f"duplicate edge {e}"
            edges.add(e)
    return edges


def test_single_straight_line_not_doubled():
    skel = np.zeros((20, 20), dtype=bool)
    skel[10, 2:18] = True
    segs = trace_segments(skel, min_length_px=3)
    # One arc, traced once. Length 16.
    assert len(segs) == 1
    assert len(segs[0]) == 16
    _canonical_edge_set(segs)


def test_y_junction_three_branches_once_each():
    # Y-shape: horizontal trunk, two angled branches diverging at (10, 10)
    skel = np.zeros((25, 25), dtype=bool)
    skel[10, 2:11] = True          # horizontal trunk up to junction
    for i in range(1, 8):          # upper branch
        skel[10 - i, 10 + i] = True
    for i in range(1, 8):          # lower branch
        skel[10 + i, 10 + i] = True
    segs = trace_segments(skel, min_length_px=3)
    # Exactly 3 arcs from the Y
    assert len(segs) == 3
    # Each pixel-edge appears at most once in the union
    _canonical_edge_set(segs)


def test_between_two_junctions_not_doubled():
    # Two Y-junctions connected by a trunk. Previously bugged code would
    # walk the trunk twice (once from each junction).
    skel = np.zeros((30, 40), dtype=bool)
    # Left Y at x=10
    skel[15, 2:11] = True
    for i in range(1, 6):
        skel[15 - i, 10 - i] = True
        skel[15 + i, 10 - i] = True
    # Trunk between the Ys
    skel[15, 10:30] = True
    # Right Y at x=30
    for i in range(1, 6):
        skel[15 - i, 30 + i] = True
        skel[15 + i, 30 + i] = True
    segs = trace_segments(skel, min_length_px=3)
    # Expect: 4 outer branches + 1 trunk = 5 (but left Y has the leftmost
    # branch too, so 6 total)
    # Counting more carefully:
    #   Left Y: 3 branches (leftmost horizontal, up-left, down-left)
    #   Trunk: 1
    #   Right Y: 2 branches (up-right, down-right)
    # Total: 6
    assert len(segs) == 6
    _canonical_edge_set(segs)


def test_isolated_loop_traced_once():
    # A closed loop with no endpoints. Previous code's second pass would
    # have walked it, but the new code should still trace it once.
    skel = np.zeros((30, 30), dtype=bool)
    center = (15, 15)
    for theta in np.linspace(0, 2 * np.pi, 100, endpoint=False):
        y = int(round(center[0] + 8 * np.sin(theta)))
        x = int(round(center[1] + 8 * np.cos(theta)))
        skel[y, x] = True
    segs = trace_segments(skel, min_length_px=5)
    # At least one segment, no duplicate edges
    assert len(segs) >= 1
    _canonical_edge_set(segs)
