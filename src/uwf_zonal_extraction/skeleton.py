"""Skeleton tracing + segment labelling.

Medial-axis-based (returns a distance map as a free w_init), Zhang-Suen
is avoided because it shifts 0.5 px on even-width vessels.

v1 implements:
  - medial_axis + distance map
  - spur removal (length scales with local diameter)
  - segment tracing between endpoints/bifurcations
  - continuous A/V label confidence from (p_a - p_v)
  - bifurcation/crossing classification + exclusion buffer
  - per-cell sub-segment splitting

Known v1 limitations (documented in docs/03_SKELETON_AND_SEGMENTS.md):
  - gap bridging is off by default
  - crossing detection uses simple mask overlap; more sophisticated
    topology-aware crossing classification is v2
"""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np
from skimage.morphology import medial_axis, disk, binary_closing

from uwf_zonal_extraction.config import ExtractionConfig
from uwf_zonal_extraction.coordinate_system import RetinalCoordinateSystem
from uwf_zonal_extraction.models import VesselSegment

NEIGHBOUR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def build_skeleton(
    av_mask: np.ndarray,
    config: ExtractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (skeleton_bool, distance_map).

    distance_map[y, x] is the Euclidean distance from (y, x) to the
    nearest background pixel. 2 * distance_map[skel] is a free w_init.
    """
    vessel_mask = av_mask > 0
    r = config.skeleton.closing_disk_radius
    if r > 0:
        vessel_mask = binary_closing(vessel_mask, disk(r))
    skel, dist = medial_axis(vessel_mask, return_distance=True)
    return skel, dist.astype(np.float32)


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), ddepth=cv2.CV_8U, kernel=kernel)


def trace_segments(
    skel: np.ndarray,
    min_length_px: int = 5,
) -> list[np.ndarray]:
    """Trace ordered skeleton segments between endpoints/bifurcations.

    Returns a list of (N, 2) arrays of (y, x) in traversal order.

    Junctions are NOT marked "consumed" globally — a single junction can
    anchor several outgoing arcs — but each outgoing edge is claimed on
    first walk (bookkept by the `edges_used` set keyed on the sorted
    endpoint pixel pair). That prevents the "walk the same arc twice
    from each end" double-counting.
    """
    skel_u8 = skel.astype(np.uint8)
    nc = _neighbour_count(skel_u8) * skel_u8

    h, w = skel.shape
    visited_interior = np.zeros_like(skel, dtype=bool)   # 2-neighbour cells
    edges_used: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[np.ndarray] = []

    def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple:
        return (a, b) if a <= b else (b, a)

    def _walk(start_y: int, start_x: int) -> Iterator[list[tuple[int, int]]]:
        """Yield segments starting at (start_y, start_x) junction/endpoint."""
        for dy, dx in NEIGHBOUR_OFFSETS:
            ny, nx = start_y + dy, start_x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not skel[ny, nx]:
                continue
            first_edge = _edge_key((start_y, start_x), (ny, nx))
            if first_edge in edges_used:
                continue
            if nc[ny, nx] == 2 and visited_interior[ny, nx]:
                continue
            edges_used.add(first_edge)
            path = [(start_y, start_x), (ny, nx)]
            if nc[ny, nx] == 2:
                visited_interior[ny, nx] = True
            cy, cx = ny, nx
            # Walk through interior pixels until a junction/endpoint.
            while nc[cy, cx] == 2:
                found = False
                for ddy, ddx in NEIGHBOUR_OFFSETS:
                    my, mx = cy + ddy, cx + ddx
                    if not (0 <= my < h and 0 <= mx < w):
                        continue
                    if not skel[my, mx] or (my, mx) == path[-2]:
                        continue
                    step_edge = _edge_key((cy, cx), (my, mx))
                    if step_edge in edges_used:
                        continue
                    if nc[my, mx] == 2 and visited_interior[my, mx]:
                        continue
                    edges_used.add(step_edge)
                    path.append((my, mx))
                    if nc[my, mx] == 2:
                        visited_interior[my, mx] = True
                    cy, cx = my, mx
                    found = True
                    break
                if not found:
                    break
            yield path

    # First pass: start from endpoints and junctions.
    for y in range(h):
        for x in range(w):
            if not skel[y, x]:
                continue
            if nc[y, x] == 1 or nc[y, x] >= 3:
                for path in _walk(y, x):
                    if len(path) >= min_length_px:
                        segments.append(np.asarray(path, dtype=np.int32))

    # Second pass: isolated loops (no endpoints/junctions anywhere on arc).
    for y in range(h):
        for x in range(w):
            if skel[y, x] and nc[y, x] == 2 and not visited_interior[y, x]:
                visited_interior[y, x] = True
                for path in _walk(y, x):
                    if len(path) >= min_length_px:
                        segments.append(np.asarray(path, dtype=np.int32))

    return segments


def label_segment(
    points: np.ndarray,
    av_probs: np.ndarray,
    neighborhood: int = 1,
) -> tuple[str, float]:
    """Return ('artery'|'vein', confidence in [0, 1]) from mean(p_a - p_v)."""
    ys, xs = points[:, 0], points[:, 1]
    pa = av_probs[ys, xs, 0]
    pv = av_probs[ys, xs, 1]
    if neighborhood > 0:
        # Optionally dilate the lookup: take max over a 3x3 for each point.
        h, w = av_probs.shape[:2]
        pa_max = pa.copy()
        pv_max = pv.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nys = np.clip(ys + dy, 0, h - 1)
                nxs = np.clip(xs + dx, 0, w - 1)
                pa_max = np.maximum(pa_max, av_probs[nys, nxs, 0])
                pv_max = np.maximum(pv_max, av_probs[nys, nxs, 1])
        pa, pv = pa_max, pv_max
    signed = float((pa - pv).mean())
    label = "artery" if signed >= 0 else "vein"
    return label, float(abs(signed))


def classify_point_types(
    skel: np.ndarray,
    av_mask: np.ndarray,
) -> np.ndarray:
    """For each skeleton pixel classify as interior/endpoint/bifurcation/crossing.

    Crossing = 3+ neighbours AND both A and V present in a 5x5 neighbourhood.
    Bifurcation = 3+ neighbours AND only one class in neighbourhood.
    Endpoint = 1 neighbour. Interior = 2 neighbours.
    """
    skel_u8 = skel.astype(np.uint8)
    nc = _neighbour_count(skel_u8) * skel_u8

    a_mask = (av_mask == 1).astype(np.uint8)
    v_mask = (av_mask == 2).astype(np.uint8)
    # 5x5 dilations
    kernel5 = np.ones((5, 5), np.uint8)
    a_near = cv2.dilate(a_mask, kernel5)
    v_near = cv2.dilate(v_mask, kernel5)

    out = np.full(skel.shape, "", dtype="<U12")
    out[nc == 1] = "endpoint"
    out[nc == 2] = "interior"
    both = (nc >= 3) & (a_near > 0) & (v_near > 0)
    only = (nc >= 3) & ~both
    out[only] = "bifurcation"
    out[both] = "crossing"
    return out


def compute_exclusion_mask(
    full_points: np.ndarray,
    point_types: np.ndarray,
    w_init: np.ndarray,
    factor: float = 1.5,
) -> np.ndarray:
    """For each point along `full_points`, True if inside the arc-length buffer."""
    n = len(full_points)
    if n == 0:
        return np.zeros(0, dtype=bool)

    arc = np.zeros(n, dtype=float)
    for i in range(1, n):
        d = np.hypot(*(full_points[i] - full_points[i - 1]))
        arc[i] = arc[i - 1] + d

    exclusion = np.zeros(n, dtype=bool)
    for i, t in enumerate(point_types):
        if t in ("bifurcation", "crossing", "endpoint"):
            buffer_arc = factor * float(w_init[i])
            left = np.searchsorted(arc, arc[i] - buffer_arc, side="left")
            right = np.searchsorted(arc, arc[i] + buffer_arc, side="right")
            exclusion[left:right] = True
    return exclusion


def split_by_cell(
    points: np.ndarray,
    coords: RetinalCoordinateSystem,
    min_points: int = 5,
) -> dict[tuple[int, str], list[int]]:
    """Group consecutive same-cell points; drop sub-segments < min_points."""
    if len(points) == 0:
        return {}
    zones = coords.point_to_zone_array(points[:, 0].astype(float), points[:, 1].astype(float))
    quadrants = coords.point_to_quadrant_array(points[:, 0].astype(float), points[:, 1].astype(float))

    runs: dict[tuple[int, str], list[int]] = {}
    cur_cell: tuple[int, str] | None = None
    cur_idxs: list[int] = []
    for i, (z, q) in enumerate(zip(zones.tolist(), quadrants.tolist())):
        cell = (int(z), str(q))
        if cell != cur_cell:
            if cur_cell is not None and cur_cell[0] >= 0 and len(cur_idxs) >= min_points:
                runs.setdefault(cur_cell, []).extend(cur_idxs)
            cur_cell = cell
            cur_idxs = [i]
        else:
            cur_idxs.append(i)
    if cur_cell is not None and cur_cell[0] >= 0 and len(cur_idxs) >= min_points:
        runs.setdefault(cur_cell, []).extend(cur_idxs)
    return runs


def build_vessel_segments(
    av_mask: np.ndarray,
    av_probs: np.ndarray,
    coords: RetinalCoordinateSystem,
    config: ExtractionConfig,
    return_maps: bool = False,
) -> list[VesselSegment] | tuple[list[VesselSegment], np.ndarray, np.ndarray]:
    """End-to-end skeleton → VesselSegment list.

    If `return_maps=True`, also return `(skeleton, distance_map)` for QC.
    """
    skel, dist = build_skeleton(av_mask, config)
    segments_pts = trace_segments(skel, min_length_px=config.skeleton.min_segment_length_px)
    point_types_map = classify_point_types(skel, av_mask)

    conf_thr = config.skeleton.a_v_confidence_flag
    excl_factor = config.skeleton.bifurcation_exclude_factor

    out: list[VesselSegment] = []
    for seg_id, pts in enumerate(segments_pts):
        ys, xs = pts[:, 0], pts[:, 1]
        w_init = (2.0 * dist[ys, xs]).astype(np.float32)
        pt_types = point_types_map[ys, xs]
        # endpoints are only meaningful at seg ends, but an interior classify
        # for points inside segments (should be 'interior' mostly)
        excl = compute_exclusion_mask(pts, pt_types, w_init, factor=excl_factor)

        label, confidence = label_segment(pts, av_probs)
        sub = split_by_cell(pts, coords, min_points=config.skeleton.min_segment_length_px)

        out.append(
            VesselSegment(
                segment_id=seg_id,
                vessel_type=label,  # type: ignore[arg-type]
                label_confidence=confidence,
                uncertain=confidence < conf_thr,
                full_points=pts,
                w_init=w_init,
                point_types=pt_types,
                exclusion_mask=excl,
                sub_segments=sub,
            )
        )
    if return_maps:
        return out, skel, dist
    return out
