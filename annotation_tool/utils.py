"""Helpers for the napari annotation tool.

Pure-NumPy/scikit-image utilities — no napari dependency. Unit-testable
on any machine whether or not napari/Qt is installed.
"""

from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage.transform import downscale_local_mean

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

ARTERY_COLOR_INDEX = 1
VEIN_COLOR_INDEX = 1   # both layers use label value 1; napari colour comes from the layer, not the value


def load_image_rgb(image_path: Path) -> np.ndarray:
    """Load an image as an (H, W, 3) uint8 RGB numpy array.

    Drops the alpha channel if present. Handles grayscale by triplicating
    across channels so napari's RGB display path is used consistently.
    """
    img = Image.open(str(image_path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def build_multiscale_pyramid(image: np.ndarray, max_side_for_single: int = 2000
                             ) -> tuple[list[np.ndarray], bool]:
    """Return a napari-ready multiscale list when the image is large.

    For images smaller than `max_side_for_single` on both sides, returns
    `[image]` with `multiscale=False` semantics (single-level). For larger
    images, returns `[full, half, quarter]` so napari can swap in the
    downscaled versions when zoomed out, keeping the UI snappy.

    The label layers should always be attached at full resolution; only
    the BACKGROUND image layer benefits from multiscale.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_side_for_single:
        return [image], False
    # Factor-2 and factor-4 downscales. Keep channel dim (factor 1).
    half = downscale_local_mean(image, (2, 2, 1)).astype(np.uint8)
    quarter = downscale_local_mean(image, (4, 4, 1)).astype(np.uint8)
    return [image, half, quarter], True


def skeletonise_mask(mask: np.ndarray) -> np.ndarray:
    """Binarise (>0) → `skeletonize` → uint8 {0, 255}.

    Returns a 1-pixel-wide skeleton PNG-ready array of the same HxW as the
    input mask.
    """
    bin_mask = mask > 0
    if not bin_mask.any():
        return np.zeros(mask.shape[:2], dtype=np.uint8)
    skel = skeletonize(bin_mask)
    return (skel.astype(np.uint8) * 255)


def binarise_mask(mask: np.ndarray) -> np.ndarray:
    """Binarise (>0) → uint8 {0, 255}, keeping the painted width.

    The boundary-mode counterpart of `skeletonise_mask`: the thickness of
    what was painted *is* the vessel width, so nothing is thinned. Matches
    the format of the DVA ground-truth masks in `databases/DVA/raw/`.
    """
    return ((mask > 0).astype(np.uint8) * 255)


def resize_region(mask: np.ndarray, region: np.ndarray, delta: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pixels to add / remove to grow or shrink `region` by `delta` px.

    Growing dilates the region; shrinking erodes it but never past the
    vessel's centreline, so a vessel can be narrowed to 1 px and no
    further — narrowing can't delete it or break it in two. Erosion is
    computed against the whole mask, not the region alone, so the cut ends
    where the segment meets its neighbours are not notched.
    """
    from scipy import ndimage

    st = np.ones((3, 3), dtype=bool)
    k = abs(int(delta))
    added = np.zeros(mask.shape[:2], dtype=bool)
    removed = np.zeros(mask.shape[:2], dtype=bool)
    if k == 0 or not region.any():
        return added, removed

    y0, y1, x0, x1 = _bbox(region, pad=k + 3)
    m = mask[y0:y1, x0:x1] > 0
    reg = region[y0:y1, x0:x1]

    if delta > 0:
        grown = ndimage.binary_dilation(reg, st, iterations=k)
        added[y0:y1, x0:x1] = grown & ~m
    else:
        # border_value=1 stops the crop edge from eroding inwards.
        eroded = ndimage.binary_erosion(m, st, iterations=k, border_value=1)
        keep = eroded | skeletonize(m)
        removed[y0:y1, x0:x1] = reg & m & ~keep
    return added, removed


def save_skeleton_png(skeleton_uint8: np.ndarray, path: Path) -> None:
    """Save a (H, W) uint8 {0, 255} mask as a lossless PNG."""
    if skeleton_uint8.dtype != np.uint8:
        raise TypeError(f"expected uint8, got {skeleton_uint8.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(skeleton_uint8, mode="L").save(str(path))


def neighbour_count_8(mask: np.ndarray) -> np.ndarray:
    """Per-pixel count of 8-neighbour foreground pixels (uint8)."""
    from scipy.signal import convolve2d
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    m = (mask > 0).astype(np.uint8)
    return convolve2d(m, k, mode="same", boundary="fill", fillvalue=0).astype(np.uint8) * m


# ---- A/V reclassification helpers -------------------------------------
#
# The annotator hovers over a mislabelled vessel and presses a key; these
# functions decide which pixels belong to "that vessel" so the caller can
# move them from one label layer to the other.

SNAP_RADIUS = 8          # px: how far from the cursor we look for a vessel
BRANCH_PAD = 48          # px: window grown around a branch when reclaiming width


def nearest_foreground(mask: np.ndarray, yx: tuple[float, float],
                       radius: int = SNAP_RADIUS) -> tuple[int, int] | None:
    """Nearest foreground pixel to `yx` within `radius`, or None.

    The cursor rarely lands exactly on a 1-px skeleton, so every click has
    to snap to the vessel it is pointing at.
    """
    m = mask > 0
    h, w = m.shape[:2]
    y, x = int(round(yx[0])), int(round(yx[1]))
    if not (0 <= y < h and 0 <= x < w):
        return None
    if m[y, x]:
        return y, x
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    ys, xs = np.nonzero(m[y0:y1, x0:x1])
    if ys.size == 0:
        return None
    d2 = (ys + y0 - y) ** 2 + (xs + x0 - x) ** 2
    i = int(np.argmin(d2))
    return int(ys[i] + y0), int(xs[i] + x0)


def owner_of_point(masks: dict[str, np.ndarray], yx: tuple[float, float],
                   radius: int = SNAP_RADIUS) -> str | None:
    """Key of the mask whose nearest foreground pixel is closest to `yx`.

    Decides which class the vessel under the cursor currently belongs to,
    so a swap can be triggered without first selecting the right layer.
    Returns None when nothing is within `radius`.
    """
    best, best_d2 = None, None
    for name, mask in masks.items():
        hit = nearest_foreground(mask, yx, radius)
        if hit is None:
            continue
        d2 = (hit[0] - yx[0]) ** 2 + (hit[1] - yx[1]) ** 2
        if best_d2 is None or d2 < best_d2:
            best, best_d2 = name, d2
    return best


def component_at(mask: np.ndarray, yx: tuple[float, float],
                 radius: int = SNAP_RADIUS) -> np.ndarray | None:
    """Whole 8-connected component of `mask` under `yx` (bool), or None.

    This is the "swap the entire vessel tree" granularity — everything that
    touches the clicked pixel, junctions included.
    """
    from scipy import ndimage

    m = mask > 0
    seed = nearest_foreground(m, yx, radius)
    if seed is None:
        return None
    lab, _ = ndimage.label(m, structure=np.ones((3, 3), dtype=np.uint8))
    return lab == lab[seed]


def _bbox(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape[:2]
    return (max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad),
            max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad))


def _voronoi_select(sub: np.ndarray, selected: np.ndarray,
                    competitors: np.ndarray, pad: int = BRANCH_PAD
                    ) -> np.ndarray:
    """Pixels of `sub` nearer to the `selected` skeleton than to `competitors`.

    Recovers the painted width around a chosen skeleton subset. Resolved in
    a window around `selected` rather than over the whole shape — a
    full-tree EDT is noticeably laggy on UWF-sized images, and a pixel
    further than `pad` from the selection cannot win it anyway.
    """
    from scipy import ndimage

    labels = np.zeros(sub.shape, dtype=np.int32)
    labels[competitors] = 2
    labels[selected] = 1
    y0, y1, x0, x1 = _bbox(selected, pad=pad)
    win = labels[y0:y1, x0:x1]
    _, (iy, ix) = ndimage.distance_transform_edt(win == 0, return_indices=True)
    out = np.zeros_like(sub)
    out[y0:y1, x0:x1] = sub[y0:y1, x0:x1] & (win[iy, ix] == 1)
    return out


def _skeleton_graph(skel: np.ndarray):
    """8-connected sparse graph over skeleton pixels, plus their coordinates."""
    from scipy.sparse import coo_matrix

    h, w = skel.shape
    idx = np.full(skel.shape, -1, dtype=np.int64)
    ys, xs = np.nonzero(skel)
    idx[ys, xs] = np.arange(ys.size)

    rows, cols, wts = [], [], []
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)):
        a = idx[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
        b = idx[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
        keep = (a >= 0) & (b >= 0)
        rows.append(a[keep])
        cols.append(b[keep])
        wts.append(np.full(int(keep.sum()), float(np.hypot(dy, dx))))
    n = ys.size
    graph = coo_matrix(
        (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    return graph, idx, ys, xs


def path_between(mask: np.ndarray, yx1: tuple[float, float],
                 yx2: tuple[float, float], radius: int = SNAP_RADIUS
                 ) -> np.ndarray | None:
    """Just the stretch of vessel between two marked points (bool), or None.

    For when only *part* of a branch is misclassified: the two ends of the
    bad stretch are marked and the geodesic path along the vessel between
    them — through junctions if need be — is what moves. Returns None when
    either end misses the mask or the two ends are on different vessels.
    """
    from scipy import ndimage
    from scipy.sparse.csgraph import dijkstra

    m = mask > 0
    p1 = nearest_foreground(m, yx1, radius)
    p2 = nearest_foreground(m, yx2, radius)
    if p1 is None or p2 is None:
        return None

    struct8 = np.ones((3, 3), dtype=np.uint8)
    lab_all, _ = ndimage.label(m, structure=struct8)
    if lab_all[p1] != lab_all[p2]:
        return None                      # two different vessels
    comp = lab_all == lab_all[p1]

    y0, y1, x0, x1 = _bbox(comp, pad=2)
    sub = comp[y0:y1, x0:x1]
    skel = skeletonize(sub)
    if not skel.any():
        return comp

    # The marks sit on the painted vessel; move them onto its centreline.
    s1 = nearest_foreground(skel, (p1[0] - y0, p1[1] - x0), radius=max(radius, 32))
    s2 = nearest_foreground(skel, (p2[0] - y0, p2[1] - x0), radius=max(radius, 32))
    if s1 is None or s2 is None:
        return comp

    graph, idx, ys, xs = _skeleton_graph(skel)
    src, dst = int(idx[s1]), int(idx[s2])
    dist, pred = dijkstra(graph, indices=src, return_predecessors=True)
    if not np.isfinite(dist[dst]):
        return None                      # skeleton is disconnected between them

    walk, node = [], dst
    while node != src and node >= 0:
        walk.append(node)
        node = int(pred[node])
    walk.append(src)

    path = np.zeros_like(skel)
    path[ys[walk], xs[walk]] = True
    region = _voronoi_select(sub, path, skel & ~path)

    out = np.zeros_like(comp)
    out[y0:y1, x0:x1] = region
    return out


def branch_at(mask: np.ndarray, yx: tuple[float, float],
              radius: int = SNAP_RADIUS) -> np.ndarray | None:
    """Single vessel *segment* of `mask` under `yx` (bool), or None.

    The component is skeletonised, cut at its junctions, and the branch
    containing the cursor is selected; the painted width is then reclaimed
    by assigning each component pixel to its nearest branch (a Voronoi
    split), so junction pixels go to whichever branch they sit closest to
    instead of being left behind in the old class.

    Falls back to the whole component when the shape has no junctions or is
    too small to skeletonise meaningfully.
    """
    from scipy import ndimage

    m = mask > 0
    seed = nearest_foreground(m, yx, radius)
    if seed is None:
        return None
    struct8 = np.ones((3, 3), dtype=np.uint8)
    lab_all, _ = ndimage.label(m, structure=struct8)
    comp = lab_all == lab_all[seed]

    y0, y1, x0, x1 = _bbox(comp, pad=2)
    sub = comp[y0:y1, x0:x1]
    sy, sx = seed[0] - y0, seed[1] - x0

    skel = skeletonize(sub)
    if not skel.any():
        return comp
    nbrs = neighbour_count_8(skel)
    junction = skel & (nbrs >= 3)
    if junction.any():
        junction = ndimage.binary_dilation(junction, structure=struct8) & skel
    segments, n_seg = ndimage.label(skel & ~junction, structure=struct8)
    if n_seg <= 1:
        return comp

    # The cursor may have landed on a junction pixel (label 0), so take the
    # label of the nearest segment rather than the one underneath it.
    _, (iy, ix) = ndimage.distance_transform_edt(segments == 0, return_indices=True)
    chosen = int(segments[iy[sy, sx], ix[sy, sx]])
    if chosen == 0:
        return comp

    seg_mask = segments == chosen
    region = _voronoi_select(sub, seg_mask, (segments > 0) & ~seg_mask)

    out = np.zeros_like(comp)
    out[y0:y1, x0:x1] = region
    return out


def disk_at(shape: tuple[int, int], yx: tuple[float, float],
            radius: int) -> np.ndarray:
    """Filled disk of `radius` centred on `yx` — the freehand escape hatch."""
    h, w = shape[:2]
    y, x = float(yx[0]), float(yx[1])
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius) + 1)
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius) + 1)
    out = np.zeros((h, w), dtype=bool)
    if y0 >= y1 or x0 >= x1:
        return out
    yy, xx = np.ogrid[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    return out


def apply_swap(src: np.ndarray, dst: np.ndarray, region: np.ndarray) -> dict:
    """Move `region` from label array `src` to `dst`, in place.

    Returns an undo record: the pixels actually cleared from `src` and the
    pixels actually set in `dst` (a crossing pixel already present in `dst`
    must not be cleared on undo).
    """
    moved = region & (src > 0)
    added = region & (dst == 0)
    src[moved] = 0
    dst[region] = 1
    return {"removed": np.nonzero(moved), "added": np.nonzero(added),
            "n": int(moved.sum())}


def undo_swap(src: np.ndarray, dst: np.ndarray, record: dict) -> None:
    """Exact inverse of `apply_swap` for one record, in place."""
    src[record["removed"]] = 1
    dst[record["added"]] = 0


class AVSwapper:
    """Undoable artery ⇄ vein reclassification over two label arrays.

    Holds the two mask arrays *by reference* and edits them in place, so a
    napari Labels layer can keep pointing at the same object. The caller
    supplies a point (the cursor) and a granularity; the swapper works out
    which class currently owns that point and moves the pixels across.
    """

    MODES = ("branch", "component", "disk")

    def __init__(self, masks: dict[str, np.ndarray],
                 snap_radius: int = SNAP_RADIUS) -> None:
        if len(masks) != 2:
            raise ValueError(f"expected exactly 2 label arrays, got {len(masks)}")
        self.masks = masks
        self.snap_radius = snap_radius
        self._undo: list[dict] = []
        self._resize_anchor: dict | None = None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    def owner_at(self, yx: tuple[float, float]) -> str | None:
        """Which class holds the vessel at `yx`, or None if there is none."""
        return owner_of_point(self.masks, yx, self.snap_radius)

    def _region(self, src: np.ndarray, yx: tuple[float, float],
                mode: str, radius: int | None) -> np.ndarray | None:
        if mode == "branch":
            return branch_at(src, yx, self.snap_radius)
        if mode == "component":
            return component_at(src, yx, self.snap_radius)
        if mode == "disk":
            return disk_at(src.shape, yx, radius or self.snap_radius) & (src > 0)
        raise ValueError(f"unknown swap mode {mode!r} (expected one of {self.MODES})")

    def swap_at(self, yx: tuple[float, float], mode: str = "branch",
                radius: int | None = None) -> dict | None:
        """Move the vessel at `yx` to the other class. None if nothing there."""
        if mode not in self.MODES:
            raise ValueError(f"unknown swap mode {mode!r} (expected one of {self.MODES})")
        src_name = owner_of_point(self.masks, yx, self.snap_radius)
        if src_name is None:
            return None
        dst_name = next(k for k in self.masks if k != src_name)
        src, dst = self.masks[src_name], self.masks[dst_name]
        region = self._region(src, yx, mode, radius)
        if region is None or not region.any():
            return None
        rec = apply_swap(src, dst, region)
        rec.update(kind=mode, src=src_name, dst=dst_name)
        self._undo.append(rec)
        self._resize_anchor = None
        return rec

    def swap_between(self, yx1: tuple[float, float],
                     yx2: tuple[float, float]) -> dict | None:
        """Move only the stretch of vessel between two marked points.

        None if either mark misses a vessel or the two are on different
        vessels — the caller should tell the annotator rather than guess.
        """
        src_name = owner_of_point(self.masks, yx1, self.snap_radius)
        if src_name is None:
            return None
        dst_name = next(k for k in self.masks if k != src_name)
        src, dst = self.masks[src_name], self.masks[dst_name]
        region = path_between(src, yx1, yx2, self.snap_radius)
        if region is None or not region.any():
            return None
        rec = apply_swap(src, dst, region)
        rec.update(kind="range", src=src_name, dst=dst_name)
        self._undo.append(rec)
        self._resize_anchor = None
        return rec

    def resize_at(self, yx: tuple[float, float], delta: int) -> dict | None:
        """Grow (delta>0) or shrink (delta<0) the vessel segment at `yx`.

        Boundary mode only makes sense when the painted thickness is the
        annotation, so this edits one class in place and leaves the other
        alone — a crossing vessel keeps its own pixels.
        """
        name = self.owner_at(yx)
        if name is None:
            return None
        mask = self.masks[name]

        # Editing the width changes the mask's skeleton, which moves the
        # junctions, which changes which branch `branch_at` would return.
        # Re-deriving it every press makes repeated [.] jump between
        # branches, so hold the segment while the cursor stays put.
        anchor = self._resize_anchor
        near = (anchor is not None and anchor["layer"] == name
                and (anchor["yx"][0] - yx[0]) ** 2 + (anchor["yx"][1] - yx[1]) ** 2
                <= self.snap_radius ** 2)
        if near:
            region = anchor["region"]
        else:
            region = branch_at(mask, yx, self.snap_radius)
            if region is None or not region.any():
                self._resize_anchor = None
                return None

        added, removed = resize_region(mask, region, delta)
        if not added.any() and not removed.any():
            return None
        mask[added] = 1
        mask[removed] = 0
        region = (region | added) & ~removed
        self._resize_anchor = {"yx": yx, "layer": name, "region": region}
        rec = {"kind": "widen" if delta > 0 else "narrow", "layer": name,
               "added": np.nonzero(added), "removed": np.nonzero(removed),
               "n": int(added.sum() + removed.sum()),
               "segment_px": int(region.sum())}
        self._undo.append(rec)
        return rec

    def swap_all(self) -> dict:
        """Exchange the two classes wholesale (a fully inverted prediction)."""
        a_name, b_name = tuple(self.masks)
        a, b = self.masks[a_name], self.masks[b_name]
        tmp = a.copy()
        a[...] = b
        b[...] = tmp
        rec = {"kind": "global", "src": a_name, "dst": b_name,
               "n": int((a > 0).sum() + (b > 0).sum())}
        self._undo.append(rec)
        self._resize_anchor = None
        return rec

    def undo(self) -> dict | None:
        """Revert the most recent swap. None if there is nothing to revert."""
        if not self._undo:
            return None
        rec = self._undo.pop()
        self._resize_anchor = None
        if "layer" in rec:                       # single-layer width edit
            mask = self.masks[rec["layer"]]
            mask[rec["added"]] = 0
            mask[rec["removed"]] = 1
        elif rec["kind"] == "global":
            a, b = (self.masks[rec["src"]], self.masks[rec["dst"]])
            tmp = a.copy()
            a[...] = b
            b[...] = tmp
        else:
            undo_swap(self.masks[rec["src"]], self.masks[rec["dst"]], rec)
        return rec


@dataclass
class SaveValidation:
    ok: bool
    messages: list[str]


def validate_saved_skeleton(png_path: Path, expected_shape: tuple[int, int],
                            check_thin: bool = True) -> SaveValidation:
    """Post-hoc sanity check on a saved mask PNG.

    - File exists and is readable
    - Shape matches `expected_shape`
    - dtype is uint8 with unique values subset of {0, 255}
    - Skeleton is 1-pixel wide: max 8-neighbour count ≤ 3 (Y/T junctions
      are allowed; 4 means an unthinned blob survived).

    The thickness probe only applies to skeleton output; pass
    `check_thin=False` for boundary mode, where thickness is the point.
    """
    msgs: list[str] = []
    if not png_path.exists():
        return SaveValidation(ok=False, messages=[f"file missing: {png_path}"])
    try:
        arr = np.asarray(Image.open(str(png_path)))
    except Exception as exc:
        return SaveValidation(ok=False, messages=[f"cannot read {png_path}: {exc}"])

    ok = True
    if arr.ndim != 2 or arr.shape != expected_shape:
        msgs.append(f"shape {arr.shape} ≠ expected {expected_shape}")
        ok = False
    if arr.dtype != np.uint8:
        msgs.append(f"dtype {arr.dtype} ≠ uint8")
        ok = False
    uniques = set(np.unique(arr).tolist())
    if not uniques.issubset({0, 255}):
        msgs.append(f"unique values {sorted(uniques)} contain values ≠ {{0, 255}}")
        ok = False
    # Skeleton thickness probe
    if check_thin and (arr > 0).any():
        nc = neighbour_count_8(arr)
        max_nc = int(nc.max()) if nc.any() else 0
        if max_nc > 3:
            msgs.append(f"skeleton may not be 1-pixel wide (max 8-neighbours = {max_nc})")
            ok = False
    return SaveValidation(ok=ok, messages=msgs)


def align_mask_to_image(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Centre-crop or centre-pad `mask` to `shape` (rows only).

    DVA masks are stored in the 720×720 letterboxed frame while the images
    in `DVA_maastricht/*/images/` are the unpadded 576×720 frames. Silently
    mis-aligning those would corrupt an annotation, so reconcile them here
    and refuse anything but a row-count difference.
    """
    mh, mw = mask.shape[:2]
    th, tw = shape
    if mw != tw:
        raise ValueError(f"mask width {mw} != image width {tw}; cannot align")
    if mh == th:
        return mask
    if mh > th:
        top = (mh - th) // 2
        return mask[top:top + th]
    out = np.zeros((th, tw), dtype=mask.dtype)
    top = (th - mh) // 2
    out[top:top + mh] = mask
    return out


def load_mask_prefill(masks_dir: Path, stem: str, shape: tuple[int, int]
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """Load filled artery/vein masks for `stem`, or None if absent.

    Accepts both layouts in use: `<dir>/artery/<stem>.png` (the DVA
    ground-truth and model-output layout) and `<dir>/<stem>_artery.png`
    (what this tool writes, so a saved annotation can be reopened).
    """
    for art_path, vein_path in (
        (masks_dir / "artery" / f"{stem}.png", masks_dir / "veins" / f"{stem}.png"),
        (masks_dir / f"{stem}_artery.png", masks_dir / f"{stem}_veins.png"),
    ):
        if art_path.exists() and vein_path.exists():
            out = []
            for p in (art_path, vein_path):
                arr = np.asarray(Image.open(str(p)))
                if arr.ndim == 3:
                    arr = arr[..., 0]
                out.append(align_mask_to_image((arr > 0).astype(np.uint8), shape))
            return out[0], out[1]
    return None


def list_images(path: Path) -> list[Path]:
    """Return the images in `path` (file → [path], dir → sorted list)."""
    if path.is_file():
        if path.suffix.lower() in IMG_EXTS:
            return [path]
        raise ValueError(f"{path} is not a supported image ({IMG_EXTS})")
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS
                      and not p.name.startswith("."))
    raise FileNotFoundError(path)


def already_annotated(image_stem: str, output_dir: Path) -> bool:
    a = output_dir / f"{image_stem}_artery.png"
    v = output_dir / f"{image_stem}_veins.png"
    return a.exists() and v.exists()


def append_time_log(csv_path: Path, row: dict) -> None:
    """Append one row to the annotation_times.csv log, creating the file + header lazily."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    fields = ["timestamp", "image_filename", "duration_seconds",
              "artery_pixel_count", "vein_pixel_count"]
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


EDIT_LOG_FIELDS = [
    "timestamp", "image_filename", "duration_seconds",
    "prefill_source", "lunet_thresh",
    "artery_seed_px", "artery_final_px",
    "artery_kept_px", "artery_added_px", "artery_removed_px", "artery_iou",
    "vein_seed_px", "vein_final_px",
    "vein_kept_px", "vein_added_px", "vein_removed_px", "vein_iou",
]


def skeleton_edit_distance(seed_skel: np.ndarray, final_skel: np.ndarray) -> dict:
    """Pixel-level comparison of two binary skeletons.

    Both inputs: (H, W) arrays where >0 means skeleton pixel. Returns raw
    counts and Jaccard IoU. Safe when either is empty (IoU=1.0 if both are).
    Note: skeletons are sparse, so even small spatial shifts drop IoU
    significantly — counts are the more robust signal.
    """
    seed_bin = seed_skel > 0
    final_bin = final_skel > 0
    kept = int((seed_bin & final_bin).sum())
    seed_px = int(seed_bin.sum())
    final_px = int(final_bin.sum())
    union = int((seed_bin | final_bin).sum())
    iou = (kept / union) if union > 0 else 1.0
    return {
        "seed_px": seed_px,
        "final_px": final_px,
        "kept_px": kept,
        "added_px": final_px - kept,
        "removed_px": seed_px - kept,
        "iou": round(float(iou), 4),
    }


def append_edit_log(csv_path: Path, row: dict) -> None:
    """Append one row to the annotation_edits.csv audit log."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDIT_LOG_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def lunet_prefill_masks(
    image_path: Path,
    model_path: Path,
    thresh: float = 0.5,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run LUNet A/V segmentation (cached) and return binary seed masks.

    Returns (artery_mask, vein_mask) as uint8 {0, 1} at the image's native
    resolution. Cached probabilities are stored to
    `cache_dir/{stem}_probs.npz` as float16 so re-opens skip inference.

    Lazy imports cv2/onnxruntime so the utils test suite stays light.
    """
    import cv2  # noqa: PLC0415

    stem = image_path.stem
    cache_path = cache_dir / f"{stem}_probs.npz" if cache_dir is not None else None

    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as z:
            art_probs = z["artery"].astype(np.float32)
            vein_probs = z["vein"].astype(np.float32)
    else:
        import sys as _sys  # noqa: PLC0415

        _src_dir = Path(__file__).resolve().parents[1] / "src"
        if _src_dir.exists() and str(_src_dir) not in _sys.path:
            _sys.path.insert(0, str(_src_dir))
        from uwf_zonal_extraction.segmentation.lunet import (  # noqa: PLC0415
            LunetSegmenter,
        )

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"cv2 could not read {image_path}")
        seg = LunetSegmenter(model_path)
        probs = seg.predict_tiled(image_bgr)
        art_probs = probs[..., 0]
        vein_probs = probs[..., 1]
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                artery=art_probs.astype(np.float16),
                vein=vein_probs.astype(np.float16),
            )

    art_mask = np.where(np.isfinite(art_probs), art_probs > thresh, False).astype(np.uint8)
    vein_mask = np.where(np.isfinite(vein_probs), vein_probs > thresh, False).astype(np.uint8)
    return art_mask, vein_mask


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def human_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"
