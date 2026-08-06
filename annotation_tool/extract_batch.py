"""Unpack a batch zip into the clinician folder layout.

This used to be shell — one implementation in bash, another in cmd, each
with its own quoting rules and its own idea of how deep to look for
`images/`. The Windows one only checked the top level, so a zip that had
picked up a wrapper folder in transit extracted nothing, reported nothing,
and still consumed the zip. Here it is in one place, testable on any
machine.

    python annotation_tool/extract_batch.py <batch.zip> <install_dir>

Prints a summary and the two directories the annotation tool needs.
Exit code 0 on success, 1 with an explanation of what the zip contained.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_WRAPPER_DEPTH = 3


class BatchLayoutError(Exception):
    """The zip is not a batch — message includes what it actually held."""


@dataclass
class ExtractResult:
    batch_name: str
    images_dir: Path
    prefill_dir: Path
    n_images: int
    n_prefill: int


def _is_junk(path: Path) -> bool:
    """macOS zips carry resource forks and __MACOSX; Windows sees them as
    real files, and a leading-dot PNG would otherwise look like an image."""
    parts = path.parts
    return any(p == "__MACOSX" or p.startswith("._") or p == ".DS_Store"
               for p in parts) or path.name.startswith(".")


def _images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and not _is_junk(p)
                  and p.suffix.lower() in IMAGE_EXTS)


def _find_dir(root: Path, name: str) -> Path | None:
    """Find `name` at the top level or up to MAX_WRAPPER_DEPTH folders down.

    Breadth-first, so the shallowest match wins and a batch that happens to
    contain a nested folder of the same name does not confuse it.
    """
    level = [root]
    for _ in range(MAX_WRAPPER_DEPTH):
        candidate = [d / name for d in level if (d / name).is_dir()]
        if candidate:
            return candidate[0]
        level = [child for d in level for child in d.iterdir()
                 if child.is_dir() and not _is_junk(child)]
        if not level:
            break
    return None


def _find_loose_images(root: Path) -> Path | None:
    """No `images/` folder — accept images sitting loose at the top.

    Deliberately only the top level, and only through wrapper folders that
    contain nothing else. Searching the whole tree would happily mistake a
    `predictions` folder full of PNGs for the images.
    """
    level = root
    for _ in range(MAX_WRAPPER_DEPTH):
        if _images_in(level):
            return level
        children = [c for c in level.iterdir() if not _is_junk(c)]
        if len(children) != 1 or not children[0].is_dir():
            return None
        level = children[0]
    return None


def _copy_tree(src: Path, dst: Path) -> tuple[int, int]:
    """Copy files and one level of subdirectories, skipping junk.

    Returns (files copied, of which images) — the caller reports image
    counts, so a stray README in `images/` must not inflate them.
    """
    dst.mkdir(parents=True, exist_ok=True)
    total = images = 0
    for item in sorted(src.iterdir()):
        if _is_junk(item):
            continue
        if item.is_file():
            shutil.copy2(item, dst / item.name)
            total += 1
            images += item.suffix.lower() in IMAGE_EXTS
        elif item.is_dir():
            sub = dst / item.name
            sub.mkdir(exist_ok=True)
            for f in sorted(item.iterdir()):
                if f.is_file() and not _is_junk(f):
                    shutil.copy2(f, sub / f.name)
                    total += 1
                    images += f.suffix.lower() in IMAGE_EXTS
    return total, images


def extract_batch(zip_path: Path, install_dir: Path) -> ExtractResult:
    """Unpack `zip_path` into `install_dir/clinician_data/...`.

    The batch name is the zip's filename without its extension, which is
    what the launcher and the annotation output folder both key off.
    """
    zip_path = Path(zip_path)
    install_dir = Path(install_dir)
    if not zip_path.is_file():
        raise BatchLayoutError(f"batch zip not found: {zip_path}")

    batch_name = zip_path.stem
    images_dir = install_dir / "clinician_data" / "images_to_annotate" / batch_name
    prefill_dir = install_dir / "clinician_data" / "predictions" / batch_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_root)
        except zipfile.BadZipFile as exc:
            raise BatchLayoutError(
                f"{zip_path.name} is not a readable zip ({exc}). "
                f"Re-download it — the transfer was probably incomplete."
            ) from exc

        src_images = _find_dir(tmp_root, "images")
        if src_images is None:
            src_images = _find_loose_images(tmp_root)
        if src_images is None:
            listing = "\n".join(
                f"    {p.relative_to(tmp_root).as_posix()}"
                for p in sorted(tmp_root.rglob("*")) if p.is_file()
            ) or "    (the zip is empty)"
            raise BatchLayoutError(
                f"no images found in {zip_path.name}.\n"
                f"Expected an 'images' folder. The zip contains:\n{listing}"
            )

        # Prefill sits beside the images, whatever wrapper they are inside.
        parent = src_images.parent
        src_prefill = None
        for candidate in ("predictions", "masks", "prefill"):
            if (parent / candidate).is_dir():
                src_prefill = parent / candidate
                break

        _, n_images = _copy_tree(src_images, images_dir)
        n_prefill = _copy_tree(src_prefill, prefill_dir)[0] if src_prefill else 0
        prefill_dir.mkdir(parents=True, exist_ok=True)

    if n_images == 0:
        raise BatchLayoutError(
            f"found '{src_images.name}' in {zip_path.name} but it holds no "
            f"images ({', '.join(sorted(IMAGE_EXTS))})."
        )

    return ExtractResult(batch_name=batch_name,
                         images_dir=images_dir.resolve(),
                         prefill_dir=prefill_dir.resolve(),
                         n_images=n_images, n_prefill=n_prefill)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    try:
        result = extract_batch(Path(argv[1]), Path(argv[2]))
    except BatchLayoutError as exc:
        print("")
        print("ERROR: " + str(exc))
        print("")
        print("The zip has been left where it is — nothing was lost.")
        return 1

    print(f"  Batch:    {result.batch_name}")
    print(f"  Images:   {result.n_images}  -> {result.images_dir}")
    if result.n_prefill:
        print(f"  Prefill:  {result.n_prefill}  -> {result.prefill_dir}")
    else:
        print("  Prefill:  none — the vessel layers will start empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
