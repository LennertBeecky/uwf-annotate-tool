"""Download (or verify-by-hash) the ONNX model files.

v1: STUB. Assumes models are copied in by hand from the parent repo
(see README.md). Prints a hash check. For a public Zenodo release the
download URLs and SHA256 sums will be added here.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

EXPECTED = {
    "lunetv2Large.onnx": None,   # SHA256 — TODO fill when a canonical release is published
    "lunetv2_odc.onnx": None,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    missing: list[str] = []
    for name in EXPECTED:
        p = MODELS_DIR / name
        if not p.exists():
            missing.append(name)
            continue
        digest = _sha256(p)
        print(f"{name}: {digest}")
        if EXPECTED[name] and digest != EXPECTED[name]:
            print(f"  WARNING: SHA256 mismatch for {name}")

    if missing:
        print()
        print("Missing models:", *missing, sep="\n  ")
        print()
        print("Copy them from the parent repo:")
        print("  cp ../Physics-Informed_Fundus/lunet/lunetv2Large.onnx  models/")
        print("  cp ../Physics-Informed_Fundus/lunet/lunetv2_odc.onnx   models/")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
