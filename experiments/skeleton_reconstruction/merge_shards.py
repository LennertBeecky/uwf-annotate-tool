"""Merge the per-shard results.csv files produced by Condor shard jobs.

Usage:
    PYTHONPATH=src python -u experiments/skeleton_reconstruction/merge_shards.py \
        [--dir experiments/skeleton_reconstruction] \
        [--glob 'shard*_results.csv'] \
        [--out results.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="experiments/skeleton_reconstruction", type=Path)
    p.add_argument("--glob", default="shard*_results.csv")
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    shard_files = sorted(args.dir.glob(args.glob))
    if not shard_files:
        print(f"no shard files matched {args.dir}/{args.glob}")
        return 1

    out_path = args.dir / args.out
    print(f"merging {len(shard_files)} shard files → {out_path}")

    rows: list[dict] = []
    fieldnames: list[str] | None = None
    for sf in shard_files:
        with sf.open() as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for r in reader:
                rows.append(r)
        print(f"  {sf.name}: {len(rows)} cumulative rows")

    # Deduplicate by (split, filename) — in case of shard overlap or re-runs.
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in rows:
        key = (r.get("split", ""), r.get("filename", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    print(f"  deduplicated: {len(rows)} → {len(unique)} rows")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in unique:
            writer.writerow(r)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
