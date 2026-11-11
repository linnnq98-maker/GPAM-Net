#!/usr/bin/env python
"""Build the GPAM-Net disk cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cached_dataset import build_cache


def parse_folds(value: str) -> list[int]:
    if "-" in value:
        a, b = value.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def load_cases_from_splits(path: Path, folds: list[int], split_key: str) -> list[str]:
    splits = json.loads(path.read_text(encoding="utf-8"))
    cases: list[str] = []
    for fold in folds:
        split = splits[fold]
        if split_key == "all":
            for key in ("train", "val", "test"):
                cases.extend(split.get(key, []))
        else:
            cases.extend(split.get(split_key, []))
    return sorted(set(cases))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inv2-root", required=True)
    p.add_argument("--flair-root", required=True)
    p.add_argument("--labels-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--splits-json", required=True)
    p.add_argument("--folds", default="0-4")
    p.add_argument("--split-key", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--ring-dilate-voxels", type=int, default=5)
    p.add_argument("--sp-root", default=None)
    p.add_argument("--sp-suffix", default="sp")
    p.add_argument("--sp-channels", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    folds = parse_folds(args.folds)
    cases = load_cases_from_splits(Path(args.splits_json), folds, args.split_key)
    print(f"[DiskCache] cases={len(cases)} folds={folds} split_key={args.split_key}", flush=True)
    written, skipped = build_cache(
        cache_root=Path(args.cache_root),
        inv2_root=Path(args.inv2_root),
        flair_root=Path(args.flair_root),
        labels_root=Path(args.labels_root),
        case_ids=cases,
        ring_dilate_voxels=args.ring_dilate_voxels,
        sp_root=Path(args.sp_root) if args.sp_root else None,
        sp_suffix=args.sp_suffix,
        sp_channels=args.sp_channels,
        overwrite=args.overwrite,
    )
    print(f"[DiskCache] done written={written} skipped={skipped} out={args.cache_root}", flush=True)


if __name__ == "__main__":
    main()

