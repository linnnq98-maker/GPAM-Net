#!/usr/bin/env python3
"""Validate generated polar-SP NPZ files against reference NIfTI images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--case-ids-file", type=Path, required=True)
    parser.add_argument("--reference-suffix", default="T1")
    parser.add_argument("--expected-channels", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_ids = [
        line.strip()
        for line in args.case_ids_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    problems: list[str] = []
    occupancies: list[float] = []

    for index, case_id in enumerate(case_ids, start=1):
        sp_path = args.sp_root / f"{case_id}_sp.npz"
        reference_path = (
            args.reference_root / f"{case_id}_{args.reference_suffix}.nii.gz"
        )
        try:
            if not sp_path.is_file():
                raise FileNotFoundError(sp_path)
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)

            with np.load(sp_path, allow_pickle=False) as archive:
                if "sp" not in archive:
                    raise KeyError("missing array 'sp'")
                sp = archive["sp"]

            reference = sitk.ReadImage(str(reference_path))
            expected_shape = (args.expected_channels, *reversed(reference.GetSize()))
            if sp.shape != expected_shape:
                raise ValueError(f"shape={sp.shape}, expected={expected_shape}")
            if not np.issubdtype(sp.dtype, np.floating):
                raise TypeError(f"unexpected dtype={sp.dtype}")
            if not np.isfinite(sp).all():
                raise ValueError("contains NaN or Inf")

            occupancy = float(np.any(sp != 0, axis=0).mean() * 100.0)
            occupancies.append(occupancy)

            print(
                f"[{index:02d}/{len(case_ids)}] {case_id}: OK "
                f"shape={sp.shape} range=[{float(sp.min()):.6g}, {float(sp.max()):.6g}] "
                f"nonzero={occupancy:.4f}%",
                flush=True,
            )
        except Exception as error:
            problems.append(f"{case_id}: {error}")
            print(f"[{index:02d}/{len(case_ids)}] {case_id}: FAILED {error}", flush=True)

    npz_count = len(list(args.sp_root.glob("*_sp.npz")))
    print("\n[FINAL]")
    print(f"case IDs: {len(case_ids)}")
    print(f"SP NPZ files: {npz_count}")
    print(f"valid cases: {len(case_ids) - len(problems)}/{len(case_ids)}")
    if occupancies:
        print(f"mean nonzero: {float(np.mean(occupancies)):.4f}%")
        print(f"min nonzero: {float(np.min(occupancies)):.4f}%")
        print(f"max nonzero: {float(np.max(occupancies)):.4f}%")
    if problems:
        print("Problems:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("SP_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
