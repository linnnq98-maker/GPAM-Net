"""Resample binary brain masks to each case's reference image grid."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def case_number(name: str) -> int | None:
    match = re.search(r"(?:sub|patient)[-_]?0*(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def find_mask(mask_root: Path, number: int, suffix: str) -> Path | None:
    names = [
        f"sub{number:03d}_{suffix}.nii.gz",
        f"sub{number:03d}_{suffix}.nii",
        f"patient{number:02d}_{suffix}.nii.gz",
        f"patient{number:02d}_{suffix}.nii",
    ]
    for name in names:
        path = mask_root / name
        if path.is_file():
            return path
    return None


def find_reference(reference_root: Path, case: str, number: int, suffix: str) -> Path | None:
    names = [
        f"{case}_{suffix}.nii.gz",
        f"{case}_{suffix}.nii",
        f"sub{number:03d}_{suffix}.nii.gz",
        f"patient{number:02d}_{suffix}.nii.gz",
    ]
    for name in names:
        path = reference_root / name
        if path.is_file():
            return path
    return None


def read_case_ids(path: Path) -> list[str]:
    cases = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(line)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--case-ids-file", type=Path, required=True)
    parser.add_argument("--mask-suffix", default="brain")
    parser.add_argument("--reference-suffix", default="T1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.mask_root.is_dir():
        raise FileNotFoundError(args.mask_root)
    if not args.reference_root.is_dir():
        raise FileNotFoundError(args.reference_root)
    cases = read_case_ids(args.case_ids_file)
    args.out_root.mkdir(parents=True, exist_ok=True)

    complete = 0
    for index, case in enumerate(cases, start=1):
        number = case_number(case)
        if number is None:
            raise ValueError(f"Cannot parse case number: {case}")
        mask_path = find_mask(args.mask_root, number, args.mask_suffix)
        ref_path = find_reference(args.reference_root, case, number, args.reference_suffix)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {case} in {args.mask_root}")
        if ref_path is None:
            raise FileNotFoundError(f"Missing reference for {case} in {args.reference_root}")

        out_path = args.out_root / f"{case}_{args.mask_suffix}.nii.gz"
        if out_path.is_file() and not args.overwrite:
            print(f"[{index:02d}/{len(cases):02d}] skip {case}", flush=True)
            complete += 1
            continue

        mask = sitk.ReadImage(str(mask_path))
        reference = sitk.ReadImage(str(ref_path))
        result = sitk.Resample(
            mask,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0.0,
            sitk.sitkUInt8,
        )
        result = sitk.Cast(result > 0, sitk.sitkUInt8)
        sitk.WriteImage(result, str(out_path), useCompression=True)
        complete += 1
        print(
            f"[{index:02d}/{len(cases):02d}] {case}: "
            f"mask={mask.GetSize()} -> ref={reference.GetSize()}",
            flush=True,
        )

    outputs = list(args.out_root.glob(f"*_{args.mask_suffix}.nii.gz"))
    print(f"[FINAL] resampled masks: {len(outputs)}/{len(cases)}")
    if complete != len(cases):
        raise RuntimeError(f"Only completed {complete}/{len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
