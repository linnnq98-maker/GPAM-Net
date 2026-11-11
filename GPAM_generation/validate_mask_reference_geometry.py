"""Validate binary masks against reference NIfTI image geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--mask-suffix", default="brain")
    parser.add_argument("--reference-suffix", default="T1")
    parser.add_argument("--affine-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    for case_id in args.case_ids:
        mask_path = args.mask_root / f"{case_id}_{args.mask_suffix}.nii.gz"
        reference_path = args.reference_root / f"{case_id}_{args.reference_suffix}.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        if not reference_path.is_file():
            raise FileNotFoundError(f"Missing reference: {reference_path}")

        mask = nib.load(str(mask_path))
        reference = nib.load(str(reference_path))
        same_shape = mask.shape == reference.shape
        same_affine = np.allclose(
            mask.affine,
            reference.affine,
            rtol=0,
            atol=args.affine_tolerance,
        )
        if not same_shape or not same_affine:
            raise RuntimeError(
                f"{case_id}: geometry mismatch: mask={mask.shape}, "
                f"reference={reference.shape}, same_affine={same_affine}"
            )

        mask_voxels = int(np.count_nonzero(np.asanyarray(mask.dataobj) > 0.5))
        if mask_voxels < 100:
            raise RuntimeError(f"{case_id}: brain mask is too small: {mask_voxels} voxels")
        print(
            f"{case_id}: OK shape={reference.shape} mask_voxels={mask_voxels}",
            flush=True,
        )

    print(f"GEOMETRY_VALIDATION_OK cases={len(args.case_ids)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
