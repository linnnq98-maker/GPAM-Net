"""Resample images and optional labels to a fixed physical spacing.

Images use B-spline interpolation. Labels use nearest-neighbor interpolation
and are resampled onto the output image grid. The script is parameterized so
it can be reused without editing machine-specific paths.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import SimpleITK as sitk

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--out-image-root", type=Path, required=True)
    parser.add_argument("--out-label-root", type=Path, default=None)
    parser.add_argument("--image-suffix", default="INV2")
    parser.add_argument("--label-suffix", default="seg")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("SX", "SY", "SZ"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resample(img: sitk.Image, target_spacing: tuple, is_label: bool) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    out = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        interp,
        img.GetOrigin(),
        target_spacing,
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )
    return out


def process_case(
    sub_dir: Path,
    label_root: Path | None,
    image_out_root: Path,
    label_out_root: Path | None,
    image_suffix: str,
    label_suffix: str,
    target_spacing: tuple[float, float, float],
    overwrite: bool,
) -> str:
    case_id = sub_dir.name
    img_in = sub_dir / f"{case_id}_{image_suffix}.nii.gz"
    if not img_in.exists():
        img_in = sub_dir / f"{case_id}_{image_suffix}.nii"
    if not img_in.exists():
        return f"SKIP {case_id} (no {image_suffix})"

    img_out = image_out_root / case_id / f"{case_id}_{image_suffix}.nii.gz"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not img_out.exists():
        img = sitk.ReadImage(str(img_in), sitk.sitkFloat32)
        img_rs = _resample(img, target_spacing, is_label=False)
        sitk.WriteImage(img_rs, str(img_out), True)

    if label_root is not None and label_out_root is not None:
        lab_in = label_root / case_id / f"{case_id}_{label_suffix}.nii.gz"
        if not lab_in.exists():
            lab_in = label_root / case_id / f"{case_id}_{label_suffix}.nii"
    else:
        lab_in = None
    if lab_in is not None and lab_in.exists():
        lab_out = label_out_root / case_id / f"{case_id}_{label_suffix}.nii.gz"
        lab_out.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not lab_out.exists():
            lab = sitk.ReadImage(str(lab_in))
            # Resample label on image grid to guarantee alignment
            img_ref = sitk.ReadImage(str(img_out))
            lab_f = sitk.Cast(lab, sitk.sitkFloat32)
            lab_rs = sitk.Resample(
                lab_f, img_ref, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32,
            )
            lab_bin = sitk.BinaryThreshold(lab_rs, 0.5, 1e9, 1, 0)
            lab_bin = sitk.Cast(lab_bin, sitk.sitkUInt8)
            sitk.WriteImage(lab_bin, str(lab_out), True)

    return f"DONE {case_id}"


def main():
    args = parse_args()
    if not args.image_root.is_dir():
        raise FileNotFoundError(args.image_root)
    args.out_image_root.mkdir(parents=True, exist_ok=True)
    if args.label_root is not None:
        if not args.label_root.is_dir():
            raise FileNotFoundError(args.label_root)
        if args.out_label_root is None:
            raise ValueError("--out-label-root is required when --label-root is provided")
        args.out_label_root.mkdir(parents=True, exist_ok=True)
    sub_dirs = sorted([p for p in args.image_root.iterdir() if p.is_dir() and p.name.startswith("sub")])
    if not sub_dirs:
        raise SystemExit(f"No subject directories found under {args.image_root}")
    workers = max(1, int(args.workers))
    spacing = tuple(float(v) for v in args.spacing)
    print(f"Total: {len(sub_dirs)} subjects, target_spacing={spacing}, workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                process_case,
                d,
                args.label_root,
                args.out_image_root,
                args.out_label_root,
                args.image_suffix,
                args.label_suffix,
                spacing,
                args.overwrite,
            ): d.name
            for d in sub_dirs
        }
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(sub_dirs)}] {f.result()}")
            except Exception as e:
                print(f"[{i}/{len(sub_dirs)}] FAILED {futs[f]}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
