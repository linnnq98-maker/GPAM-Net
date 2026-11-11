"""Generate white-matter-centered spatial position (SP) maps.

Input: one white matter mask per case. By default the script writes three maps:

  0: wm_edt   normalized Euclidean distance to the nearest WM voxel
  1: theta    azimuth angle around the WM centroid, normalized to [-1, 1]
  2: phi      elevation angle around the WM centroid, normalized to [-1, 1]

This matches the idea of using the WM centroid as the spatial origin while also
including an Euclidean distance transform (EDT) of the WM mask. If you want a
strict spherical radius from the WM centroid instead of WM EDT, use
`--radial-map centroid`.

Outputs:
  - <case_id>_sp.npz with array `sp` shaped [C, Z, Y, X]
  - <case_id>_sp.json metadata
  - optional per-channel NIfTI files for visual QC when using `--write-nifti`

The implementation intentionally avoids SimpleITK/nibabel so it can run on the
local minimal Anaconda environment. It reads/writes standard single-file NIfTI-1
(.nii/.nii.gz) images directly and uses scipy for the EDT.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any


_DLL_DIR_HANDLES = []
if os.name == "nt":
    py_root = Path(sys.executable).resolve().parent
    for dll_dir in (py_root / "Library" / "bin", py_root, py_root / "Scripts"):
        if dll_dir.is_dir():
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402


EPS = np.float32(1e-6)

_DTYPES: dict[int, Any] = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
    1280: np.uint64,
}


def _read_bytes(path: Path) -> bytes:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def _open_write(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "wb")
    return path.open("wb")


def _endian(header: bytes) -> str:
    if len(header) < 348:
        raise ValueError("File is too small to be a NIfTI-1 image.")
    if struct.unpack("<i", header[:4])[0] == 348:
        return "<"
    if struct.unpack(">i", header[:4])[0] == 348:
        return ">"
    raise ValueError("Unsupported NIfTI header: sizeof_hdr is not 348.")


def _affine_from_header(header: bytes, endian: str) -> np.ndarray:
    pixdim = np.asarray(struct.unpack(endian + "8f", header[76:108]), dtype=np.float32)
    spacing = np.asarray([abs(pixdim[1]), abs(pixdim[2]), abs(pixdim[3])], dtype=np.float32)
    qfac = -1.0 if pixdim[0] < 0 else 1.0

    qform_code = struct.unpack(endian + "h", header[252:254])[0]
    sform_code = struct.unpack(endian + "h", header[254:256])[0]
    if sform_code > 0:
        sx = struct.unpack(endian + "4f", header[280:296])
        sy = struct.unpack(endian + "4f", header[296:312])
        sz = struct.unpack(endian + "4f", header[312:328])
        return np.asarray([sx, sy, sz, (0.0, 0.0, 0.0, 1.0)], dtype=np.float32)

    if qform_code > 0:
        b = float(struct.unpack(endian + "f", header[256:260])[0])
        c = float(struct.unpack(endian + "f", header[260:264])[0])
        d = float(struct.unpack(endian + "f", header[264:268])[0])
        aa = 1.0 - (b * b + c * c + d * d)
        if aa < 1e-7:
            scale = math.sqrt(max(b * b + c * c + d * d, 1e-12))
            b, c, d = b / scale, c / scale, d / scale
            a = 0.0
        else:
            a = math.sqrt(aa)
        rotation = np.asarray(
            [
                [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
                [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b],
            ],
            dtype=np.float32,
        )
        affine = np.eye(4, dtype=np.float32)
        affine[:3, :3] = rotation @ np.diag(
            [float(spacing[0]), float(spacing[1]), float(spacing[2]) * qfac]
        )
        affine[0, 3] = struct.unpack(endian + "f", header[268:272])[0]
        affine[1, 3] = struct.unpack(endian + "f", header[272:276])[0]
        affine[2, 3] = struct.unpack(endian + "f", header[276:280])[0]
        return affine

    # Last-resort Analyze-style fallback when neither sform nor qform exists.
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2] * qfac
    affine[0, 3] = struct.unpack(endian + "f", header[268:272])[0]
    affine[1, 3] = struct.unpack(endian + "f", header[272:276])[0]
    affine[2, 3] = struct.unpack(endian + "f", header[276:280])[0]
    return affine


def _read_nifti(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    blob = _read_bytes(path)
    header = bytearray(blob[:348])
    endian = _endian(header)

    dim = struct.unpack(endian + "8h", header[40:56])
    ndim = int(dim[0])
    if ndim != 3:
        raise ValueError(f"Only 3D NIfTI images are supported, got ndim={ndim}: {path}")
    x, y, z = (int(dim[1]), int(dim[2]), int(dim[3]))
    nvox = x * y * z

    datatype = struct.unpack(endian + "h", header[70:72])[0]
    if datatype not in _DTYPES:
        raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")
    dtype = np.dtype(_DTYPES[datatype]).newbyteorder(endian)

    vox_offset = int(round(struct.unpack(endian + "f", header[108:112])[0]))
    if vox_offset < 348:
        vox_offset = 352
    data = np.frombuffer(blob, dtype=dtype, count=nvox, offset=vox_offset)
    if data.size != nvox:
        raise ValueError(f"Could not read expected {nvox} voxels from {path}")
    arr = data.reshape((z, y, x), order="C")

    slope = struct.unpack(endian + "f", header[112:116])[0]
    inter = struct.unpack(endian + "f", header[116:120])[0]
    if slope not in (0.0, 1.0) and np.isfinite(slope):
        arr = arr.astype(np.float32) * float(slope) + float(inter)
    elif inter != 0.0 and np.isfinite(inter):
        arr = arr.astype(np.float32) + float(inter)
    else:
        arr = arr.astype(np.float32, copy=False)

    pixdim = np.asarray(struct.unpack(endian + "8f", header[76:108]), dtype=np.float32)
    spacing_xyz = np.asarray([abs(pixdim[1]), abs(pixdim[2]), abs(pixdim[3])], dtype=np.float32)
    info = {
        "header": header,
        "endian": endian,
        "shape_zyx": (z, y, x),
        "shape_xyz": (x, y, z),
        "spacing_xyz": spacing_xyz,
        "affine": _affine_from_header(header, endian),
    }
    return arr, info


def _write_nifti_like(arr_zyx: np.ndarray, ref_info: dict[str, Any], path: Path) -> None:
    arr = np.asarray(arr_zyx, dtype=np.float32, order="C")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for NIfTI write, got shape={arr.shape}")
    z, y, x = arr.shape
    endian = ref_info["endian"]
    header = bytearray(ref_info["header"])

    struct.pack_into(endian + "8h", header, 40, 3, x, y, z, 1, 1, 1, 1)
    struct.pack_into(endian + "h", header, 70, 16)  # float32
    struct.pack_into(endian + "h", header, 72, 32)
    vox_offset = int(round(struct.unpack(endian + "f", header[108:112])[0]))
    if vox_offset < 348:
        vox_offset = 352
    struct.pack_into(endian + "f", header, 108, float(vox_offset))
    struct.pack_into(endian + "f", header, 112, 1.0)
    struct.pack_into(endian + "f", header, 116, 0.0)
    struct.pack_into(endian + "i", header, 140, int(np.nanmax(arr)))
    struct.pack_into(endian + "i", header, 144, int(np.nanmin(arr)))

    out_dtype = np.dtype(np.float32).newbyteorder(endian)
    padding = b"\x00" * max(0, vox_offset - len(header))
    with _open_write(path) as f:
        f.write(header)
        f.write(padding)
        f.write(arr.astype(out_dtype, copy=False).tobytes(order="C"))


def _find_case_file(root: Path, cid: str, suffix: str | None = None) -> Path | None:
    """Find a case image/mask with common FSL project naming patterns."""
    suffixes: list[str] = []
    if suffix:
        suffixes.extend([
            f"{cid}_{suffix}.nii.gz",
            f"{cid}_{suffix}.nii",
            f"{cid}-{suffix}.nii.gz",
            f"{cid}-{suffix}.nii",
        ])
    suffixes.extend([f"{cid}.nii.gz", f"{cid}.nii"])

    for name in suffixes:
        for p in (root / cid / name, root / name):
            if p.is_file():
                return p

    patterns = [f"{cid}*.nii.gz", f"{cid}*.nii"]
    if suffix:
        patterns = [
            f"{cid}*{suffix}*.nii.gz",
            f"{cid}*{suffix}*.nii",
            f"*{cid}*{suffix}*.nii.gz",
            f"*{cid}*{suffix}*.nii",
        ] + patterns

    for sr in (root / cid, root):
        if not sr.is_dir():
            continue
        for pat in patterns:
            matches = sorted(sr.glob(pat))
            if matches:
                return matches[0]
    return None


def _case_id_from_path(path: Path) -> str:
    for part in path.name.replace(".nii.gz", "").replace(".nii", "").split("_"):
        if part.startswith("sub-"):
            return part
    return path.name.split("_")[0]


def _cases_from_wm_root(wm_root: Path, wm_suffix: str) -> list[str]:
    patterns = [
        f"sub-*_{wm_suffix}.nii.gz",
        f"sub-*_{wm_suffix}.nii",
        f"sub*_{wm_suffix}.nii.gz",
        f"sub*_{wm_suffix}.nii",
        "sub-*.nii.gz",
        "sub-*.nii",
        "sub*.nii.gz",
        "sub*.nii",
    ]
    cases: list[str] = []
    for pat in patterns:
        cases.extend(_case_id_from_path(p) for p in sorted(wm_root.glob(pat)))
        cases.extend(_case_id_from_path(p) for p in sorted(wm_root.glob(f"sub-*/{pat}")))
        cases.extend(_case_id_from_path(p) for p in sorted(wm_root.glob(f"sub*/{pat}")))
    return sorted(set(cases))


def _cases_from_splits(path: Path, fold: int | None, split_key: str) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    if fold is None:
        cases: list[str] = []
        for split in splits:
            for key in ("train", "val"):
                cases.extend(split.get(key, []))
        return sorted(set(cases))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold={fold} out of range for {path} with {len(splits)} splits")
    split = splits[fold]
    if split_key == "all":
        return sorted(set(split.get("train", []) + split.get("val", [])))
    return list(split.get(split_key, []))


def _load_cases(args: argparse.Namespace, wm_root: Path) -> list[str]:
    cases: list[str] = []
    if args.case_ids:
        cases.extend(args.case_ids)
    if args.case_ids_file:
        with Path(args.case_ids_file).open("r", encoding="utf-8") as f:
            cases.extend([line.strip() for line in f if line.strip()])
    if args.splits_json:
        cases.extend(_cases_from_splits(Path(args.splits_json), args.fold, args.split_key))
    if not cases:
        cases.extend(_cases_from_wm_root(wm_root, args.wm_suffix))
    if not cases:
        raise SystemExit("Provide --case-ids, --case-ids-file, --splits-json, or WM masks under --wm-root.")
    return sorted(set(cases))


def _make_brain_mask(reference_arr: np.ndarray | None, wm_mask: np.ndarray, nonzero_threshold: float) -> np.ndarray:
    if reference_arr is not None:
        brain = np.asarray(reference_arr > float(nonzero_threshold), dtype=bool)
        if int(brain.sum()) >= 100:
            return brain
    return np.ones_like(wm_mask, dtype=bool)


def _normalize_positive_map(values: np.ndarray, valid_mask: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    valid_values = values[valid_mask] if int(valid_mask.sum()) > 0 else values.reshape(-1)
    scale = float(np.percentile(valid_values, float(percentile)))
    scale = max(scale, 1e-6)
    norm = np.clip(values / scale, 0.0, 1.0).astype(np.float32, copy=False)
    return norm, scale


def _physical_coordinate_offsets(
    shape_zyx: tuple[int, int, int],
    affine: np.ndarray,
    centroid_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = np.indices(shape_zyx, dtype=np.float32)
    px = affine[0, 0] * x + affine[0, 1] * y + affine[0, 2] * z + affine[0, 3]
    py = affine[1, 0] * x + affine[1, 1] * y + affine[1, 2] * z + affine[1, 3]
    pz = affine[2, 0] * x + affine[2, 1] * y + affine[2, 2] * z + affine[2, 3]
    return px - centroid_xyz[0], py - centroid_xyz[1], pz - centroid_xyz[2]


def _voxel_coordinate_offsets(shape_zyx: tuple[int, int, int], centroid_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = np.indices(shape_zyx, dtype=np.float32)
    return x - centroid_zyx[2], y - centroid_zyx[1], z - centroid_zyx[0]


def _wm_edt(wm_mask: np.ndarray, spacing_xyz: np.ndarray, use_physical: bool) -> np.ndarray:
    sampling = None
    if use_physical:
        sampling = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    dist = ndimage.distance_transform_edt(~wm_mask, sampling=sampling)
    return dist.astype(np.float32, copy=False)


def generate_sp(
    wm_arr: np.ndarray,
    ref_info: dict[str, Any],
    reference_arr: np.ndarray | None,
    wm_threshold: float = 0.5,
    radius_percentile: float = 99.0,
    use_physical: bool = True,
    zero_outside_brain: bool = True,
    nonzero_threshold: float = 0.0,
    radial_map: str = "wm-edt",
    angle_encoding: str = "raw",
) -> tuple[np.ndarray, dict[str, Any]]:
    wm_mask = np.asarray(wm_arr > float(wm_threshold), dtype=bool)
    if int(wm_mask.sum()) < 10:
        raise ValueError(f"WM mask is too small after threshold={wm_threshold}: voxels={int(wm_mask.sum())}")

    brain_mask = _make_brain_mask(reference_arr, wm_mask, nonzero_threshold)
    wm_zyx = np.argwhere(wm_mask)
    centroid_zyx = wm_zyx.mean(axis=0).astype(np.float32)

    if use_physical:
        affine = np.asarray(ref_info["affine"], dtype=np.float32)
        centroid_index_xyz = np.asarray([centroid_zyx[2], centroid_zyx[1], centroid_zyx[0], 1.0], dtype=np.float32)
        centroid_xyz = (affine @ centroid_index_xyz)[:3]
        dx, dy, dz = _physical_coordinate_offsets(wm_mask.shape, affine, centroid_xyz)
        centroid_for_meta = centroid_xyz
        centroid_key = "centroid_physical_xyz"
    else:
        dx, dy, dz = _voxel_coordinate_offsets(wm_mask.shape, centroid_zyx)
        centroid_for_meta = centroid_zyx
        centroid_key = "centroid_voxel_zyx"

    radius = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)
    r_norm, radius_scale = _normalize_positive_map(radius, brain_mask, radius_percentile)
    rho_xy = np.sqrt(dx * dx + dy * dy).astype(np.float32, copy=False)
    theta = np.arctan2(dy, dx)
    phi = np.arctan2(dz, rho_xy + EPS)

    if radial_map == "wm-edt":
        radial, radial_scale = _normalize_positive_map(
            _wm_edt(wm_mask, np.asarray(ref_info["spacing_xyz"], dtype=np.float32), use_physical=use_physical),
            brain_mask,
            radius_percentile,
        )
        radial_name = "wm_edt"
    elif radial_map == "centroid":
        radial = r_norm
        radial_scale = radius_scale
        radial_name = "r"
    else:
        raise ValueError(f"Unsupported radial_map={radial_map!r}")

    if angle_encoding == "raw":
        channel_names = (radial_name, "theta", "phi")
        sp = np.stack(
            [
                radial,
                (theta / np.pi).astype(np.float32, copy=False),
                (phi / (np.pi / 2.0)).astype(np.float32, copy=False),
            ],
            axis=0,
        )
    elif angle_encoding == "sincos":
        channel_names = (radial_name, "sin_theta", "cos_theta", "sin_phi", "cos_phi")
        sp = np.stack(
            [
                radial,
                np.sin(theta).astype(np.float32, copy=False),
                np.cos(theta).astype(np.float32, copy=False),
                np.sin(phi).astype(np.float32, copy=False),
                np.cos(phi).astype(np.float32, copy=False),
            ],
            axis=0,
        )
    else:
        raise ValueError(f"Unsupported angle_encoding={angle_encoding!r}")

    if zero_outside_brain:
        sp[:, ~brain_mask] = 0.0

    meta = {
        "channels": list(channel_names),
        "shape_zyx": list(wm_mask.shape),
        "wm_voxels": int(wm_mask.sum()),
        "brain_voxels": int(brain_mask.sum()),
        "radius_percentile": float(radius_percentile),
        "radius_scale": float(radius_scale),
        "radial_map": str(radial_map),
        "radial_scale": float(radial_scale),
        "angle_encoding": str(angle_encoding),
        "use_physical": bool(use_physical),
        "zero_outside_brain": bool(zero_outside_brain),
        "centroid_voxel_zyx": [float(v) for v in centroid_zyx],
        centroid_key: [float(v) for v in centroid_for_meta],
    }
    return sp.astype(np.float32, copy=False), meta


def main() -> None:
    p = argparse.ArgumentParser(description="Generate WM-centered polar spatial-position maps.")
    p.add_argument("--wm-root", required=True, help="Directory containing WM masks.")
    p.add_argument("--out-root", required=True, help="Output directory for SP maps.")
    p.add_argument("--reference-root", default=None, help="Optional reference image root for geometry and brain/nonzero mask.")
    p.add_argument("--wm-suffix", default="WM", help="WM mask suffix, for example WM or wm_mask.")
    p.add_argument("--reference-suffix", default="INV2", help="Reference suffix, for example INV2.")
    p.add_argument("--brain-mask-root", default=None, help="Optional aligned binary brain-mask root used to zero SP outside the brain.")
    p.add_argument("--brain-mask-suffix", default="brain", help="Brain-mask suffix when --brain-mask-root is used.")
    p.add_argument("--case-ids", nargs="+", default=None)
    p.add_argument("--case-ids-file", default=None)
    p.add_argument("--splits-json", default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--split-key", choices=["train", "val", "all"], default="all")
    p.add_argument("--wm-threshold", type=float, default=0.5)
    p.add_argument("--nonzero-threshold", type=float, default=0.0)
    p.add_argument("--radius-percentile", type=float, default=99.0)
    p.add_argument("--radial-map", choices=["wm-edt", "centroid"], default="wm-edt", help="First SP channel.")
    p.add_argument("--angle-encoding", choices=["raw", "sincos"], default="raw", help="raw writes 3 maps; sincos writes 5 maps.")
    p.add_argument("--coordinate-space", choices=["physical", "voxel"], default="physical")
    p.add_argument("--keep-background", action="store_true", help="Do not zero SP channels outside the reference nonzero mask.")
    p.add_argument("--write-nifti", action="store_true", help="Also write one 3D NIfTI per SP channel for visual checks.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    wm_root = Path(args.wm_root)
    ref_root = Path(args.reference_root) if args.reference_root else None
    brain_root = Path(args.brain_mask_root) if args.brain_mask_root else None
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(args, wm_root)
    print(f"[WMPolarSP] cases={len(cases)} out={out_root}", flush=True)
    for cid in cases:
        out_npz = out_root / f"{cid}_sp.npz"
        out_json = out_root / f"{cid}_sp.json"
        if out_npz.is_file() and not args.overwrite:
            print(f"[WMPolarSP] skip existing {cid}: {out_npz}", flush=True)
            continue

        wm_path = _find_case_file(wm_root, cid, args.wm_suffix)
        if wm_path is None:
            print(f"[WMPolarSP][WARN] missing WM mask for {cid} in {wm_root}", flush=True)
            continue

        wm_arr, wm_info = _read_nifti(wm_path)
        ref_arr = None
        ref_info = wm_info
        ref_path = None
        if ref_root is not None:
            ref_path = _find_case_file(ref_root, cid, args.reference_suffix)
            if ref_path is None:
                print(f"[WMPolarSP][WARN] missing reference for {cid}; using WM geometry", flush=True)
            else:
                ref_arr, ref_info = _read_nifti(ref_path)
                if ref_arr.shape != wm_arr.shape:
                    raise ValueError(
                        f"{cid}: reference shape {ref_arr.shape} != WM shape {wm_arr.shape}. "
                        "Resample WM masks to reference space first."
                    )
                if not np.allclose(ref_info["affine"], wm_info["affine"], rtol=0.0, atol=1e-4):
                    raise ValueError(
                        f"{cid}: reference affine != WM affine. "
                        "Resample the mask to the reference grid first."
                    )

        brain_arr = None
        brain_path = None
        if brain_root is not None:
            brain_path = _find_case_file(brain_root, cid, args.brain_mask_suffix)
            if brain_path is None:
                raise FileNotFoundError(
                    f"{cid}: missing brain mask with suffix={args.brain_mask_suffix!r} under {brain_root}"
                )
            brain_arr, brain_info = _read_nifti(brain_path)
            if brain_arr.shape != wm_arr.shape:
                raise ValueError(
                    f"{cid}: brain-mask shape {brain_arr.shape} != WM shape {wm_arr.shape}"
                )
            if not np.allclose(brain_info["affine"], ref_info["affine"], rtol=0.0, atol=1e-4):
                raise ValueError(
                    f"{cid}: brain-mask affine != reference affine. "
                    "Resample the brain mask to the reference grid first."
                )

        valid_reference = brain_arr if brain_arr is not None else ref_arr

        sp, meta = generate_sp(
            wm_arr=wm_arr,
            ref_info=ref_info,
            reference_arr=valid_reference,
            wm_threshold=args.wm_threshold,
            radius_percentile=args.radius_percentile,
            use_physical=args.coordinate_space == "physical",
            zero_outside_brain=not args.keep_background,
            nonzero_threshold=args.nonzero_threshold,
            radial_map=args.radial_map,
            angle_encoding=args.angle_encoding,
        )
        meta.update({
            "case_id": cid,
            "wm_path": str(wm_path),
            "reference_path": str(ref_path) if ref_path else None,
            "brain_mask_path": str(brain_path) if brain_path else None,
        })

        channel_names = tuple(meta["channels"])
        np.savez_compressed(out_npz, sp=sp, channels=np.asarray(channel_names), meta=json.dumps(meta))
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if args.write_nifti:
            case_out = out_root / cid
            case_out.mkdir(parents=True, exist_ok=True)
            for i, name in enumerate(channel_names):
                _write_nifti_like(sp[i], ref_info, case_out / f"{cid}_sp_{name}.nii.gz")

        print(
            f"[WMPolarSP] {cid}: sp={tuple(sp.shape)} wm_voxels={meta['wm_voxels']} "
            f"radial_scale={meta['radial_scale']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
