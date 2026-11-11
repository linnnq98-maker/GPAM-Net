"""Convert SPM c1/c2/c3 tissue probability maps to binary whole-brain masks.

SPM native-space outputs:
  c1*.nii = gray matter probability
  c2*.nii = white matter probability
  c3*.nii = CSF probability

Whole-brain mask:
  brain = (c1 + c2 + c3) > threshold

The script intentionally avoids nibabel/SimpleITK so it can run with the local
minimal Anaconda install.
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
from pathlib import Path
import re
import struct
import sys


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


_DTYPES = {
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


def _read_nifti_flat(path: Path) -> tuple[np.ndarray, bytearray, str, tuple[int, ...], int]:
    blob = _read_bytes(path)
    header = bytearray(blob[:348])
    endian = _endian(header)

    dim = struct.unpack(endian + "8h", header[40:56])
    ndim = int(dim[0])
    if ndim < 1 or ndim > 7:
        raise ValueError(f"Invalid NIfTI dimension count: {ndim}")
    shape = tuple(int(x) for x in dim[1 : ndim + 1])
    nvox = math.prod(shape)

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

    slope = struct.unpack(endian + "f", header[112:116])[0]
    inter = struct.unpack(endian + "f", header[116:120])[0]
    if slope not in (0.0, 1.0) and np.isfinite(slope):
        values = data.astype(np.float32) * float(slope) + float(inter)
    elif inter != 0.0 and np.isfinite(inter):
        values = data.astype(np.float32) + float(inter)
    else:
        values = data.astype(np.float32, copy=False)
    return values, header, endian, shape, vox_offset


def _write_uint8_mask(mask: np.ndarray, header: bytearray, endian: str, vox_offset: int, dst: Path) -> None:
    mask = np.asarray(mask, dtype=np.uint8)

    # Write a valid uint8 NIfTI while preserving geometry/qform/sform fields.
    struct.pack_into(endian + "h", header, 70, 2)  # datatype: uint8
    struct.pack_into(endian + "h", header, 72, 8)  # bitpix: 8
    struct.pack_into(endian + "f", header, 112, 1.0)  # scl_slope
    struct.pack_into(endian + "f", header, 116, 0.0)  # scl_inter
    struct.pack_into(endian + "i", header, 140, int(mask.max()))  # glmax
    struct.pack_into(endian + "i", header, 144, int(mask.min()))  # glmin

    padding = b"\x00" * max(0, vox_offset - len(header))
    with _open_write(dst) as f:
        f.write(header)
        f.write(padding)
        f.write(mask.tobytes(order="C"))


def _find_tissue(case_dir: Path, tissue_prefix: str) -> Path | None:
    hits = sorted(case_dir.glob(f"{tissue_prefix}*.nii"))
    if hits:
        return hits[0]
    hits = sorted(case_dir.glob(f"{tissue_prefix}*.nii.gz"))
    if hits:
        return hits[0]
    return None


def _refine_brain_mask(
    mask: np.ndarray,
    shape_zyx: tuple[int, int, int],
    fill_holes: bool = True,
    closing_iter: int = 1,
    keep_largest: bool = True,
) -> np.ndarray:
    # NIfTI stores x as the fastest-changing axis. Reshaping to [Z, Y, X]
    # preserves physical voxel neighbors for scipy morphology.
    arr = np.asarray(mask.reshape(shape_zyx, order="C") > 0)
    if fill_holes:
        arr = ndimage.binary_fill_holes(arr)
    if int(closing_iter) > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        arr = ndimage.binary_closing(arr, structure=structure, iterations=int(closing_iter))
    if keep_largest:
        lab, n_lab = ndimage.label(arr)
        if n_lab > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep_id = int(sizes.argmax())
            arr = lab == keep_id
    return arr.astype(np.uint8, copy=False).reshape(-1, order="C")


def main() -> None:
    p = argparse.ArgumentParser(description="Create whole-brain masks from SPM c1+c2+c3 probability maps.")
    p.add_argument("--spm-root", required=True, help="Directory containing one c1/c2/c3 subdirectory per case.")
    p.add_argument("--out-root", required=True, help="Output directory for binary brain masks.")
    p.add_argument(
        "--output-suffix",
        default="brain",
        help="Output suffix after the case ID. Default: brain.",
    )
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Threshold for c1+c2+c3 probability sum. Default 0.3.")
    p.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True,
                   help="Fill internal holes in the brain mask. Default: true.")
    p.add_argument("--closing-iter", type=int, default=1,
                   help="3x3x3 binary closing iterations. Default: 1.")
    p.add_argument("--keep-largest", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep only the largest connected component. Default: true.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    spm_root = Path(args.spm_root)
    out_root = Path(args.out_root)
    output_suffix = str(args.output_suffix).strip().strip("_")
    if not output_suffix:
        raise ValueError("--output-suffix must not be empty")
    if not spm_root.is_dir():
        raise FileNotFoundError(spm_root)
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        [
            path
            for path in spm_root.iterdir()
            if path.is_dir()
            and re.fullmatch(r"sub-?\d+", path.name, flags=re.IGNORECASE)
        ],
        key=lambda path: (int(re.search(r"\d+", path.name).group()), path.name),
    )
    if not case_dirs:
        raise SystemExit(f"No sub-* directories found under {spm_root}")

    complete = 0
    for case_dir in case_dirs:
        cid = case_dir.name
        out_path = out_root / f"{cid}_{output_suffix}.nii.gz"
        if out_path.is_file() and not args.overwrite:
            print(f"[SPM12Brain] skip existing {out_path}")
            complete += 1
            continue

        c1 = _find_tissue(case_dir, "c1")
        c2 = _find_tissue(case_dir, "c2")
        c3 = _find_tissue(case_dir, "c3")
        if c1 is None or c2 is None or c3 is None:
            print(f"[SPM12Brain][WARN] missing c1/c2/c3 for {cid}")
            continue

        v1, header, endian, shape, vox_offset = _read_nifti_flat(c1)
        v2, _, _, shape2, _ = _read_nifti_flat(c2)
        v3, _, _, shape3, _ = _read_nifti_flat(c3)
        if shape2 != shape or shape3 != shape:
            raise ValueError(f"{cid}: tissue map shape mismatch: c1={shape}, c2={shape2}, c3={shape3}")

        prob_sum = v1 + v2 + v3
        mask = (prob_sum > float(args.threshold)).astype(np.uint8, copy=False)
        raw_voxels = int(mask.sum())
        shape_zyx = (int(shape[2]), int(shape[1]), int(shape[0]))
        mask = _refine_brain_mask(
            mask,
            shape_zyx,
            fill_holes=bool(args.fill_holes),
            closing_iter=int(args.closing_iter),
            keep_largest=bool(args.keep_largest),
        )
        _write_uint8_mask(mask, header, endian, vox_offset, out_path)
        complete += 1
        print(
            f"[SPM12Brain] {cid}: raw_voxels={raw_voxels} voxels={int(mask.sum())} "
            f"thr={float(args.threshold):.3f} fill_holes={bool(args.fill_holes)} "
            f"closing={int(args.closing_iter)} keep_largest={bool(args.keep_largest)} -> {out_path}"
        )

    if complete != len(case_dirs):
        raise RuntimeError(f"Only generated {complete}/{len(case_dirs)} brain masks")
    print(f"[SPM12Brain] complete={complete}/{len(case_dirs)} out={out_root}")


if __name__ == "__main__":
    main()
