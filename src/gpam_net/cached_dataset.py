"""Disk-cache dataset for SP no-ring stage-1 training.

This module keeps the model/loss/sampling behavior identical to the raw NIfTI
dataset, but moves repeated image loading and preprocessing into a one-time
cache-building step.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import (
    _augment,
    _boundary_mask,
    _crop_to_common_shape,
    _label_components,
    _patchify_case,
    _ring_mask,
    _sample_component_index,
    _sample_voxel,
    find_sp_case,
    load_sp_case,
)
from .utils import find_image, read_nifti, zscore


def _find_seg(root: Path, cid: str) -> Path | None:
    for p in (
        root / cid / f"{cid}_seg.nii.gz",
        root / f"{cid}_seg.nii.gz",
        root / cid / f"{cid}_mask.nii.gz",
        root / f"{cid}_mask.nii.gz",
    ):
        if p.is_file():
            return p
    return None


def _cache_paths(cache_root: Path, cid: str) -> dict[str, Path]:
    case_dir = Path(cache_root) / cid
    return {
        "dir": case_dir,
        "arrays": case_dir / f"{cid}_arrays.npz",
        "meta": case_dir / f"{cid}_meta.pkl",
    }


def build_case_cache(
    cache_root: Path,
    inv2_root: Path,
    flair_root: Path,
    labels_root: Path,
    cid: str,
    ring_dilate_voxels: int = 5,
    sp_root: Path | None = None,
    sp_suffix: str = "sp",
    sp_channels: int = 3,
    overwrite: bool = False,
) -> bool:
    """Build one cached case. Returns True when a file was written."""
    paths = _cache_paths(cache_root, cid)
    if paths["arrays"].is_file() and paths["meta"].is_file() and not overwrite:
        return False

    inv2_path = find_image(inv2_root, cid, "INV2")
    flair_path = find_image(flair_root, cid, "FLAIR")
    seg_path = _find_seg(labels_root, cid)
    if inv2_path is None or seg_path is None:
        raise FileNotFoundError(f"{cid}: missing INV2 or label")

    inv2_raw, _ = read_nifti(inv2_path)
    flair_raw = read_nifti(flair_path)[0] if flair_path is not None else None
    seg_raw = read_nifti(seg_path)[0]

    inv2 = zscore(inv2_raw)
    flair = zscore(flair_raw) if flair_raw is not None else None
    seg = (seg_raw > 0.5).astype(np.float32)

    sp = None
    sp_path = None
    if sp_root is not None and int(sp_channels) > 0:
        sp_path = find_sp_case(sp_root, cid, sp_suffix)
        if sp_path is None:
            raise FileNotFoundError(f"{cid}: missing SP in {sp_root} suffix={sp_suffix!r}")
        sp = load_sp_case(sp_path.parent, cid, sp_suffix, expected_shape=inv2.shape, channels=sp_channels)

    if flair is None:
        inv2, seg, sp = _crop_to_common_shape(inv2, seg, sp)
        flair = np.zeros_like(inv2, dtype=np.float32)
    else:
        inv2, flair, seg, sp = _crop_to_common_shape(inv2, flair, seg, sp)

    boundary = _boundary_mask(seg).astype(np.uint8)
    ring = _ring_mask(seg, ring_dilate_voxels).astype(np.float32)

    paths["dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "inv2": inv2.astype(np.float32, copy=False),
        "flair": flair.astype(np.float32, copy=False),
        "seg": seg.astype(np.float32, copy=False),
        "ring": ring.astype(np.float32, copy=False),
        "boundary": boundary,
    }
    if sp is not None:
        payload["sp"] = sp.astype(np.float32, copy=False)
    # Keep the cache uncompressed: these files are larger, but much faster to
    # load repeatedly during training than gzipped NIfTI or compressed npz.
    np.savez(paths["arrays"], **payload)

    meta = {
        "case_id": cid,
        "shape": tuple(int(v) for v in inv2.shape),
        "inv2_path": str(inv2_path),
        "flair_path": str(flair_path) if flair_path is not None else None,
        "seg_path": str(seg_path),
        "sp_path": str(sp_path) if sp_path is not None else None,
        "sp_channels": int(sp.shape[0]) if sp is not None else 0,
        "ring_dilate_voxels": int(ring_dilate_voxels),
    }
    with paths["meta"].open("wb") as f:
        pickle.dump(meta, f)
    return True


def build_cache(
    cache_root: Path,
    inv2_root: Path,
    flair_root: Path,
    labels_root: Path,
    case_ids: list[str],
    ring_dilate_voxels: int = 5,
    sp_root: Path | None = None,
    sp_suffix: str = "sp",
    sp_channels: int = 3,
    overwrite: bool = False,
) -> tuple[int, int]:
    written = 0
    skipped = 0
    for i, cid in enumerate(case_ids, 1):
        did_write = build_case_cache(
            cache_root=cache_root,
            inv2_root=inv2_root,
            flair_root=flair_root,
            labels_root=labels_root,
            cid=cid,
            ring_dilate_voxels=ring_dilate_voxels,
            sp_root=sp_root,
            sp_suffix=sp_suffix,
            sp_channels=sp_channels,
            overwrite=overwrite,
        )
        written += int(did_write)
        skipped += int(not did_write)
        print(f"[DiskCache] {i}/{len(case_ids)} {cid} {'written' if did_write else 'skip'}", flush=True)
    return written, skipped


class JointDetSegCachedDataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        case_ids: List[str],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        samples_per_case: int = 8,
        fg_ratio: float = 0.6,
        center_jitter: int = 16,
        augment: bool = True,
        flair_dropout_prob: float = 0.5,
        boundary_fg_prob: float = 0.35,
        patch_scales: Tuple[float, ...] = (1.0,),
        sp_channels: int = 3,
        keep_in_memory: bool = False,
    ):
        self.cache_root = Path(cache_root)
        self.patch_size = patch_size
        self.samples_per_case = samples_per_case
        self.fg_ratio = fg_ratio
        self.center_jitter = center_jitter
        self.augment = augment
        self.flair_dropout_prob = flair_dropout_prob
        self.boundary_fg_prob = boundary_fg_prob
        self.sp_channels = int(sp_channels)
        self.keep_in_memory = bool(keep_in_memory)
        self.cache_max_cases = None if self.keep_in_memory else 64
        self.patch_scales = tuple(float(s) for s in patch_scales if float(s) > 0)
        if not self.patch_scales:
            self.patch_scales = (1.0,)

        self.cases: list[tuple[Path, str]] = []
        for cid in case_ids:
            paths = _cache_paths(self.cache_root, cid)
            if not paths["arrays"].is_file():
                print(f"[JointDetSegCachedDataset] WARN: {cid} missing cache, skipped", flush=True)
                continue
            self.cases.append((paths["arrays"], cid))
        if not self.cases:
            raise RuntimeError("JointDetSegCachedDataset has 0 valid cases")
        self._cache: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_case

    def _load(self, case_idx: int):
        if case_idx in self._cache:
            return self._cache[case_idx]
        arrays_path, cid = self.cases[case_idx]
        with np.load(arrays_path, allow_pickle=False) as f:
            inv2 = f["inv2"].astype(np.float32, copy=False)
            flair = f["flair"].astype(np.float32, copy=False)
            seg = f["seg"].astype(np.float32, copy=False)
            ring = f["ring"].astype(np.float32, copy=False)
            boundary = f["boundary"].astype(bool, copy=False)
            sp = f["sp"].astype(np.float32, copy=False) if "sp" in f.files else None
        if sp is not None and self.sp_channels > 0:
            sp = sp[: self.sp_channels]
        comps = _label_components(seg)
        loaded = (inv2, flair, seg, ring, sp, comps, boundary, cid)
        if self.keep_in_memory or (self.cache_max_cases is not None and self.cache_max_cases > 0):
            if (
                self.cache_max_cases is not None
                and len(self._cache) >= self.cache_max_cases
                and case_idx not in self._cache
            ):
                self._cache.pop(next(iter(self._cache)))
            self._cache[case_idx] = loaded
        return loaded

    def _sample_center(self, comps: list[np.ndarray], shape: tuple[int, int, int]) -> tuple[int, int, int]:
        if np.random.rand() < self.fg_ratio and len(comps) > 0:
            comp = comps[_sample_component_index(comps)]
            cz, cy, cx = _sample_voxel(comp)
        else:
            z_dim, y_dim, x_dim = shape
            cz, cy, cx = np.random.randint(z_dim), np.random.randint(y_dim), np.random.randint(x_dim)
        if self.center_jitter > 0:
            j = self.center_jitter
            cz += np.random.randint(-j, j + 1)
            cy += np.random.randint(-j, j + 1)
            cx += np.random.randint(-j, j + 1)
        return int(cz), int(cy), int(cx)

    def __getitem__(self, idx: int) -> dict:
        case_idx = idx % len(self.cases)
        inv2, flair, seg, ring, sp, comps, boundary, cid = self._load(case_idx)
        use_flair = np.random.rand() >= self.flair_dropout_prob
        if not use_flair:
            flair = np.zeros_like(inv2)
            m_flair = 0.0
        else:
            m_flair = 1.0

        if np.random.rand() < self.fg_ratio and len(comps) > 0:
            comp = comps[_sample_component_index(comps)]
            if boundary.any() and np.random.rand() < self.boundary_fg_prob:
                center = _sample_voxel(np.argwhere(boundary))
            else:
                center = _sample_voxel(comp)
            if self.center_jitter > 0:
                j = self.center_jitter
                center = (
                    int(center[0] + np.random.randint(-j, j + 1)),
                    int(center[1] + np.random.randint(-j, j + 1)),
                    int(center[2] + np.random.randint(-j, j + 1)),
                )
        else:
            center = self._sample_center(comps, inv2.shape)

        scale = float(self.patch_scales[np.random.randint(len(self.patch_scales))])
        crop_size = tuple(max(16, int(round(s * scale))) for s in self.patch_size)
        inv2_c, flair_c, seg_c, ring_c, sp_c = _patchify_case(
            inv2,
            flair,
            seg,
            ring,
            sp,
            center,
            crop_size,
            self.patch_size,
            self.augment,
        )
        sample = {
            "inv2": torch.from_numpy(np.ascontiguousarray(inv2_c[None])).float(),
            "flair": torch.from_numpy(np.ascontiguousarray(flair_c[None])).float(),
            "seg": torch.from_numpy(np.ascontiguousarray(seg_c[None])).float(),
            "ring": torch.from_numpy(np.ascontiguousarray(ring_c[None])).float(),
            "m_inv2": torch.tensor(1.0),
            "m_flair": torch.tensor(float(m_flair)),
            "case_id": cid,
        }
        if sp_c is not None:
            sample["sp"] = torch.from_numpy(np.ascontiguousarray(sp_c)).float()
        return sample


def load_cached_full_val_cases(
    cache_root: Path,
    case_ids: list[str],
    crop_nonzero: bool = False,
    crop_margin: int = 16,
) -> list[dict]:
    data = []
    for cid in case_ids:
        paths = _cache_paths(Path(cache_root), cid)
        if not paths["arrays"].is_file():
            print(f"[DiskCache][WARN] full-val missing cache for {cid}", flush=True)
            continue
        with np.load(paths["arrays"], allow_pickle=False) as f:
            vol_channels = [f["inv2"].astype(np.float32, copy=False), f["flair"].astype(np.float32, copy=False)]
            if "sp" in f.files:
                sp = f["sp"].astype(np.float32, copy=False)
                vol_channels.extend([sp[i] for i in range(sp.shape[0])])
            gt = (f["seg"] > 0.5).astype(np.uint8)
        original_shape = tuple(int(v) for v in gt.shape)
        bbox = None
        if crop_nonzero:
            nonzero = np.zeros_like(gt, dtype=bool)
            for channel in vol_channels:
                nonzero |= np.abs(channel) > 1e-6
            nonzero |= gt > 0
            coords = np.argwhere(nonzero)
            if coords.size > 0:
                margin = max(0, int(crop_margin))
                lo = np.maximum(coords.min(axis=0) - margin, 0)
                hi = np.minimum(coords.max(axis=0) + margin + 1, np.array(gt.shape))
                bbox = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
                vol_channels = [channel[bbox] for channel in vol_channels]
                gt = gt[bbox]
        data.append({
            "cid": cid,
            "vol": np.stack(vol_channels, axis=0).astype(np.float32, copy=False),
            "gt": gt,
            "bbox": bbox,
            "original_shape": original_shape,
        })
    return data

