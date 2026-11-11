"""Full-volume validation helpers used by GPAM-Net training."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import load_sp_case
from .utils import (
    dice,
    find_image,
    progress,
    read_nifti,
    remove_small_components,
    sliding_window_detseg,
    zscore,
)

def prepare_full_val_cases(
    inv2_root: Path,
    flair_root: Path,
    labels_root: Path,
    cases: list[str],
    crop_nonzero: bool = False,
    crop_margin: int = 16,
    sp_root: Path | None = None,
    sp_suffix: str = "sp",
    sp_channels: int = 3,
) -> list[dict]:
    """Load and normalize raw full-volume validation inputs once."""
    data = []
    sp_root = Path(sp_root) if sp_root else None
    for cid in progress(cases, desc="joint full-val cache", total=len(cases), leave=False):
        inv2_path = find_image(inv2_root, cid, "INV2")
        flair_path = find_image(flair_root, cid, "FLAIR")
        gt_path = find_image(labels_root, cid, "seg")
        if inv2_path is None or gt_path is None:
            print(f"[JointDetSeg] WARN: {cid} missing input or label")
            continue
        inv2_raw, _ = read_nifti(inv2_path)
        flair_raw = None
        if flair_path is not None:
            flair_raw, _ = read_nifti(flair_path)
        gt, _ = read_nifti(gt_path)

        if flair_raw is None:
            flair_raw = np.zeros_like(inv2_raw, dtype=np.float32)
        z = min(inv2_raw.shape[0], flair_raw.shape[0], gt.shape[0])
        y = min(inv2_raw.shape[1], flair_raw.shape[1], gt.shape[1])
        x = min(inv2_raw.shape[2], flair_raw.shape[2], gt.shape[2])
        inv2_raw = inv2_raw[:z, :y, :x]
        flair_raw = flair_raw[:z, :y, :x]
        gt_bin = (gt[:z, :y, :x] > 0.5).astype(np.uint8)
        sp = None
        if sp_root is not None:
            sp = load_sp_case(
                sp_root,
                cid,
                sp_suffix,
                expected_shape=inv2_raw.shape,
                channels=sp_channels,
            )[:, :z, :y, :x]

        inv2 = zscore(inv2_raw)
        flair = zscore(flair_raw) if flair_path is not None else np.zeros_like(inv2, dtype=np.float32)
        bbox = None
        if crop_nonzero:
            nonzero = (np.abs(inv2_raw) > 1e-6) | (np.abs(flair_raw) > 1e-6) | (gt_bin > 0)
            coords = np.argwhere(nonzero)
            if coords.size > 0:
                margin = max(0, int(crop_margin))
                lo = np.maximum(coords.min(axis=0) - margin, 0)
                hi = np.minimum(coords.max(axis=0) + margin + 1, np.array(nonzero.shape))
                bbox = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
                inv2 = inv2[bbox]
                flair = flair[bbox]
                gt_bin = gt_bin[bbox]
                if sp is not None:
                    sp = sp[:, bbox[0], bbox[1], bbox[2]]

        vol_channels = [inv2, flair]
        if sp is not None:
            vol_channels.extend([sp[i] for i in range(sp.shape[0])])

        data.append({
            "cid": cid,
            "vol": np.stack(vol_channels, axis=0).astype(np.float32, copy=False),
            "gt": gt_bin,
            "bbox": bbox,
            "original_shape": (z, y, x),
        })
    return data


def _full_val(
    model,
    inv2_root: Path,
    flair_root: Path,
    labels_root: Path,
    cases: list[str],
    patch_size: int,
    step: int,
    thresholds: list[float],
    min_sizes: list[int],
    device,
    tta: bool = True,
    batch_size: int = 1,
    cached_cases: list[dict] | None = None,
    sp_alpha: float | None = None,
):
    input_cases = cached_cases
    if input_cases is None:
        input_cases = prepare_full_val_cases(inv2_root, flair_root, labels_root, cases)

    data = []
    for case in progress(input_cases, desc="joint full-val infer", total=len(input_cases), leave=False):
        vol = case["vol"]
        seg_prob, ring_prob = sliding_window_detseg(
            model,
            vol,
            patch_size,
            step,
            device,
            tta=tta,
            batch_size=batch_size,
            sp_alpha=sp_alpha,
        )
        z, y, x = case["gt"].shape
        data.append({
            "cid": case["cid"],
            "seg_prob": seg_prob[:z, :y, :x],
            "ring_prob": ring_prob[:z, :y, :x],
            "gt": case["gt"],
        })
    if not data:
        return 0.0, 0.0, thresholds[0], min_sizes[0], [], []

    best = (-1.0, -1.0, thresholds[0], min_sizes[0], [], [])
    for thr in thresholds:
        for ms in min_sizes:
            dices = []
            recalls = []
            for case in data:
                pred = (case["seg_prob"] > thr).astype(np.uint8)
                pred = remove_small_components(pred, ms)
                dices.append(dice(pred, case["gt"]))
                recalls.append(float("nan"))
            mean_d = float(np.mean(dices))
            if mean_d > best[0]:
                best = (mean_d, float("nan"), thr, ms, dices, recalls)
    return best



