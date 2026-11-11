"""Utilities for joint detection + segmentation."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def progress(iterable, desc: str | None = None, total: int | None = None, leave: bool = False):
    if tqdm is None or not desc:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True, file=sys.stdout)


def find_image(root: Path, cid: str, suffix: str) -> Path | None:
    for p in [root / cid / f"{cid}_{suffix}.nii.gz", root / f"{cid}_{suffix}.nii.gz"]:
        if p.is_file():
            return p
    return None


def read_nifti(path: Path) -> tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, img


def zscore(x: np.ndarray) -> np.ndarray:
    fg = x > 0
    if int(fg.sum()) < 100:
        mean, std = float(x.mean()), float(x.std()) + 1e-6
    else:
        mean, std = float(x[fg].mean()), float(x[fg].std()) + 1e-6
    return ((x - mean) / std).astype(np.float32, copy=False)


def crop_common(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    z = min(a.shape[0] for a in arrays)
    y = min(a.shape[1] for a in arrays)
    x = min(a.shape[2] for a in arrays)
    return tuple(a[:z, :y, :x] for a in arrays)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask.astype(np.uint8)
    lab, n_lab = ndimage.label(mask > 0)
    out = mask.astype(np.uint8).copy()
    for i in range(1, n_lab + 1):
        if int((lab == i).sum()) < min_size:
            out[lab == i] = 0
    return out


def component_bboxes(mask: np.ndarray, min_size: int = 1, margin: int = 0) -> list[dict]:
    lab, n_lab = ndimage.label(mask > 0)
    boxes: list[dict] = []
    z_max, y_max, x_max = mask.shape
    for i in range(1, n_lab + 1):
        coords = np.argwhere(lab == i)
        size = int(coords.shape[0])
        if size < min_size:
            continue
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0) + 1
        boxes.append({
            "component": i,
            "size": size,
            "z0": int(max(0, z0 - margin)),
            "z1": int(min(z_max, z1 + margin)),
            "y0": int(max(0, y0 - margin)),
            "y1": int(min(y_max, y1 + margin)),
            "x0": int(max(0, x0 - margin)),
            "x1": int(min(x_max, x1 + margin)),
        })
    return boxes


def write_bboxes_csv(path: Path, cid: str, boxes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "component", "size", "z0", "z1", "y0", "y1", "x0", "x1"],
        )
        writer.writeheader()
        for box in boxes:
            row = {"case": cid}
            row.update(box)
            writer.writerow(row)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0.5
    g = gt > 0.5
    inter = int((p & g).sum())
    denom = int(p.sum() + g.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def make_center_heatmap(seg: np.ndarray, sigma: float = 3.0, min_voxels: int = 1) -> np.ndarray:
    """Create a lesion-center heatmap from connected components of a binary mask.

    This draws a compact Gaussian around each connected-component center instead of
    materializing a full-volume coordinate grid for every lesion.
    """
    seg_bin = seg > 0.5
    lab, n_lab = ndimage.label(seg_bin)
    heat = np.zeros(seg.shape, dtype=np.float32)
    if n_lab <= 0:
        return heat

    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    z_max, y_max, x_max = seg.shape

    for i in range(1, n_lab + 1):
        coords = np.argwhere(lab == i)
        if coords.shape[0] < min_voxels:
            continue
        center = coords.mean(axis=0)
        cz, cy, cx = float(center[0]), float(center[1]), float(center[2])

        z0 = max(0, int(np.floor(cz - radius)))
        z1 = min(z_max, int(np.ceil(cz + radius + 1)))
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(y_max, int(np.ceil(cy + radius + 1)))
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(x_max, int(np.ceil(cx + radius + 1)))
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue

        zz = np.arange(z0, z1, dtype=np.float32)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float32)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float32)[None, None, :]
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        blob = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
        np.maximum(heat[z0:z1, y0:y1, x0:x1], blob, out=heat[z0:z1, y0:y1, x0:x1])

    return np.clip(heat, 0.0, 1.0)


def _starts(length: int, ps: int, step: int) -> list[int]:
    starts = list(range(0, length - ps + 1, step))
    if not starts or starts[-1] + ps < length:
        starts.append(max(0, length - ps))
    return starts


def _gaussian_weight(ps: int) -> np.ndarray:
    sigma = ps / 4
    ax = np.arange(ps) - ps / 2 + 0.5
    g1 = np.exp(-ax * ax / (2 * sigma * sigma))
    return (g1[:, None, None] * g1[None, :, None] * g1[None, None, :]).astype(np.float32)


def _single_pass(
    model,
    volume: np.ndarray,
    ps: int,
    step: int,
    device,
    gauss: np.ndarray,
    batch_size: int = 1,
    sp_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _, z_dim, y_dim, x_dim = volume.shape
    seg_sum = np.zeros((z_dim, y_dim, x_dim), dtype=np.float32)
    ring_sum = np.zeros_like(seg_sum)
    weight = np.zeros_like(seg_sum)
    batch_size = max(int(batch_size), 1)
    patches: list[np.ndarray] = []
    coords: list[tuple[int, int, int]] = []

    def _flush() -> None:
        if not patches:
            return
        inp = torch.from_numpy(np.stack(patches, axis=0)).float().to(device)
        with torch.inference_mode():
            if inp.shape[1] > 2:
                out = model(inp[:, :1], inp[:, 1:2], sp=inp[:, 2:], sp_alpha=sp_alpha)
            else:
                out = model(inp)
            seg_prob = torch.sigmoid(out["seg"]).detach().cpu().numpy()
            ring_prob = torch.sigmoid(out["ring"]).detach().cpu().numpy()
        for i, (z0, y0, x0) in enumerate(coords):
            seg_sum[z0:z0 + ps, y0:y0 + ps, x0:x0 + ps] += seg_prob[i, 0] * gauss
            ring_sum[z0:z0 + ps, y0:y0 + ps, x0:x0 + ps] += ring_prob[i, 0] * gauss
            weight[z0:z0 + ps, y0:y0 + ps, x0:x0 + ps] += gauss
        patches.clear()
        coords.clear()

    for z0 in _starts(z_dim, ps, step):
        for y0 in _starts(y_dim, ps, step):
            for x0 in _starts(x_dim, ps, step):
                patches.append(volume[:, z0:z0 + ps, y0:y0 + ps, x0:x0 + ps])
                coords.append((z0, y0, x0))
                if len(patches) >= batch_size:
                    _flush()
    _flush()
    denom = np.maximum(weight, 1e-6)
    return seg_sum / denom, ring_sum / denom


def sliding_window_detseg(
    model,
    volume: np.ndarray,
    patch_size: int,
    step: int,
    device,
    tta: bool = True,
    batch_size: int = 1,
    sp_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Channel-first [C,Z,Y,X] sliding-window inference returning seg and ring probabilities."""
    if volume.ndim != 4:
        raise ValueError(f"Expected channel-first 4D volume, got {volume.shape}")
    if hasattr(model, "module"):
        model = model.module
    _, z_dim, y_dim, x_dim = volume.shape
    pad_z = max(0, patch_size - z_dim)
    pad_y = max(0, patch_size - y_dim)
    pad_x = max(0, patch_size - x_dim)
    if pad_z or pad_y or pad_x:
        volume = np.pad(volume, ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
    gauss = _gaussian_weight(patch_size)
    model.eval()
    if not tta:
        seg, ring = _single_pass(model, volume, patch_size, step, device, gauss, batch_size=batch_size, sp_alpha=sp_alpha)
    else:
        seg_acc = np.zeros(volume.shape[1:], dtype=np.float32)
        ring_acc = np.zeros_like(seg_acc)
        n_aug = 0
        for fz in (False, True):
            for fy in (False, True):
                for fx in (False, True):
                    axes = []
                    if fz:
                        axes.append(1)
                    if fy:
                        axes.append(2)
                    if fx:
                        axes.append(3)
                    aug = np.flip(volume, axis=axes).copy() if axes else volume
                    seg_p, ring_p = _single_pass(
                        model,
                        aug,
                        patch_size,
                        step,
                        device,
                        gauss,
                        batch_size=batch_size,
                        sp_alpha=sp_alpha,
                    )
                    if axes:
                        prob_axes = [a - 1 for a in axes]
                        seg_p = np.flip(seg_p, axis=prob_axes).copy()
                        ring_p = np.flip(ring_p, axis=prob_axes).copy()
                    seg_acc += seg_p
                    ring_acc += ring_p
                    n_aug += 1
        seg = seg_acc / max(n_aug, 1)
        ring = ring_acc / max(n_aug, 1)
    out_z = volume.shape[1] - pad_z if pad_z else volume.shape[1]
    out_y = volume.shape[2] - pad_y if pad_y else volume.shape[2]
    out_x = volume.shape[3] - pad_x if pad_x else volume.shape[3]
    return seg[:out_z, :out_y, :out_x], ring[:out_z, :out_y, :out_x]
