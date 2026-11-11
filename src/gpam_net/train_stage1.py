"""Train stage-1 T1+T2+SP boundary/uncertainty model without ring auxiliary loss."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import (
    JointDetSegDataset,
    JointDetSegNNUNetPreprocessedDataset,
    infer_nnunet_channel_indices,
    find_nnunet_preprocessed_case,
    load_nnunet_preprocessed_case,
    load_sp_case,
)
from .validation import (
    _full_val as joint_full_val,
    prepare_full_val_cases,
)
from .utils import (
    dice as dice_np,
    find_image,
    read_nifti,
    remove_small_components,
    sliding_window_detseg,
    zscore,
)

from .model import JointCascadeDINOStage1
from .cached_dataset import (
    JointDetSegCachedDataset,
    build_cache,
    load_cached_full_val_cases,
)
from .utils import parse_float_list, parse_int_list, progress

try:
    from scipy.ndimage import label as _cc_label
    _HAVE_SCIPY = True
except Exception:
    _cc_label = None
    _HAVE_SCIPY = False


DEFAULT_TRAIN = [
    "sub-001", "sub-002", "sub-003", "sub-004", "sub-005",
    "sub-006", "sub-007", "sub-008", "sub-009", "sub-010",
    "sub-011", "sub-014", "sub-016", "sub-017", "sub-018",
    "sub-019", "sub-020", "sub-021", "sub-022", "sub-023",
    "sub-025", "sub-026", "sub-027", "sub-030",
]
DEFAULT_VAL = ["sub-012", "sub-013", "sub-015", "sub-024"]


def _infer_splits_json(nnunet_preprocessed_root: str | None) -> Path | None:
    if not nnunet_preprocessed_root:
        return None
    root = Path(nnunet_preprocessed_root)
    for p in (root / "splits_final.json", root.parent / "splits_final.json"):
        if p.is_file():
            return p
    return root.parent / "splits_final.json"


def _load_fold_cases(splits_json: Path, fold: int) -> tuple[list[str], list[str]]:
    if not splits_json.is_file():
        raise FileNotFoundError(f"Could not find splits_final.json: {splits_json}")
    splits = json.loads(splits_json.read_text(encoding="utf-8"))
    if not isinstance(splits, list):
        raise ValueError(f"{splits_json} must contain a list of folds")
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold={fold} is out of range for {splits_json}; available folds: 0..{len(splits) - 1}")
    split = splits[fold]
    train_cases = [str(x) for x in split.get("train", [])]
    val_cases = [str(x) for x in split.get("val", [])]
    if not train_cases or not val_cases:
        raise ValueError(f"fold={fold} in {splits_json} has empty train/val cases")
    return train_cases, val_cases


def _state_dict(model: nn.Module) -> dict:
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def _load_model_weights(model: nn.Module, path: str, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Could not load model weights from {path}: unsupported checkpoint type {type(state)}")
    if any(str(k).startswith("module.") for k in state.keys()):
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[JointCascadeStage1] loaded pretrained={path} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[JointCascadeStage1] pretrained missing keys sample={missing[:5]}", flush=True)
    if unexpected:
        print(f"[JointCascadeStage1] pretrained unexpected keys sample={unexpected[:5]}", flush=True)


def _load_model_state(model: nn.Module, state: dict, strict: bool = False) -> tuple[list[str], list[str]]:
    if any(str(k).startswith("module.") for k in state.keys()):
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def _write_like(path: Path, arr: np.ndarray, ref_img: sitk.Image, dtype=np.float32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_img = sitk.GetImageFromArray(arr.astype(dtype, copy=False))
    if out_img.GetSize() == ref_img.GetSize():
        out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, str(path))


def _parse_case_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]


def _target_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _model_sp_alpha(model: nn.Module, epoch: int) -> float | None:
    target = _target_model(model)
    if hasattr(target, "_sp_alpha"):
        return float(target._sp_alpha(epoch=epoch))
    return None


def _gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Use the last high-resolution decoder feature block before seg/ring heads."""
    target = _target_model(model)
    base = getattr(target, "base", target)
    if hasattr(base, "base"):
        base = getattr(base, "base")

    up1 = getattr(base, "up1", None)
    if isinstance(up1, nn.Module):
        # Hook the fused high-resolution decoder feature rather than the whole up block.
        fuse = getattr(up1, "fuse", None)
        if isinstance(fuse, nn.Module):
            return fuse
        return up1

    layer = getattr(base, "seg_head", None)
    if isinstance(layer, nn.Module):
        return layer
    raise RuntimeError("Could not locate a Grad-CAM target layer on the stage-1 model.")


def _gradcam_connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if _HAVE_SCIPY and _cc_label is not None:
        return _cc_label(mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.int32))
    # Fallback: treat the whole foreground as one component if scipy is unavailable.
    out = mask.astype(np.int32)
    return out, int(mask.any())


def _gradcam_metrics(
    cam: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    topk_ratio: float,
    lesion_cam_threshold: float,
    raw_cam: np.ndarray | None = None,
) -> dict[str, float]:
    gt_bin = gt > 0.5
    pred_bin = pred > 0.5
    cam = np.nan_to_num(cam.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    cam = np.clip(cam, 0.0, 1.0)
    raw = cam if raw_cam is None else np.nan_to_num(raw_cam.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    if cam.size <= 0:
        cam_top = np.zeros_like(gt_bin, dtype=bool)
        cam_top_threshold = float("nan")
    else:
        ratio = min(max(float(topk_ratio), 1e-6), 1.0)
        k = max(1, int(round(cam.size * ratio)))
        flat = cam.ravel()
        if k >= flat.size:
            top_idx = np.arange(flat.size)
        else:
            top_idx = np.argpartition(flat, flat.size - k)[flat.size - k:]
        top_flat = np.zeros(flat.shape, dtype=bool)
        top_flat[top_idx] = True
        cam_top = top_flat.reshape(cam.shape)
        cam_top_threshold = float(flat[top_idx].min()) if top_idx.size > 0 else float("nan")

    inter = float(np.logical_and(cam_top, gt_bin).sum())
    cam_sum = float(cam_top.sum())
    gt_sum = float(gt_bin.sum())
    pred_sum = float(pred_bin.sum())
    cam_gt_dice = (2.0 * inter) / max(cam_sum + gt_sum, 1.0)
    cam_gt_overlap = inter / max(gt_sum, 1.0)
    cam_fp_ratio = float(np.logical_and(cam_top, ~gt_bin).sum()) / max(cam_sum, 1.0)
    pred_dice = (2.0 * float(np.logical_and(pred_bin, gt_bin).sum())) / max(pred_sum + gt_sum, 1.0)

    lab, n_comp = _gradcam_connected_components(gt_bin)
    bins = {
        "tiny_1_5": [0, 0],
        "small_6_20": [0, 0],
        "medium_21_100": [0, 0],
        "large_gt100": [0, 0],
    }
    pointed = 0
    for lid in range(1, int(n_comp) + 1):
        lesion = lab == lid
        size = int(lesion.sum())
        if size <= 0:
            continue
        hit = float(cam[lesion].max(initial=0.0)) >= float(lesion_cam_threshold)
        pointed += int(hit)
        if size <= 5:
            key = "tiny_1_5"
        elif size <= 20:
            key = "small_6_20"
        elif size <= 100:
            key = "medium_21_100"
        else:
            key = "large_gt100"
        bins[key][0] += int(hit)
        bins[key][1] += 1

    metrics = {
        "cam_gt_dice_topk": float(cam_gt_dice),
        "cam_gt_overlap_topk": float(cam_gt_overlap),
        "cam_fp_ratio_topk": float(cam_fp_ratio),
        "pred_dice_patch": float(pred_dice),
        "lesion_pointing_recall": float(pointed / max(int(n_comp), 1)),
        "gt_lesions": float(n_comp),
        "gt_voxels": float(gt_sum),
        "pred_voxels": float(pred_sum),
        "cam_top_voxels": float(cam_sum),
        "cam_top_threshold": float(cam_top_threshold),
        "cam_norm_max": float(cam.max(initial=0.0)),
        "cam_norm_mean": float(cam.mean()) if cam.size else 0.0,
        "cam_norm_std": float(cam.std()) if cam.size else 0.0,
        "cam_norm_nonzero_ratio": float((cam > 1e-6).sum()) / max(float(cam.size), 1.0),
        "cam_raw_max": float(raw.max(initial=0.0)),
        "cam_raw_mean": float(raw.mean()) if raw.size else 0.0,
        "cam_raw_std": float(raw.std()) if raw.size else 0.0,
        "cam_raw_nonzero_ratio": float((raw > 1e-12).sum()) / max(float(raw.size), 1.0),
        "cam_degenerate": float((float(raw.max(initial=0.0)) <= 1e-12) or (float(raw.std()) <= 1e-12)),
    }
    for key, (hit, total) in bins.items():
        metrics[f"{key}_pointing_recall"] = float(hit / total) if total > 0 else float("nan")
        metrics[f"{key}_lesions"] = float(total)
    return metrics


def _centered_slices(
    center: tuple[int, int, int],
    shape: tuple[int, int, int],
    patch_size: int,
) -> tuple[slice, slice, slice]:
    slices = []
    for c, dim in zip(center, shape):
        start = int(np.clip(int(c) - patch_size // 2, 0, max(0, dim - patch_size)))
        end = min(dim, start + patch_size)
        start = max(0, end - patch_size)
        slices.append(slice(start, end))
    return tuple(slices)  # type: ignore[return-value]


def _pad_patch(arr: np.ndarray, patch_size: int, channel_first: bool = False) -> np.ndarray:
    if channel_first:
        c, z, y, x = arr.shape
        pad = ((0, 0), (0, max(0, patch_size - z)), (0, max(0, patch_size - y)), (0, max(0, patch_size - x)))
    else:
        z, y, x = arr.shape
        pad = ((0, max(0, patch_size - z)), (0, max(0, patch_size - y)), (0, max(0, patch_size - x)))
    if any(p[1] > 0 for p in pad):
        arr = np.pad(arr, pad, mode="constant")
    if channel_first:
        return arr[:, :patch_size, :patch_size, :patch_size]
    return arr[:patch_size, :patch_size, :patch_size]


def _run_gradcam_diagnostics(
    model: nn.Module,
    epoch: int,
    args,
    device: torch.device,
    sp_root: Path | None,
    sp_channels: int,
    sp_alpha: float | None,
    epoch_label: int | None = None,
) -> None:
    epoch_label = int(epoch + 1 if epoch_label is None else epoch_label)
    cases = _parse_case_list(args.gradcam_cases)
    if not cases:
        cases = list(args.val_cases[: max(1, int(args.gradcam_max_cases))])
    cases = cases[: max(1, int(args.gradcam_max_cases))]

    out_root = Path(args.gradcam_out_dir) if args.gradcam_out_dir else Path(args.save).parent / "gradcam"
    epoch_dir = out_root / f"epoch_{epoch_label:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float | str | int]] = []

    model_was_training = model.training
    model.eval()
    target = _target_model(model)
    layer = _gradcam_target_layer(model)

    for cid in cases:
        inv2_path = find_image(Path(args.inv2_root), cid, "INV2")
        flair_path = find_image(Path(args.flair_root), cid, "FLAIR")
        gt_path = find_image(Path(args.labels_root), cid, "seg")
        if inv2_path is None or gt_path is None:
            print(f"[GradCAM][WARN] {cid}: missing INV2 or label, skipped", flush=True)
            continue

        inv2_raw, ref_img = read_nifti(inv2_path)
        flair_raw = read_nifti(flair_path)[0] if flair_path is not None else np.zeros_like(inv2_raw, dtype=np.float32)
        gt_raw = (read_nifti(gt_path)[0] > 0.5).astype(np.float32)
        z = min(inv2_raw.shape[0], flair_raw.shape[0], gt_raw.shape[0])
        y = min(inv2_raw.shape[1], flair_raw.shape[1], gt_raw.shape[1])
        x = min(inv2_raw.shape[2], flair_raw.shape[2], gt_raw.shape[2])
        inv2 = zscore(inv2_raw[:z, :y, :x])
        flair = zscore(flair_raw[:z, :y, :x]) if flair_path is not None else np.zeros((z, y, x), dtype=np.float32)
        gt = gt_raw[:z, :y, :x]
        sp = None
        if sp_root is not None and sp_channels > 0:
            sp = load_sp_case(
                sp_root,
                cid,
                args.sp_suffix,
                expected_shape=(z, y, x),
                channels=sp_channels,
            )

        coords = np.argwhere(gt > 0.5)
        if coords.size > 0:
            center_arr = coords.mean(axis=0).round().astype(int)
            center = (int(center_arr[0]), int(center_arr[1]), int(center_arr[2]))
        else:
            center = (z // 2, y // 2, x // 2)
        sl = _centered_slices(center, (z, y, x), int(args.patch_size))

        inv2_p = _pad_patch(inv2[sl], int(args.patch_size))
        flair_p = _pad_patch(flair[sl], int(args.patch_size))
        gt_p = _pad_patch(gt[sl], int(args.patch_size))
        sp_p = None if sp is None else _pad_patch(sp[:, sl[0], sl[1], sl[2]], int(args.patch_size), channel_first=True)

        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def _forward_hook(_module, _inp, out):
            activations.clear()
            activations.append(out if torch.is_tensor(out) else out[0])

        def _backward_hook(_module, _grad_in, grad_out):
            gradients.clear()
            gradients.append(grad_out[0])

        h1 = layer.register_forward_hook(_forward_hook)
        h2 = layer.register_full_backward_hook(_backward_hook)
        try:
            model.zero_grad(set_to_none=True)
            inv2_t = torch.from_numpy(inv2_p[None, None]).float().to(device)
            flair_t = torch.from_numpy(flair_p[None, None]).float().to(device)
            sp_t = None if sp_p is None else torch.from_numpy(sp_p[None]).float().to(device)
            m = torch.ones(1, device=device)
            with torch.enable_grad():
                out = model(inv2_t, flair_t, m, m, sp=sp_t, epoch=epoch, sp_alpha=sp_alpha)
                seg_logit = out["seg"]
                prob = torch.sigmoid(seg_logit)
                gt_t = torch.from_numpy(gt_p[None, None]).float().to(device)
                if bool(gt_t.sum() > 0):
                    lesion_logits = seg_logit[gt_t > 0.5]
                    k = max(1, int(round(float(lesion_logits.numel()) * 0.20)))
                    score = torch.topk(lesion_logits, k=min(k, lesion_logits.numel())).values.mean()
                else:
                    k = max(1, int(round(float(seg_logit.numel()) * 0.01)))
                    score = torch.topk(seg_logit.flatten(), k=min(k, seg_logit.numel())).values.mean()
                score.backward()
                # Clear gradients after backward to avoid accumulation
                target.zero_grad()

            if not activations or not gradients:
                print(f"[GradCAM][WARN] {cid}: hook did not capture activations/gradients", flush=True)
                continue
            act = activations[0].detach()
            grad = gradients[0].detach()
            weights = grad.mean(dim=(2, 3, 4), keepdim=True)
            cam_t = F.relu((weights * act).sum(dim=1, keepdim=True))
            cam_t = F.interpolate(cam_t, size=seg_logit.shape[2:], mode="trilinear", align_corners=False)
            raw_cam = cam_t[0, 0].detach().cpu().numpy().astype(np.float32)
            cam = raw_cam - float(raw_cam.min())
            cam_max = float(cam.max())
            cam = cam / cam_max if cam_max > 1e-12 else np.zeros_like(cam, dtype=np.float32)
            prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
            pred_np = (prob_np > float(args.gradcam_pred_threshold)).astype(np.uint8)

            cam_full = np.zeros((z, y, x), dtype=np.float32)
            prob_full = np.zeros((z, y, x), dtype=np.float32)
            pred_full = np.zeros((z, y, x), dtype=np.uint8)
            z_len, y_len, x_len = inv2[sl].shape
            cam_full[sl] = cam[:z_len, :y_len, :x_len]
            prob_full[sl] = prob_np[:z_len, :y_len, :x_len]
            pred_full[sl] = pred_np[:z_len, :y_len, :x_len]

            case_dir = epoch_dir / cid
            _write_like(case_dir / f"{cid}_gradcam.nii.gz", cam_full, ref_img, np.float32)
            _write_like(case_dir / f"{cid}_seg_prob_patch.nii.gz", prob_full, ref_img, np.float32)
            _write_like(case_dir / f"{cid}_pred_patch.nii.gz", pred_full, ref_img, np.uint8)

            metrics = _gradcam_metrics(
                cam[:z_len, :y_len, :x_len],
                gt[sl],
                pred_np[:z_len, :y_len, :x_len],
                topk_ratio=float(args.gradcam_topk_ratio),
                lesion_cam_threshold=float(args.gradcam_lesion_threshold),
                raw_cam=raw_cam[:z_len, :y_len, :x_len],
            )
            row: dict[str, float | str | int] = {
                "epoch": int(epoch_label),
                "epoch_index": int(epoch),
                "case": cid,
                "z0": int(sl[0].start or 0),
                "y0": int(sl[1].start or 0),
                "x0": int(sl[2].start or 0),
                "sp_alpha": float(sp_alpha) if sp_alpha is not None else float("nan"),
            }
            row.update(metrics)
            metrics_rows.append(row)
            print(
                f"[GradCAM] ep={epoch_label:03d} {cid} "
                f"cam_overlap={metrics['cam_gt_overlap_topk']:.3f} "
                f"pointing={metrics['lesion_pointing_recall']:.3f} "
                f"cam_fp={metrics['cam_fp_ratio_topk']:.3f}",
                flush=True,
            )
        finally:
            h1.remove()
            h2.remove()
            activations.clear()
            gradients.clear()

    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        for csv_name in (f"metrics_ep{epoch_label:03d}.csv", "metrics.csv"):
            with (epoch_dir / csv_name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics_rows)
    if model_was_training:
        model.train()


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not isinstance(state, dict):
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "torch_cuda_all" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception as e:
        print(f"[JointCascadeStage1][WARN] failed to restore RNG state: {e}", flush=True)


def _resize_target(target: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    if target.shape[2:] == size:
        return target
    return F.interpolate(target, size=size, mode="nearest")


def _dice_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logit)
    inter = (p * target).sum(dim=(2, 3, 4))
    denom = p.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def _modified_ce_hard_example_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Modified CE hard-example loss: LmCE = L_L1-CE + L_Err-CE."""
    if target.shape != logit.shape:
        target = _resize_target(target, logit.shape[2:])
    target = (target > 0.5).float()
    prob = torch.sigmoid(logit).clamp(min=eps, max=1.0 - eps)

    ce = -(target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob))
    l1_weight = torch.abs(target - prob)
    l_l1_ce = (l1_weight * ce).mean()

    pred = (prob >= float(threshold)).float()
    fn_mask = (target > 0.5) & (pred < 0.5)
    fp_mask = (target < 0.5) & (pred > 0.5)

    l_err = logit.sum() * 0.0
    if fn_mask.any():
        l_err = l_err + (-torch.log(prob[fn_mask])).mean()
    if fp_mask.any():
        l_err = l_err + (-torch.log(1.0 - prob[fp_mask])).mean()

    return l_l1_ce + l_err


def _tversky_loss(logit: torch.Tensor, target: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    p = torch.sigmoid(logit)
    dims = (2, 3, 4)
    tp = (p * target).sum(dim=dims)
    fp = (p * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - p) * target).sum(dim=dims)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return (1.0 - score).mean()


def _lesion_wise_dice_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    min_lesion_voxels: int = 2,
    local_margin: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    if not _HAVE_SCIPY or _cc_label is None:
        return logit.sum() * 0.0

    prob = torch.sigmoid(logit)
    if target.ndim == 4:
        target = target.unsqueeze(1)
    target = (target > 0.5).float()
    if target.shape != prob.shape:
        target = F.interpolate(target, size=prob.shape[2:], mode="nearest")

    structure = np.ones((3, 3, 3), dtype=np.int32)
    per_case_losses = []
    for b in range(prob.shape[0]):
        tgt_np = target[b, 0].detach().cpu().numpy().astype(np.uint8)
        if int(tgt_np.sum()) == 0:
            continue
        labels_np, n_comp = _cc_label(tgt_np, structure=structure)
        if n_comp == 0:
            continue
        sizes = np.bincount(labels_np.ravel())
        valid_ids = [int(i) for i in range(1, n_comp + 1) if sizes[i] >= min_lesion_voxels]
        if not valid_ids:
            continue

        labels_t = torch.from_numpy(labels_np).to(device=prob.device)
        prob_case = prob[b, 0]
        z_max, y_max, x_max = prob_case.shape
        lesion_losses = []
        for lid in valid_ids:
            coords = np.argwhere(labels_np == lid)
            z0, y0, x0 = coords.min(axis=0)
            z1, y1, x1 = coords.max(axis=0) + 1
            z0 = max(0, int(z0) - local_margin)
            y0 = max(0, int(y0) - local_margin)
            x0 = max(0, int(x0) - local_margin)
            z1 = min(z_max, int(z1) + local_margin)
            y1 = min(y_max, int(y1) + local_margin)
            x1 = min(x_max, int(x1) + local_margin)

            pred_roi = prob_case[z0:z1, y0:y1, x0:x1]
            lesion_roi = (labels_t[z0:z1, y0:y1, x0:x1] == lid).to(dtype=pred_roi.dtype)
            inter = (pred_roi * lesion_roi).sum()
            denom = pred_roi.sum() + lesion_roi.sum()
            dice = (2.0 * inter + smooth) / (denom + smooth)
            lesion_losses.append(1.0 - dice)

        if lesion_losses:
            per_case_losses.append(torch.stack(lesion_losses).mean())

    if not per_case_losses:
        return prob.sum() * 0.0
    return torch.stack(per_case_losses).mean()


# ============================================================
# New methods: UWL + UPG (from few-shot segmentation paper)
# ============================================================

def _uncertainty_weighted_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    uncertainty_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Uncertainty-weighted loss (UWL): focus on high-uncertainty regions.

    High uncertainty (boundary, small lesions) -> high weight
    Low uncertainty (background, large lesion center) -> low weight
    """
    prob = torch.sigmoid(logit)
    if target.shape != prob.shape:
        target = _resize_target(target, prob.shape[2:])
    target = (target > 0.5).float()

    # 1. Calculate prediction uncertainty (entropy)
    entropy = -(prob * torch.log(prob + eps) + (1 - prob) * torch.log(1 - prob + eps))

    # 2. Normalize to [0, 1]
    entropy_max = entropy.view(entropy.shape[0], -1).max(dim=1, keepdim=True)[0]
    entropy_max = entropy_max.view(entropy.shape[0], 1, 1, 1, 1)
    entropy_norm = entropy / (entropy_max + eps)

    # 3. Uncertainty weight map (high entropy -> high weight)
    uncertainty_map = 1.0 + uncertainty_weight * entropy_norm

    # 4. Weighted BCE loss
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    weighted_bce = (uncertainty_map * bce).mean()

    # 5. Weighted Dice loss
    inter = (prob * target * uncertainty_map).sum(dim=(2, 3, 4))
    union = (prob * uncertainty_map).sum(dim=(2, 3, 4)) + (target * uncertainty_map).sum(dim=(2, 3, 4))
    dice_loss = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()

    return weighted_bce + dice_loss


def _prototype_based_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    features: torch.Tensor,
    num_local_prototypes: int = 64,
    patch_size: int = 8,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    UPG-inspired prototype-based loss: global + local prototypes.

    For MS lesions:
    - Global prototype: captures overall lesion features
    - Local prototypes: captures small lesion patterns
    - Uncertainty filtering: keeps reliable prototypes only
    """
    if not _HAVE_SCIPY or _cc_label is None:
        return logit.sum() * 0.0

    prob = torch.sigmoid(logit)
    if target.ndim == 4:
        target = target.unsqueeze(1)
    target = (target > 0.5).float()
    if target.shape != prob.shape:
        target = F.interpolate(target, size=prob.shape[2:], mode="nearest")
    if features.shape[2:] != prob.shape[2:]:
        features = F.interpolate(features, size=prob.shape[2:], mode="trilinear", align_corners=False)

    structure = np.ones((3, 3, 3), dtype=np.int32)
    per_case_losses = []

    for b in range(prob.shape[0]):
        tgt_np = target[b, 0].detach().cpu().numpy().astype(np.uint8)
        if int(tgt_np.sum()) == 0:
            continue

        labels_np, n_comp = _cc_label(tgt_np, structure=structure)
        if n_comp == 0:
            continue

        feat_case = features[b]  # [C, D, H, W]
        prob_case = prob[b, 0]

        # 1. Extract global prototype (average of all lesion features)
        lesion_mask = torch.from_numpy(tgt_np).to(device=feat_case.device, dtype=feat_case.dtype)
        global_proto = (feat_case * lesion_mask).sum(dim=(1, 2, 3)) / (lesion_mask.sum() + 1e-8)  # [C]

        # 2. Extract local prototypes (patch-wise)
        local_protos = []
        D, H, W = feat_case.shape[1:]

        for i in range(0, D, patch_size):
            for j in range(0, H, patch_size):
                for k in range(0, W, patch_size):
                    i_end = min(i + patch_size, D)
                    j_end = min(j + patch_size, H)
                    k_end = min(k + patch_size, W)

                    patch_feat = feat_case[:, i:i_end, j:j_end, k:k_end]
                    patch_mask = lesion_mask[i:i_end, j:j_end, k:k_end]

                    if patch_mask.sum() > 0:
                        local_proto = (patch_feat * patch_mask).sum(dim=(1, 2, 3)) / (patch_mask.sum() + 1e-8)
                        local_protos.append(local_proto)

        if not local_protos:
            continue

        # 3. Uncertainty-based filtering (keep top-k confident prototypes)
        local_protos_tensor = torch.stack(local_protos)  # [N, C]

        # Compute similarity between prototypes and global prototype
        similarity = F.cosine_similarity(local_protos_tensor, global_proto.unsqueeze(0), dim=1)

        # Keep top-k similar prototypes
        k = min(num_local_prototypes, len(local_protos))
        _, top_indices = torch.topk(similarity, k)
        filtered_protos = local_protos_tensor[top_indices]

        # 4. Compute prototype-based prediction
        all_protos = torch.cat([global_proto.unsqueeze(0), filtered_protos], dim=0)  # [1+k, C]

        # Compute similarity between features and prototypes
        feat_flat = feat_case.view(feat_case.shape[0], -1).t()  # [D*H*W, C]
        proto_sim = torch.mm(feat_flat, all_protos.t())  # [D*H*W, 1+k]
        proto_pred = proto_sim.max(dim=1)[0]  # [D*H*W]
        proto_pred = proto_pred.view(D, H, W)
        proto_pred = torch.sigmoid(proto_pred)

        # 5. Compute Dice loss between prototype prediction and target
        tgt_case = target[b, 0]
        inter = (proto_pred * tgt_case).sum()
        union = proto_pred.sum() + tgt_case.sum()
        dice = (2.0 * inter + smooth) / (union + smooth)
        per_case_losses.append(1.0 - dice)

    if not per_case_losses:
        return prob.sum() * 0.0
    return torch.stack(per_case_losses).mean()


def _seg_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    tversky_weight: float,
    lesion_weight: float,
    hard_mce_weight: float = 0.0,
    hard_mce_threshold: float = 0.5,
    uncertainty_weight: float = 0.0,
    prototype_weight: float = 0.0,
    num_local_prototypes: int = 64,
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = (
        _dice_loss(logit, target)
        + tversky_weight * _tversky_loss(logit, target, alpha, beta)
    )

    if lesion_weight > 0:
        loss = loss + lesion_weight * _lesion_wise_dice_loss(
            logit, target, min_lesion_voxels=2
        )

    if hard_mce_weight > 0:
        loss = loss + hard_mce_weight * _modified_ce_hard_example_loss(
            logit,
            target,
            threshold=hard_mce_threshold,
        )

    # New: Uncertainty-weighted loss (UWL)
    if uncertainty_weight > 0:
        loss = loss + uncertainty_weight * _uncertainty_weighted_loss(logit, target)

    # New: Prototype-based loss (UPG)
    if prototype_weight > 0 and features is not None:
        loss = loss + prototype_weight * _prototype_based_loss(
            logit, target, features, num_local_prototypes=num_local_prototypes
        )

    return loss


def _seg_ds_loss(
    seg_ds: list[torch.Tensor],
    target: torch.Tensor,
    alpha: float,
    beta: float,
    tversky_weight: float,
    lesion_weight: float,
    hard_mce_weight: float = 0.0,
    hard_mce_threshold: float = 0.5,
    uncertainty_weight: float = 0.0,
    prototype_weight: float = 0.0,
    num_local_prototypes: int = 64,
    features_ds: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    weights = [1.0, 0.4, 0.2, 0.1][:len(seg_ds)]  # More focus on main output for higher Dice
    loss = 0.0
    for i, (w, out) in enumerate(zip(weights, seg_ds)):
        tgt = _resize_target(target, out.shape[2:])
        lw = lesion_weight if i == 0 else 0.0
        mce_w = hard_mce_weight if i == 0 else 0.0
        uw = uncertainty_weight if i == 0 else 0.0  # Only main output
        pw = prototype_weight if i == 0 else 0.0    # Only main output
        feat = features_ds[i] if features_ds and i < len(features_ds) else None

        loss = loss + w * _seg_loss(
            out, tgt, alpha, beta, tversky_weight, lw,
            hard_mce_weight=mce_w,
            hard_mce_threshold=hard_mce_threshold,
            uncertainty_weight=uw,
            prototype_weight=pw,
            num_local_prototypes=num_local_prototypes,
            features=feat,
        )
    return loss / sum(weights)


def _ring_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target) + _dice_loss(logit, target)


def _uncertainty_head_loss(
    uncertainty_logit: torch.Tensor,
    seg_target: torch.Tensor,
    ring_target: torch.Tensor | None = None,
    boundary_weight: float = 0.5,
) -> torch.Tensor:
    if seg_target.shape != uncertainty_logit.shape:
        seg_target = _resize_target(seg_target, uncertainty_logit.shape[2:])
    seg_bin = (seg_target > 0.5).float()

    eroded = 1.0 - F.max_pool3d(1.0 - seg_bin, kernel_size=3, stride=1, padding=1)
    seg_boundary = (seg_bin - eroded).clamp(0.0, 1.0)
    # No auxiliary ring mask in this branch: uncertainty is supervised on lesion borders only.
    uncertain_target = seg_boundary
    weights = 1.0 + float(boundary_weight) * uncertain_target
    return F.binary_cross_entropy_with_logits(
        uncertainty_logit,
        uncertain_target,
        weight=weights,
    )


def _dice_score(logit: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logit) > 0.5).float()
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return 1.0 if denom <= 0 else float(2.0 * inter / denom)


def _full_val_nnunet_preprocessed(
    model,
    preprocessed_root: Path,
    cases: list[str],
    patch_size: int,
    step: int,
    thresholds: list[float],
    min_sizes: list[int],
    device,
    num_image_channels: int = 2,
    tta: bool = False,
    batch_size: int = 1,
    mode: str = "sweep",
    raw_threshold: float = 0.5,
    raw_min_size: int = 0,
    sp_root: Path | None = None,
    sp_suffix: str = "sp",
    sp_channels: int = 3,
    sp_alpha: float | None = None,
) -> tuple[float, float, float, int, list[float], list[float]]:
    if hasattr(model, "module"):
        model = model.module
    mode = str(mode).lower().strip()
    if mode not in {"sweep", "raw"}:
        raise ValueError(f"Unknown full-val mode: {mode}")
    if mode == "raw":
        tta = False
        thresholds = [float(raw_threshold)]
        min_sizes = [int(raw_min_size)]
    channel_indices, channel_meta = infer_nnunet_channel_indices(
        preprocessed_root,
        num_image_channels,
        preferred_channel_names=("INV2", "FLAIR"),
    )
    if channel_meta and channel_meta.get("selected_channel_names"):
        print(
            f"[JointCascadeStage1] full-val channel_order={channel_meta['selected_channel_names']} "
            f"indices={channel_indices}",
            flush=True,
        )
    elif channel_meta and channel_meta.get("warning"):
        print(
            f"[JointCascadeStage1] WARN: {channel_meta['warning']} during full-val, using leading channels {channel_indices}",
            flush=True,
        )
    data = []
    for cid in progress(cases, desc="joint full-val nnunet-pre", total=len(cases), leave=False):
        case_path = find_nnunet_preprocessed_case(preprocessed_root, cid)
        if case_path is None:
            print(f"[JointCascadeStage1] WARN: {cid} missing nnUNet preprocessed case", flush=True)
            continue
        image, gt, _ = load_nnunet_preprocessed_case(
            case_path,
            num_image_channels,
            channel_indices=channel_indices,
        )
        if image.shape[0] < 1:
            print(f"[JointCascadeStage1] WARN: {cid} has no image channels", flush=True)
            continue
        inv2 = image[0]
        flair = image[1] if image.shape[0] > 1 else np.zeros_like(inv2, dtype=np.float32)
        z = min(inv2.shape[0], flair.shape[0], gt.shape[0])
        y = min(inv2.shape[1], flair.shape[1], gt.shape[1])
        x = min(inv2.shape[2], flair.shape[2], gt.shape[2])
        vol_channels = [inv2[:z, :y, :x], flair[:z, :y, :x]]
        if sp_root is not None and sp_channels > 0:
            sp = load_sp_case(
                sp_root,
                cid,
                sp_suffix,
                expected_shape=(z, y, x),
                channels=sp_channels,
            )
            vol_channels.extend([sp[i] for i in range(sp.shape[0])])
        vol = np.stack(vol_channels, axis=0).astype(np.float32, copy=False)
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
        data.append({
            "cid": cid,
            "seg_prob": seg_prob[:z, :y, :x],
            "ring_prob": ring_prob[:z, :y, :x],
            "gt": (gt[:z, :y, :x] > 0.5).astype(np.uint8),
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
                dices.append(dice_np(pred, case["gt"]))
                recalls.append(float("nan"))
            mean_d = float(np.mean(dices))
            if mean_d > best[0]:
                best = (mean_d, float("nan"), thr, ms, dices, recalls)
    return best


def parse_args():
    p = argparse.ArgumentParser(description="Joint cascade stage 1: T1+T2 lesion mask + peri-lesion ring")
    p.add_argument("--seed", type=int, default=20261017,
                   help="Random seed for Python, NumPy, and PyTorch. Default: 20261017.")
    p.add_argument("--inv2-root", default=None)
    p.add_argument("--flair-root", default=None)
    p.add_argument("--labels-root", default=None)
    p.add_argument("--nnunet-preprocessed-root", type=str, default=None,
                   help="Optional nnUNet preprocessed folder, e.g. Dataset502/.../nnUNetPlans_3d_fullres")
    p.add_argument("--nnunet-num-image-channels", type=int, default=2,
                   help="Number of image channels before segmentation in nnUNet preprocessed arrays.")
    p.add_argument("--cache-root", type=str, default=None,
                   help="Optional disk cache root built by build_disk_cache.py. When set, raw NIfTI training reads cached arrays.")
    p.add_argument("--build-cache", action="store_true",
                   help="Build missing disk-cache cases before training. Requires raw roots.")
    p.add_argument("--cache-overwrite", action="store_true",
                   help="Overwrite existing cached arrays when --build-cache is used.")
    p.add_argument("--cache-keep-in-memory", action="store_true",
                   help="Keep cached case arrays in each worker process after first load. Faster, but duplicates RAM across workers.")
    p.add_argument("--fold", type=int, default=None,
                   help="Use fold train/val cases from nnUNet splits_final.json. Overrides --train-cases/--val-cases.")
    p.add_argument("--splits-json", type=str, default=None,
                   help="Path to nnUNet splits_final.json. Default: infer from --nnunet-preprocessed-root.")
    p.add_argument("--save", required=True)
    p.add_argument("--pretrained", type=str, default=None,
                   help="Initialize model weights from an existing checkpoint/stage1_best.pth.")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume full training state from checkpoint_latest.pth. Overrides --pretrained.")
    p.add_argument("--save-latest", type=str, default=None,
                   help="Path for full-state latest checkpoint. Default: <save_dir>/checkpoint_latest.pth")
    p.add_argument("--save-every", type=int, default=0,
                   help="Save a full checkpoint every N completed epochs as checkpoint_epoch_XXX.pth. Default 0 disables it.")
    p.add_argument("--train-cases", nargs="+", default=DEFAULT_TRAIN)
    p.add_argument("--val-cases", nargs="+", default=DEFAULT_VAL)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--samples-per-case", type=int, default=8)
    p.add_argument("--fg-ratio", type=float, default=0.75)
    p.add_argument("--center-jitter", type=int, default=12)
    p.add_argument("--patch-scales", type=str, default="0.75,1.0,1.25")
    p.add_argument("--flair-dropout", type=float, default=0.0)
    p.add_argument("--heatmap-sigma", type=float, default=3.0,
                   help="Deprecated compatibility arg; ring auxiliary head does not use heatmap targets.")
    p.add_argument("--boundary-fg-prob", type=float, default=0.45)
    p.add_argument("--det-weight", type=float, default=0.0)
    p.add_argument("--ring-weight", type=float, default=None,
                   help="Weight for peri-lesion ring auxiliary loss. Defaults to --det-weight for compatibility.")
    p.add_argument("--ring-dilate-voxels", type=int, default=5,
                   help="Create ring target as lesion expanded by this many voxels minus lesion.")
    p.add_argument("--det-ramp-epochs", type=int, default=15)
    p.add_argument("--lesion-weight", type=float, default=0.8)
    p.add_argument("--tversky-alpha", type=float, default=0.15)
    p.add_argument("--tversky-beta", type=float, default=0.85)
    p.add_argument("--tversky-weight", type=float, default=0.4)
    p.add_argument("--full-val-every", type=int, default=1)
    p.add_argument("--full-val-thresholds", default="0.50")
    p.add_argument("--full-val-min-sizes", default="10")
    p.add_argument("--full-val-step", type=int, default=96)
    p.add_argument("--full-val-batch-size", type=int, default=16,
                   help="Batch size for full-volume sliding-window inference.")
    p.add_argument("--full-val-mode", choices=["sweep", "raw"], default="raw",
                   help="sweep: search thresholds/min sizes; raw: fixed thr=0.5 min_size=10 for fair Dice comparison.")
    p.add_argument("--full-val-raw-threshold", type=float, default=0.5)
    p.add_argument("--full-val-raw-min-size", type=int, default=10)
    p.add_argument("--full-val-tta", action="store_true", help="Enable TTA during full-volume validation.")
    p.add_argument("--full-val-crop-nonzero", action="store_true",
                   help="For raw full-val, cache and infer only a nonzero foreground crop.")
    p.add_argument("--full-val-crop-margin", type=int, default=16,
                   help="Voxel margin around the raw full-val nonzero crop.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-num-workers", type=int, default=0)
    p.add_argument("--data-parallel", action="store_true")
    p.add_argument("--mamba-block", choices=["simplified", "spatial_freq"], default="simplified")
    p.add_argument("--no-attention-gates", action="store_true")
    p.add_argument("--dinov3-repo", type=str, default=None,
                   help="Local facebookresearch/dinov3 repo. If omitted, DINOv3 branch is disabled.")
    p.add_argument("--dinov3-weights", type=str, default=None,
                   help="Local DINOv3 weight file or torch.hub-compatible weight id.")
    p.add_argument("--dinov3-model", type=str, default="dinov3_vits16")
    p.add_argument("--dinov3-image-size", type=int, default=224)
    p.add_argument("--dinov3-aux-channels", type=int, default=8)
    p.add_argument("--boundary-blur-kernel", type=int, default=3,
                   help="Max-pool kernel used to widen the INV2 structural-boundary proxy.")
    p.add_argument("--uncertainty-head-weight", type=float, default=0.03,
                   help="Weight for explicit uncertainty-head supervision.")
    p.add_argument("--uncertainty-boundary-weight", type=float, default=0.5,
                   help="Extra uncertainty target weight on lesion-boundary voxels.")
    p.add_argument("--sp-root", type=str, default=None,
                   help="Optional spatial-position prior root. Enables SP when provided.")
    p.add_argument("--sp-suffix", type=str, default="sp")
    p.add_argument("--sp-channels", type=int, default=3)
    p.add_argument("--sp-fusion-mode", choices=["early", "logit_adapter"], default="early")
    p.add_argument("--sp-adapter-channels", type=int, default=16)
    p.add_argument("--sp-gate-max", type=float, default=0.15)
    p.add_argument("--sp-warmup-epochs", type=int, default=50)
    p.add_argument("--sp-rampup-epochs", type=int, default=120)
    p.add_argument("--sp-dropout", type=float, default=0.1,
                   help="Case/channel dropout probability for SP channels during training.")

    p.add_argument("--hard-mce-weight", type=float, default=0.0,
                   help="Weight for modified CE hard-example loss LmCE=L_L1-CE+L_Err-CE. Default 0 disables it.")
    p.add_argument("--hard-mce-threshold", type=float, default=0.5,
                   help="Probability threshold used to form FN/FP sets for --hard-mce-weight.")

    # New: UWL and UPG from few-shot segmentation paper
    p.add_argument("--uncertainty-weight", type=float, default=0.0,
                   help="Weight for uncertainty-weighted loss (UWL). Focuses on high-uncertainty regions (boundaries, small lesions). Default 0 disables it.")
    p.add_argument("--prototype-weight", type=float, default=0.0,
                   help="Weight for prototype-based loss (UPG). Uses global+local prototypes for small lesion detection. Default 0 disables it.")
    p.add_argument("--num-local-prototypes", type=int, default=80,
                   help="Number of local prototypes for UPG module. Default 64.")

    p.add_argument("--gradcam-every", type=int, default=0,
                   help="Save Grad-CAM diagnostics every N epochs. Default 0 disables it.")
    p.add_argument("--gradcam-cases", type=str, default="",
                   help="Comma-separated case ids for Grad-CAM diagnostics. Default: first validation cases.")
    p.add_argument("--gradcam-max-cases", type=int, default=2,
                   help="Maximum number of Grad-CAM cases per diagnostic epoch.")
    p.add_argument("--gradcam-out-dir", type=str, default="",
                   help="Output directory for Grad-CAM diagnostics. Default: <save_dir>/gradcam.")
    p.add_argument("--gradcam-topk-ratio", type=float, default=0.005,
                   help="Top CAM voxel ratio used for CAM-GT overlap metrics.")
    p.add_argument("--gradcam-lesion-threshold", type=float, default=0.30,
                   help="CAM threshold used for lesion pointing recall.")
    p.add_argument("--gradcam-pred-threshold", type=float, default=0.30,
                   help="Segmentation probability threshold used for saved Grad-CAM patch predictions.")

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[JointCascadeStage1] random_seed={args.seed}", flush=True)
    using_nnunet_preprocessed = bool(args.nnunet_preprocessed_root)
    using_disk_cache = bool(args.cache_root)
    if using_nnunet_preprocessed and using_disk_cache:
        raise SystemExit("--cache-root is only supported for raw NIfTI mode, not --nnunet-preprocessed-root")
    splits_json = None
    if args.fold is not None:
        splits_json = Path(args.splits_json) if args.splits_json else _infer_splits_json(args.nnunet_preprocessed_root)
        if splits_json is None:
            raise SystemExit("--fold requires --splits-json or --nnunet-preprocessed-root")
        args.train_cases, args.val_cases = _load_fold_cases(splits_json, args.fold)
        args.splits_json = str(splits_json)
        if "{fold}" in args.save:
            args.save = args.save.format(fold=args.fold)
    raw_roots_missing = not args.inv2_root or not args.flair_root or not args.labels_root
    if not using_nnunet_preprocessed and not using_disk_cache and raw_roots_missing:
        raise SystemExit("--inv2-root, --flair-root and --labels-root are required unless --nnunet-preprocessed-root is set")
    if args.build_cache and raw_roots_missing:
        raise SystemExit("--build-cache requires --inv2-root, --flair-root and --labels-root")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ps = (args.patch_size, args.patch_size, args.patch_size)
    patch_scales = tuple(float(x.strip()) for x in args.patch_scales.split(",") if x.strip())
    thresholds = parse_float_list(args.full_val_thresholds)
    min_sizes = parse_int_list(args.full_val_min_sizes)
    ring_weight = 0.0
    sp_root = Path(args.sp_root) if args.sp_root else None
    model_sp_channels = int(args.sp_channels) if sp_root is not None else 0

    if using_disk_cache and args.build_cache:
        cache_cases = sorted(set(args.train_cases) | set(args.val_cases))
        print(
            f"[JointCascadeStage1] disk-cache build start cases={len(cache_cases)} "
            f"root={args.cache_root}",
            flush=True,
        )
        written, skipped = build_cache(
            cache_root=Path(args.cache_root),
            inv2_root=Path(args.inv2_root),
            flair_root=Path(args.flair_root),
            labels_root=Path(args.labels_root),
            case_ids=cache_cases,
            ring_dilate_voxels=args.ring_dilate_voxels,
            sp_root=sp_root,
            sp_suffix=args.sp_suffix,
            sp_channels=model_sp_channels,
            overwrite=args.cache_overwrite,
        )
        print(
            f"[JointCascadeStage1] disk-cache build done written={written} skipped={skipped}",
            flush=True,
        )

    if using_nnunet_preprocessed:
        pre_root = Path(args.nnunet_preprocessed_root)
        train_ds = JointDetSegNNUNetPreprocessedDataset(
            pre_root, args.train_cases,
            patch_size=ps, samples_per_case=args.samples_per_case, fg_ratio=args.fg_ratio,
            center_jitter=args.center_jitter, augment=True, flair_dropout_prob=args.flair_dropout,
            heatmap_sigma=args.heatmap_sigma, boundary_fg_prob=args.boundary_fg_prob,
            patch_scales=patch_scales, num_image_channels=args.nnunet_num_image_channels,
            ring_dilate_voxels=args.ring_dilate_voxels,
            sp_root=sp_root, sp_suffix=args.sp_suffix, sp_channels=model_sp_channels,
        )
        val_ds = JointDetSegNNUNetPreprocessedDataset(
            pre_root, args.val_cases,
            patch_size=ps, samples_per_case=4, fg_ratio=0.8,
            center_jitter=args.center_jitter, augment=False, flair_dropout_prob=0.0,
            heatmap_sigma=args.heatmap_sigma, boundary_fg_prob=args.boundary_fg_prob,
            num_image_channels=args.nnunet_num_image_channels,
            ring_dilate_voxels=args.ring_dilate_voxels,
            sp_root=sp_root, sp_suffix=args.sp_suffix, sp_channels=model_sp_channels,
        )
    elif using_disk_cache:
        cache_root = Path(args.cache_root)
        train_ds = JointDetSegCachedDataset(
            cache_root, args.train_cases,
            patch_size=ps, samples_per_case=args.samples_per_case, fg_ratio=args.fg_ratio,
            center_jitter=args.center_jitter, augment=True, flair_dropout_prob=args.flair_dropout,
            boundary_fg_prob=args.boundary_fg_prob,
            patch_scales=patch_scales,
            sp_channels=model_sp_channels,
            keep_in_memory=args.cache_keep_in_memory,
        )
        val_ds = JointDetSegCachedDataset(
            cache_root, args.val_cases,
            patch_size=ps, samples_per_case=4, fg_ratio=0.8,
            center_jitter=args.center_jitter, augment=False, flair_dropout_prob=0.0,
            boundary_fg_prob=args.boundary_fg_prob,
            sp_channels=model_sp_channels,
            keep_in_memory=args.cache_keep_in_memory,
        )
    else:
        train_ds = JointDetSegDataset(
            Path(args.inv2_root), Path(args.flair_root), Path(args.labels_root), args.train_cases,
            patch_size=ps, samples_per_case=args.samples_per_case, fg_ratio=args.fg_ratio,
            center_jitter=args.center_jitter, augment=True, flair_dropout_prob=args.flair_dropout,
            heatmap_sigma=args.heatmap_sigma, boundary_fg_prob=args.boundary_fg_prob,
            patch_scales=patch_scales,
            ring_dilate_voxels=args.ring_dilate_voxels,
            sp_root=sp_root, sp_suffix=args.sp_suffix, sp_channels=model_sp_channels,
        )
        val_ds = JointDetSegDataset(
            Path(args.inv2_root), Path(args.flair_root), Path(args.labels_root), args.val_cases,
            patch_size=ps, samples_per_case=4, fg_ratio=0.8,
            center_jitter=args.center_jitter, augment=False, flair_dropout_prob=0.0,
            heatmap_sigma=args.heatmap_sigma, boundary_fg_prob=args.boundary_fg_prob,
            ring_dilate_voxels=args.ring_dilate_voxels,
            sp_root=sp_root, sp_suffix=args.sp_suffix, sp_channels=model_sp_channels,
        )
    pin_memory = device.type == "cuda"
    train_persistent_workers = bool(using_disk_cache and args.num_workers > 0)
    val_persistent_workers = bool(using_disk_cache and args.val_num_workers > 0)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True, persistent_workers=train_persistent_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.val_num_workers,
        pin_memory=pin_memory, persistent_workers=val_persistent_workers,
    )

    model = JointCascadeDINOStage1(
        in_channels=2,
        deep_supervision=True,
        mamba_block=args.mamba_block,
        attention_gates=not args.no_attention_gates,
        dinov3_repo=args.dinov3_repo,
        dinov3_weights=args.dinov3_weights,
        dinov3_model=args.dinov3_model,
        dinov3_image_size=args.dinov3_image_size,
        dinov3_aux_channels=args.dinov3_aux_channels,
        boundary_blur_kernel=args.boundary_blur_kernel,
        sp_channels=model_sp_channels,
        sp_dropout=args.sp_dropout if model_sp_channels > 0 else 0.0,
        sp_fusion_mode=args.sp_fusion_mode,
        sp_adapter_channels=args.sp_adapter_channels,
        sp_gate_max=args.sp_gate_max,
        sp_warmup_epochs=args.sp_warmup_epochs,
        sp_rampup_epochs=args.sp_rampup_epochs,
    ).to(device)
    if args.resume and args.pretrained:
        print("[JointCascadeStage1] --resume provided; ignoring --pretrained", flush=True)
    if args.pretrained and not args.resume:
        _load_model_weights(model, args.pretrained, device)
    print(f"[JointCascadeStage1] DINOv3 enabled={model.dinov3_enabled}", flush=True)
    if args.data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[JointCascadeStage1] DataParallel enabled on {torch.cuda.device_count()} visible GPUs", flush=True)
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path = Path(args.save_latest) if args.save_latest else save_path.parent / "checkpoint_latest.pth"
    best = -1.0
    start_epoch = 0

    input_mode = "nnunet_preprocessed" if using_nnunet_preprocessed else ("disk_cache" if using_disk_cache else "raw_nifti")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if not isinstance(state, dict):
            raise TypeError(f"--resume checkpoint has no model_state_dict/state_dict: {resume_path}")
        target_model = model.module if isinstance(model, nn.DataParallel) else model
        missing, unexpected = _load_model_state(target_model, state, strict=False)
        if missing or unexpected:
            print(
                f"[JointCascadeStage1][resume] model compat load: "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("[JointCascadeStage1][resume][WARN] optimizer_state_dict missing", flush=True)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            print("[JointCascadeStage1][resume][WARN] scheduler_state_dict missing", flush=True)
        _restore_rng_state(ckpt.get("rng_state"))
        start_epoch = int(ckpt.get("next_epoch", int(ckpt.get("epoch", -1)) + 1))
        best = float(ckpt.get("best", ckpt.get("best_score", ckpt.get("full_val_dice", -1.0))) or -1.0)
        print(
            f"[JointCascadeStage1] resumed from {resume_path} "
            f"start_epoch={start_epoch} best={best:.6f}",
            flush=True,
        )

    print(f"[JointCascadeStage1] input_mode={input_mode}", flush=True)
    print(
        f"[JointCascadeStage1] boundary_uncertainty enabled=True "
        f"boundary_blur_kernel={args.boundary_blur_kernel} "
        f"uncertainty_head_weight={args.uncertainty_head_weight} "
        f"ring_aux_enabled=False",
        flush=True,
    )
    print(
        f"[JointCascadeStage1] spatial_position enabled={sp_root is not None} "
        f"root={sp_root} channels={model_sp_channels} dropout={args.sp_dropout if model_sp_channels > 0 else 0.0} "
        f"fusion={args.sp_fusion_mode} gate_max={args.sp_gate_max} "
        f"warmup={args.sp_warmup_epochs} ramp={args.sp_rampup_epochs}",
        flush=True,
    )
    if args.fold is not None:
        print(f"[JointCascadeStage1] fold={args.fold} splits={args.splits_json}", flush=True)
    print(f"[JointCascadeStage1] train={len(train_ds.cases)} val={len(val_ds.cases)} patch={args.patch_size}", flush=True)
    print(f"[JointCascadeStage1] steps train={len(train_loader)} val={len(val_loader)}", flush=True)

    raw_full_val_cache = None
    if not using_nnunet_preprocessed and args.full_val_every > 0:
        print("[JointCascadeStage1] raw full-val cache start", flush=True)
        if using_disk_cache:
            raw_full_val_cache = load_cached_full_val_cases(
                Path(args.cache_root),
                args.val_cases,
                crop_nonzero=args.full_val_crop_nonzero,
                crop_margin=args.full_val_crop_margin,
            )
        else:
            raw_full_val_cache = prepare_full_val_cases(
                Path(args.inv2_root),
                Path(args.flair_root),
                Path(args.labels_root),
                args.val_cases,
                crop_nonzero=args.full_val_crop_nonzero,
                crop_margin=args.full_val_crop_margin,
                sp_root=sp_root,
                sp_suffix=args.sp_suffix,
                sp_channels=model_sp_channels,
            )
        voxels = sum(int(case["gt"].size) for case in raw_full_val_cache)
        print(
            f"[JointCascadeStage1] raw full-val cache done cases={len(raw_full_val_cache)} "
            f"voxels={voxels} crop_nonzero={bool(args.full_val_crop_nonzero)}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        train_losses, train_seg_d = [], []
        current_sp_alpha = _model_sp_alpha(model, epoch)

        train_iter = progress(train_loader, f"Ep {epoch:03d}/{args.epochs} train", len(train_loader), True)
        for batch in train_iter:
            inv2 = batch["inv2"].to(device, non_blocking=True)
            flair = batch["flair"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            m_inv2 = batch["m_inv2"].to(device, non_blocking=True)
            m_flair = batch["m_flair"].to(device, non_blocking=True)
            sp = batch.get("sp")
            if sp is not None:
                sp = sp.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(inv2, flair, m_inv2, m_flair, sp=sp, epoch=epoch)
            seg_logit = out["seg"]

            # Extract features for prototype-based loss (if enabled)
            features_ds = None
            if args.prototype_weight > 0:
                # Get encoder features (before decoder)
                # Assuming model has 'features' or similar output
                features_ds = out.get("features_ds", None)
                if features_ds is None and hasattr(model, 'get_features'):
                    features_ds = model.get_features(inv2, flair, m_inv2, m_flair)

            seg_loss = _seg_ds_loss(
                out.get("seg_ds", [seg_logit]),
                seg,
                args.tversky_alpha,
                args.tversky_beta,
                args.tversky_weight,
                args.lesion_weight,
                hard_mce_weight=args.hard_mce_weight,
                hard_mce_threshold=args.hard_mce_threshold,
                uncertainty_weight=args.uncertainty_weight,
                prototype_weight=args.prototype_weight,
                num_local_prototypes=args.num_local_prototypes,
                features_ds=features_ds,
            )
            loss = seg_loss
            if args.uncertainty_head_weight > 0 and "uncertainty_logit" in out:
                loss = loss + args.uncertainty_head_weight * _uncertainty_head_loss(
                    out["uncertainty_logit"],
                    seg,
                    boundary_weight=args.uncertainty_boundary_weight,
                )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_seg_d.append(_dice_score(seg_logit.detach(), seg.detach()))
            if hasattr(train_iter, "set_postfix"):
                train_iter.set_postfix(
                    loss=f"{train_losses[-1]:.4f}",
                    seg_d=f"{train_seg_d[-1]:.4f}",
                )
        scheduler.step()

        model.eval()
        val_losses, val_seg_d = [], []
        with torch.no_grad():
            val_iter = progress(val_loader, f"Ep {epoch:03d}/{args.epochs} val", len(val_loader), True)
            for batch in val_iter:
                inv2 = batch["inv2"].to(device, non_blocking=True)
                flair = batch["flair"].to(device, non_blocking=True)
                seg = batch["seg"].to(device, non_blocking=True)
                m_inv2 = batch["m_inv2"].to(device, non_blocking=True)
                m_flair = batch["m_flair"].to(device, non_blocking=True)
                sp = batch.get("sp")
                if sp is not None:
                    sp = sp.to(device, non_blocking=True)
                out = model(inv2, flair, m_inv2, m_flair, sp=sp, epoch=epoch)
                seg_logit = out["seg"]

                # Extract features for prototype-based loss (if enabled)
                features_ds = None
                if args.prototype_weight > 0:
                    features_ds = out.get("features_ds", None)
                    if features_ds is None and hasattr(model, 'get_features'):
                        features_ds = model.get_features(inv2, flair, m_inv2, m_flair)

                loss = _seg_ds_loss(
                    out.get("seg_ds", [seg_logit]),
                    seg,
                    args.tversky_alpha,
                    args.tversky_beta,
                    args.tversky_weight,
                    args.lesion_weight,
                    hard_mce_weight=args.hard_mce_weight,
                    hard_mce_threshold=args.hard_mce_threshold,
                    uncertainty_weight=args.uncertainty_weight,
                    prototype_weight=args.prototype_weight,
                    num_local_prototypes=args.num_local_prototypes,
                    features_ds=features_ds,
                )
                if args.uncertainty_head_weight > 0 and "uncertainty_logit" in out:
                    loss = loss + args.uncertainty_head_weight * _uncertainty_head_loss(
                        out["uncertainty_logit"],
                        seg,
                        boundary_weight=args.uncertainty_boundary_weight,
                    )
                val_losses.append(float(loss.item()))
                val_seg_d.append(_dice_score(seg_logit, seg))

        full_dice = float("nan")
        full_ring = float("nan")
        full_thr = float("nan")
        full_min = -1
        case_dices = []
        case_recalls = []
        if args.full_val_every > 0 and (epoch + 1) % args.full_val_every == 0:
            print(f"[JointCascadeStage1] epoch {epoch:03d} full-val start", flush=True)
            if using_nnunet_preprocessed:
                full_dice, full_ring, full_thr, full_min, case_dices, case_recalls = _full_val_nnunet_preprocessed(
                    model,
                    Path(args.nnunet_preprocessed_root),
                    args.val_cases,
                    args.patch_size,
                    args.full_val_step,
                    thresholds,
                    min_sizes,
                    device,
                    args.nnunet_num_image_channels,
                    args.full_val_tta,
                    args.full_val_batch_size,
                    args.full_val_mode,
                    args.full_val_raw_threshold,
                    args.full_val_raw_min_size,
                    sp_root=sp_root,
                    sp_suffix=args.sp_suffix,
                    sp_channels=model_sp_channels,
                    sp_alpha=current_sp_alpha,
                )
            else:
                full_dice, full_ring, full_thr, full_min, case_dices, case_recalls = joint_full_val(
                    model,
                    Path(args.inv2_root),
                    Path(args.flair_root),
                    Path(args.labels_root),
                    args.val_cases,
                    args.patch_size,
                    args.full_val_step,
                    thresholds,
                    min_sizes,
                    device,
                    tta=args.full_val_tta,
                    batch_size=args.full_val_batch_size,
                    cached_cases=raw_full_val_cache,
                    sp_alpha=current_sp_alpha,
                )
            print(f"[JointCascadeStage1] epoch {epoch:03d} full-val done", flush=True)

        if (
            args.gradcam_every > 0
            and not using_nnunet_preprocessed
            and (epoch + 1) % args.gradcam_every == 0
        ):
            epoch_label = int(epoch + 1)
            print(f"[JointCascadeStage1] epoch {epoch_label:03d} Grad-CAM start", flush=True)
            _run_gradcam_diagnostics(
                model=model,
                epoch=epoch,
                args=args,
                device=device,
                sp_root=sp_root,
                sp_channels=model_sp_channels,
                sp_alpha=current_sp_alpha,
                epoch_label=epoch_label,
            )
            print(f"[JointCascadeStage1] epoch {epoch_label:03d} Grad-CAM done", flush=True)

        score = full_dice if np.isfinite(full_dice) else float(np.mean(val_seg_d))
        tag = ""
        best_payload = {
            "epoch": epoch,
            "model_state_dict": _state_dict(model),
            "val_loss": float(np.mean(val_losses)) if val_losses else float("inf"),
            "val_seg_dice": float(np.mean(val_seg_d)) if val_seg_d else 0.0,
            "val_ring_dice": float("nan"),
            "full_val_dice": full_dice,
            "full_val_ring_metric": full_ring,
            "full_val_thr": full_thr,
            "full_val_min_size": full_min,
            "full_val_case_dices": case_dices,
            "full_val_case_recalls": case_recalls,
            "input_mode": input_mode,
            "cache_root": args.cache_root,
            "cache_keep_in_memory": bool(args.cache_keep_in_memory),
            "fold": args.fold,
            "splits_json": args.splits_json,
            "train_cases": list(args.train_cases),
            "val_cases": list(args.val_cases),
            "nnunet_preprocessed_root": args.nnunet_preprocessed_root,
            "nnunet_num_image_channels": args.nnunet_num_image_channels,
            "sp_root": str(sp_root) if sp_root is not None else None,
            "sp_suffix": args.sp_suffix,
            "sp_channels": int(model_sp_channels),
            "sp_dropout": float(args.sp_dropout if model_sp_channels > 0 else 0.0),
            "sp_fusion_mode": args.sp_fusion_mode,
            "sp_adapter_channels": int(args.sp_adapter_channels),
            "sp_gate_max": float(args.sp_gate_max),
            "sp_warmup_epochs": int(args.sp_warmup_epochs),
            "sp_rampup_epochs": int(args.sp_rampup_epochs),
            "sp_alpha": None if current_sp_alpha is None else float(current_sp_alpha),
            "boundary_blur_kernel": int(args.boundary_blur_kernel),
            "uncertainty_head_weight": float(args.uncertainty_head_weight),
            "uncertainty_boundary_weight": float(args.uncertainty_boundary_weight),
            "lesion_weight": float(args.lesion_weight),
            "tversky_alpha": float(args.tversky_alpha),
            "tversky_beta": float(args.tversky_beta),
            "tversky_weight": float(args.tversky_weight),
            "uncertainty_weight": float(args.uncertainty_weight),
            "prototype_weight": float(args.prototype_weight),
            "num_local_prototypes": int(args.num_local_prototypes),
            "pretrained": args.pretrained,
            "resume": args.resume,
            "full_val_tta": bool(args.full_val_tta),
            "full_val_batch_size": int(args.full_val_batch_size),
            "full_val_crop_nonzero": bool(args.full_val_crop_nonzero),
            "full_val_crop_margin": int(args.full_val_crop_margin),
            "det_weight": float(args.det_weight),
            "ring_weight": 0.0,
            "ring_dilate_voxels": int(args.ring_dilate_voxels),
            "aux_target": "none",
            "ring_aux_enabled": False,
            "hard_mce_weight": float(args.hard_mce_weight),
            "hard_mce_threshold": float(args.hard_mce_threshold),
            "gradcam_every": int(args.gradcam_every),
            "gradcam_cases": args.gradcam_cases,
            "gradcam_topk_ratio": float(args.gradcam_topk_ratio),
            "gradcam_lesion_threshold": float(args.gradcam_lesion_threshold),
            "gradcam_pred_threshold": float(args.gradcam_pred_threshold),
            "mamba_block": args.mamba_block,
            "attention_gates": not args.no_attention_gates,
            "dinov3_enabled": bool(model.module.dinov3_enabled if isinstance(model, nn.DataParallel) else model.dinov3_enabled),
            "dinov3_model": args.dinov3_model,
            "dinov3_aux_channels": args.dinov3_aux_channels,
            "dinov3_repo": args.dinov3_repo,
            "dinov3_weights": args.dinov3_weights,
            "dinov3_image_size": args.dinov3_image_size,
        }

        if score > best:
            best = score
            torch.save(best_payload, save_path)
            tag = " *BEST*"

        latest_payload = dict(best_payload)
        latest_payload.update({
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best": float(best),
            "best_score": float(best),
            "next_epoch": int(epoch + 1),
            "args": vars(args),
            "rng_state": _rng_state(),
        })
        torch.save(latest_payload, latest_path)

        if args.save_every > 0 and (epoch + 1) % int(args.save_every) == 0:
            periodic_path = save_path.parent / f"checkpoint_epoch_{epoch + 1:03d}.pth"
            torch.save(latest_payload, periodic_path)
            print(f"[JointCascadeStage1] saved periodic checkpoint -> {periodic_path}", flush=True)

        elapsed = time.time() - t0
        print(
            f"Ep {epoch:03d}/{args.epochs} | train_loss={np.mean(train_losses):.4f} "
            f"seg_d={np.mean(train_seg_d):.4f} | "
            f"val_loss={np.mean(val_losses):.4f} val_seg_d={np.mean(val_seg_d):.4f} "
            f"| full_val={full_dice:.4f} "
            f"full_ring={full_ring:.4f} thr={full_thr:.2f} min={full_min} | "
            f"unc_w={args.uncertainty_head_weight:.3f} "
            f"sp_a={current_sp_alpha if current_sp_alpha is not None else float('nan'):.3f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} {elapsed:.1f}s{tag}",
            flush=True,
        )

    print(f"[JointCascadeStage1] Done -> {save_path}", flush=True)
    print(f"[JointCascadeStage1] Latest checkpoint -> {latest_path}", flush=True)


if __name__ == "__main__":
    main()

