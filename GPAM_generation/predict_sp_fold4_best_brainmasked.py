from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage


def read_nifti(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, img


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5)
    )


def find_image(case_dir: Path, stem: str) -> Path:
    for name in (f"{stem}.nii.gz", f"{stem}.nii"):
        path = case_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {stem}.nii.gz/.nii under {case_dir}")


def find_brain_mask(brain_root: Path, case_id: str) -> Path:
    names = (
        f"{case_id}_brainmask.nii.gz",
        f"{case_id}_brainmask.nii",
        f"{case_id}_brain.nii.gz",
        f"{case_id}_brain.nii",
        f"{case_id}.nii.gz",
        f"{case_id}.nii",
    )
    for name in names:
        for path in (brain_root / name, brain_root / case_id / name):
            if path.is_file():
                return path
    raise FileNotFoundError(f"{case_id}: missing brain mask under {brain_root}")


def write_like(path: Path, arr: np.ndarray, ref_img: sitk.Image, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.GetImageFromArray(arr.astype(dtype, copy=False))
    if out.GetSize() == ref_img.GetSize():
        out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(path))


def zscore(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    m = np.isfinite(x) & (x != 0)
    if int(m.sum()) < 100:
        m = np.isfinite(x)
    if int(m.sum()) == 0:
        return np.zeros_like(x, dtype=np.float32)
    mean = float(x[m].mean())
    std = float(x[m].std())
    if std < 1e-6:
        std = 1.0
    return ((x - mean) / std).astype(np.float32, copy=False)


def crop_pair(a, b):
    shape = tuple(min(a.shape[i], b.shape[i]) for i in range(3))
    return a[:shape[0], :shape[1], :shape[2]], b[:shape[0], :shape[1], :shape[2]]


def paste_to_shape(arr, shape, fill=0):
    if tuple(arr.shape) == tuple(shape):
        return arr
    out = np.full(shape, fill, dtype=arr.dtype)
    z = min(shape[0], arr.shape[0])
    y = min(shape[1], arr.shape[1])
    x = min(shape[2], arr.shape[2])
    out[:z, :y, :x] = arr[:z, :y, :x]
    return out


def load_sp(path: Path, channels: int, expected_shape):
    with np.load(path, allow_pickle=False) as f:
        sp = f["sp"] if "sp" in f else f[list(f.keys())[0]]
    sp = np.asarray(sp, dtype=np.float32)
    if sp.ndim == 3:
        sp = sp[None]
    if sp.ndim != 4:
        raise ValueError(f"Expected SP [C,Z,Y,X], got {sp.shape}: {path}")
    if sp.shape[0] < channels:
        raise ValueError(f"SP channels {sp.shape[0]} < expected {channels}: {path}")
    sp = sp[:channels]
    if tuple(sp.shape[1:]) != tuple(expected_shape):
        raise ValueError(f"SP shape mismatch: {path} {sp.shape[1:]} vs image {expected_shape}")
    return sp


def find_sp(sp_root: Path, cid: str, suffix: str):
    names = [f"{cid}_{suffix}.npz", f"{cid}.npz"]
    for name in names:
        for p in (sp_root / name, sp_root / cid / name):
            if p.is_file():
                return p
    hits = sorted(sp_root.glob(f"{cid}*.npz"))
    if hits:
        return hits[0]
    hits = sorted((sp_root / cid).glob(f"{cid}*.npz")) if (sp_root / cid).is_dir() else []
    return hits[0] if hits else None


def strip_module_prefix(state):
    if any(str(k).startswith("module.") for k in state):
        return {str(k).removeprefix("module."): v for k, v in state.items()}
    return state


def _prepend_sys_path(path_str: str) -> None:
    p = Path(path_str)
    if not p.exists():
        return
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


def _add_packaged_src() -> None:
    # Keep the release package self-contained instead of relying on the
    # original machine-specific /data/wsc/SP paths.
    package_root = Path(__file__).resolve().parents[1]
    src_root = package_root / "src"
    if src_root.is_dir():
        _prepend_sys_path(str(src_root))


def load_model(ckpt_path: Path, device):
    _add_packaged_src()

    from gpam_net.model import JointCascadeDINOStage1
    from gpam_net.mamba_blocks import _HAS_MAMBA

    if not _HAS_MAMBA:
        raise RuntimeError(
            "mamba_ssm is unavailable. Refusing to load the checkpoint with "
            "the lightweight fallback because that would change the trained model."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state is None:
        raise KeyError(f"No model_state_dict/state_dict in {ckpt_path}")
    state = strip_module_prefix(state)

    candidates = dict(
        in_channels=2,
        deep_supervision=True,
        mamba_block=str(ckpt.get("mamba_block", "simplified")),
        attention_gates=bool(ckpt.get("attention_gates", True)),
        dinov3_repo=ckpt.get("dinov3_repo") if ckpt.get("dinov3_enabled", False) else None,
        dinov3_weights=ckpt.get("dinov3_weights") if ckpt.get("dinov3_enabled", False) else None,
        dinov3_model=str(ckpt.get("dinov3_model", "dinov3_vits16")),
        dinov3_image_size=int(ckpt.get("dinov3_image_size", 224)),
        dinov3_aux_channels=int(ckpt.get("dinov3_aux_channels", 8)),
        boundary_blur_kernel=int(ckpt.get("boundary_blur_kernel", 3) or 3),
        sp_channels=int(ckpt.get("sp_channels", 3) or 3),
        sp_dropout=float(ckpt.get("sp_dropout", 0.0) or 0.0),
        sp_fusion_mode=str(ckpt.get("sp_fusion_mode", "early")),
        sp_adapter_channels=int(ckpt.get("sp_adapter_channels", 16) or 16),
        sp_gate_max=float(ckpt.get("sp_gate_max", 0.15) or 0.15),
        sp_warmup_epochs=int(ckpt.get("sp_warmup_epochs", 0) or 0),
        sp_rampup_epochs=int(ckpt.get("sp_rampup_epochs", 1) or 1),
    )

    sig = inspect.signature(JointCascadeDINOStage1.__init__)
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
    model = JointCascadeDINOStage1(**kwargs).to(device)

    incompatible = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")
    if incompatible.missing_keys:
        print("[load] missing sample:", incompatible.missing_keys[:8])
    if incompatible.unexpected_keys:
        print("[load] unexpected sample:", incompatible.unexpected_keys[:8])
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch. Use the packaged source with the same "
            "torch, CUDA, and mamba_ssm versions as training."
        )

    model.eval()
    return model, ckpt


def starts(size, patch, step):
    if size <= patch:
        return [0]
    s = list(range(0, size - patch + 1, step))
    if s[-1] != size - patch:
        s.append(size - patch)
    return s


def gaussian_weight(ps):
    sigma = ps / 4
    ax = np.arange(ps) - ps / 2 + 0.5
    g = np.exp(-ax * ax / (2 * sigma * sigma))
    return (g[:, None, None] * g[None, :, None] * g[None, None, :]).astype(np.float32)


def model_forward(model, inp, sp_alpha=1.0):
    if inp.shape[1] > 2:
        try:
            out = model(inp[:, :1], inp[:, 1:2], sp=inp[:, 2:], sp_alpha=sp_alpha)
        except TypeError as e:
            if "sp_alpha" in str(e):
                out = model(inp[:, :1], inp[:, 1:2], sp=inp[:, 2:])
            else:
                raise
    else:
        out = model(inp)

    if isinstance(out, dict):
        seg = out["seg"]
        # The packaged model returns uncertainty after sigmoid.
        ambiguity = out.get("uncertainty", torch.zeros_like(seg))
        ambiguity_is_probability = True
    elif isinstance(out, (tuple, list)):
        seg = out[0]
        ambiguity = out[1] if len(out) > 1 else torch.zeros_like(seg)
        ambiguity_is_probability = False
    else:
        seg = out
        ambiguity = torch.zeros_like(seg)
        ambiguity_is_probability = False
    return seg, ambiguity, ambiguity_is_probability


def sliding_window(model, volume, patch_size, step, device, batch_size=1, sp_alpha=1.0):
    if volume.ndim != 4:
        raise ValueError(f"Expected [C,Z,Y,X], got {volume.shape}")

    c, z0, y0, x0 = volume.shape
    pz = max(0, patch_size - z0)
    py = max(0, patch_size - y0)
    px = max(0, patch_size - x0)
    if pz or py or px:
        volume = np.pad(volume, ((0, 0), (0, pz), (0, py), (0, px)), mode="constant")

    _, z, y, x = volume.shape
    seg_sum = np.zeros((z, y, x), dtype=np.float32)
    ambiguity_sum = np.zeros_like(seg_sum)
    weight = np.zeros_like(seg_sum)
    gauss = gaussian_weight(patch_size)

    patches, coords = [], []

    def flush():
        nonlocal patches, coords
        if not patches:
            return
        inp = torch.from_numpy(np.stack(patches, axis=0)).float().to(device)
        with torch.inference_mode():
            seg, ambiguity, ambiguity_is_probability = model_forward(model, inp, sp_alpha=sp_alpha)
            seg_prob = torch.sigmoid(seg).detach().cpu().numpy()
            if ambiguity_is_probability:
                ambiguity_prob = ambiguity.detach().cpu().numpy()
            else:
                ambiguity_prob = torch.sigmoid(ambiguity).detach().cpu().numpy()
        for i, (zz, yy, xx) in enumerate(coords):
            seg_sum[zz:zz+patch_size, yy:yy+patch_size, xx:xx+patch_size] += seg_prob[i, 0] * gauss
            ambiguity_sum[zz:zz+patch_size, yy:yy+patch_size, xx:xx+patch_size] += ambiguity_prob[i, 0] * gauss
            weight[zz:zz+patch_size, yy:yy+patch_size, xx:xx+patch_size] += gauss
        patches, coords = [], []

    for zz in starts(z, patch_size, step):
        for yy in starts(y, patch_size, step):
            for xx in starts(x, patch_size, step):
                patches.append(volume[:, zz:zz+patch_size, yy:yy+patch_size, xx:xx+patch_size])
                coords.append((zz, yy, xx))
                if len(patches) >= batch_size:
                    flush()
    flush()

    seg = seg_sum / np.maximum(weight, 1e-6)
    ambiguity = ambiguity_sum / np.maximum(weight, 1e-6)
    return seg[:z0, :y0, :x0], ambiguity[:z0, :y0, :x0]


def remove_small(mask, min_size):
    if min_size <= 0:
        return mask.astype(np.uint8)
    lab, n = ndimage.label(mask > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[lab].astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sp-root", required=True)
    ap.add_argument("--brain-mask-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--prob-root", required=True)
    ap.add_argument("--ambiguity-root", required=True)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-size", type=int, default=10)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--step", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--sp-suffix", default="sp")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    sp_root = Path(args.sp_root)
    brain_root = Path(args.brain_mask_root)
    out_root = Path(args.out_root)
    prob_root = Path(args.prob_root)
    ambiguity_root = Path(args.ambiguity_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prob_root.mkdir(parents=True, exist_ok=True)
    ambiguity_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    model, ckpt = load_model(Path(args.ckpt), device)
    sp_channels = int(ckpt.get("sp_channels", 3) or 3)
    print("[INFO] sp_channels:", sp_channels)

    cases = sorted([p for p in data_root.iterdir() if p.is_dir()])
    print("[INFO] cases:", len(cases))

    done = 0
    for case_dir in cases:
        cid = case_dir.name
        t1_path = find_image(case_dir, "T1")
        flair_path = find_image(case_dir, "FLAIR")
        brain_path = find_brain_mask(brain_root, cid)

        t1_raw, ref_img = read_nifti(t1_path)
        flair_raw, flair_img = read_nifti(flair_path)
        brain_raw, brain_img = read_nifti(brain_path)
        if not same_geometry(ref_img, flair_img):
            raise ValueError(f"{cid}: T1/FLAIR geometry mismatch")
        if not same_geometry(ref_img, brain_img):
            raise ValueError(f"{cid}: T1/brain-mask geometry mismatch: {brain_path}")
        ref_shape = tuple(t1_raw.shape)

        t1, flair = crop_pair(t1_raw, flair_raw)
        if tuple(brain_raw.shape) != tuple(t1.shape):
            raise ValueError(f"{cid}: brain-mask shape {brain_raw.shape} != image shape {t1.shape}")
        brainmask = brain_raw > 0.5
        if int(brainmask.sum()) < 100:
            raise ValueError(f"{cid}: brain mask is empty or too small: {brain_path}")

        sp_path = find_sp(sp_root, cid, args.sp_suffix)
        if sp_path is None:
            raise FileNotFoundError(f"{cid}: missing SP npz under {sp_root}")
        sp = load_sp(sp_path, sp_channels, t1.shape)

        volume = np.stack([zscore(t1), zscore(flair)] + [sp[i] for i in range(sp_channels)], axis=0).astype(np.float32)

        print(f"[RUN] {cid}: volume={volume.shape} sp={sp_path.name}")
        seg_prob, ambiguity_prob = sliding_window(
            model,
            volume,
            patch_size=args.patch_size,
            step=args.step,
            device=device,
            batch_size=args.batch_size,
            sp_alpha=1.0,
        )

        # SP is an input feature, not a hard output constraint. Explicitly clip
        # probabilities and the binary prediction to the aligned brain mask.
        seg_prob[~brainmask] = 0.0
        ambiguity_prob[~brainmask] = 0.0
        mask = ((seg_prob > args.threshold) & brainmask).astype(np.uint8)
        mask = remove_small(mask, args.min_size)

        seg_prob_full = paste_to_shape(seg_prob, ref_shape)
        ambiguity_full = paste_to_shape(ambiguity_prob, ref_shape)
        mask_full = paste_to_shape(mask, ref_shape)

        write_like(prob_root / f"{cid}_prob.nii.gz", seg_prob_full, ref_img, np.float32)
        write_like(ambiguity_root / f"{cid}_ambiguity.nii.gz", ambiguity_full, ref_img, np.float32)
        write_like(out_root / f"{cid}_mask_thr{args.threshold:.2f}_min{args.min_size}.nii.gz", mask_full, ref_img, np.uint8)

        print(
            f"[DONE] {cid}: brain_vox={int(brainmask.sum())} "
            f"mask_vox={int(mask.sum())} max_prob={float(seg_prob.max()):.4f}"
        )
        done += 1

    print(f"[SUMMARY] done={done}/{len(cases)}")
    print(f"[SUMMARY] masks: {out_root}")
    print(f"[SUMMARY] probs: {prob_root}")
    print(f"[SUMMARY] ambiguity: {ambiguity_root}")


if __name__ == "__main__":
    main()

