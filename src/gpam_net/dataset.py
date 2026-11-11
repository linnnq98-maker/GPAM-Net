"""Dataset for T1+T2 joint detection + segmentation."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from scipy.ndimage import zoom as _zoom

from .utils import crop_common, find_image

try:
    import blosc2
except Exception:
    blosc2 = None


def _read_nifti(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    fg = x > 0
    if int(fg.sum()) < 100:
        mean, std = float(x.mean()), float(x.std()) + 1e-6
    else:
        mean, std = float(x[fg].mean()), float(x[fg].std()) + 1e-6
    return ((x - mean) / std).astype(np.float32, copy=False)


def _find_seg(root: Path, cid: str) -> Path | None:
    for p in [root / cid / f"{cid}_seg.nii.gz", root / f"{cid}_seg.nii.gz"]:
        if p.is_file():
            return p
    return None


def _label_components(seg: np.ndarray) -> list[np.ndarray]:
    lab, n = ndimage.label(seg > 0.5)
    comps = []
    for i in range(1, n + 1):
        coords = np.argwhere(lab == i)
        if len(coords) > 0:
            comps.append(coords)
    return comps


def _boundary_mask(seg: np.ndarray) -> np.ndarray:
    seg_bin = seg > 0.5
    if not np.any(seg_bin):
        return np.zeros_like(seg_bin, dtype=bool)
    eroded = ndimage.binary_erosion(seg_bin, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return seg_bin & ~eroded


def _ring_mask(seg: np.ndarray, radius_voxels: int = 5) -> np.ndarray:
    """Peri-lesion ring: lesion mask expanded by N voxels, excluding lesion."""
    seg_bin = seg > 0.5
    if not np.any(seg_bin) or int(radius_voxels) <= 0:
        return np.zeros_like(seg, dtype=np.float32)
    dist_to_lesion = ndimage.distance_transform_edt(~seg_bin)
    ring = (dist_to_lesion > 0.0) & (dist_to_lesion <= float(radius_voxels))
    return ring.astype(np.float32, copy=False)


def find_sp_case(root: Path, cid: str, suffix: str = "sp") -> Path | None:
    """Find a per-case spatial-position npz generated as <case>_<suffix>.npz."""
    root = Path(root)
    suffix = str(suffix).removesuffix(".npz").strip("_")
    names = [f"{cid}_{suffix}.npz", f"{cid}.npz"] if suffix else [f"{cid}.npz"]
    for name in names:
        for p in (root / cid / name, root / name):
            if p.is_file():
                return p
    patterns = [f"{cid}*{suffix}*.npz", f"{cid}*.npz"] if suffix else [f"{cid}*.npz"]
    for sr in (root / cid, root):
        if not sr.is_dir():
            continue
        for pat in patterns:
            matches = sorted(sr.glob(pat))
            if matches:
                return matches[0]
    return None


def load_sp_case(
    root: Path,
    cid: str,
    suffix: str = "sp",
    expected_shape: tuple[int, int, int] | None = None,
    channels: int | None = None,
) -> np.ndarray:
    """Load spatial-position channels shaped [C,Z,Y,X] for a case."""
    path = find_sp_case(root, cid, suffix)
    if path is None:
        raise FileNotFoundError(f"Could not find SP npz for {cid} in {root} with suffix={suffix!r}")
    with np.load(path, allow_pickle=False) as f:
        if "sp" in f:
            sp = f["sp"]
        else:
            keys = list(f.keys())
            if not keys:
                raise KeyError(f"{path} does not contain any arrays")
            sp = f[keys[0]]
    sp = np.asarray(sp, dtype=np.float32)
    if sp.ndim == 3:
        sp = sp[None]
    if sp.ndim != 4:
        raise ValueError(f"Expected SP shape [C,Z,Y,X], got {sp.shape} from {path}")
    if channels is not None and int(channels) > 0:
        channels = int(channels)
        if sp.shape[0] < channels:
            raise ValueError(f"{path} has {sp.shape[0]} SP channels, expected at least {channels}")
        sp = sp[:channels]
    if expected_shape is not None and tuple(sp.shape[1:]) != tuple(expected_shape):
        raise ValueError(
            f"{cid}: SP shape {tuple(sp.shape[1:])} from {path} does not match input shape "
            f"{tuple(expected_shape)}. Generate/resample SP in the same space as the current input."
        )
    return sp.astype(np.float32, copy=False)


def _sample_component_index(comps: list[np.ndarray]) -> int:
    sizes = np.array([max(1, len(c)) for c in comps], dtype=np.float32)
    weights = 1.0 / np.sqrt(sizes)
    weights = weights / weights.sum()
    return int(np.random.choice(len(comps), p=weights))


def _sample_voxel(coords: np.ndarray) -> Tuple[int, int, int]:
    if coords.size == 0:
        return 0, 0, 0
    idx = int(np.random.randint(len(coords)))
    z, y, x = coords[idx]
    return int(z), int(y), int(x)


def _augment(*arrays: np.ndarray, intensity_jitter: bool = True) -> Tuple[np.ndarray, ...]:
    out = list(arrays)
    for axis in range(3):
        if np.random.rand() > 0.5:
            out = [np.flip(a, axis=axis).copy() for a in out]
    if intensity_jitter and np.random.rand() > 0.5:
        shift = float(np.random.uniform(-0.05, 0.05))
        scale = float(np.random.uniform(0.95, 1.05))
        out[0] = out[0] * scale + shift
        if len(out) > 1:
            out[1] = out[1] * scale + shift
    if intensity_jitter and np.random.rand() > 0.5:
        noise = np.random.randn(*out[0].shape).astype(np.float32) * 0.02
        out[0] = out[0] + noise
        if len(out) > 1:
            out[1] = out[1] + noise
    return tuple(out)


def _pad_to_patch(arrays: List[np.ndarray], patch_size: Tuple[int, int, int]) -> List[np.ndarray]:
    pz, py, px = patch_size
    z_dim, y_dim, x_dim = arrays[0].shape
    pad_z = max(0, pz - z_dim)
    pad_y = max(0, py - y_dim)
    pad_x = max(0, px - x_dim)
    if pad_z or pad_y or pad_x:
        return [np.pad(a, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant") for a in arrays]
    return arrays


def _crop_at_center(arrays: List[np.ndarray], center: Tuple[int, int, int],
                    patch_size: Tuple[int, int, int]) -> List[np.ndarray]:
    pz, py, px = patch_size
    z_dim, y_dim, x_dim = arrays[0].shape
    cz, cy, cx = center
    z0 = int(np.clip(cz - pz // 2, 0, max(0, z_dim - pz)))
    y0 = int(np.clip(cy - py // 2, 0, max(0, y_dim - py)))
    x0 = int(np.clip(cx - px // 2, 0, max(0, x_dim - px)))
    return [a[z0:z0 + pz, y0:y0 + py, x0:x0 + px] for a in arrays]


def _force_shape(a: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    pz, py, px = patch_size
    z_dim, y_dim, x_dim = a.shape
    pad = ((0, max(0, pz - z_dim)), (0, max(0, py - y_dim)), (0, max(0, px - x_dim)))
    if any(p[1] > 0 for p in pad):
        a = np.pad(a, pad, mode="constant")
    z_dim, y_dim, x_dim = a.shape
    z0 = (z_dim - pz) // 2
    y0 = (y_dim - py) // 2
    x0 = (x_dim - px) // 2
    return a[z0:z0 + pz, y0:y0 + py, x0:x0 + px]


def _resize_to_shape(a: np.ndarray, out_shape: Tuple[int, int, int], order: int) -> np.ndarray:
    if tuple(a.shape) == tuple(out_shape):
        return a.astype(np.float32, copy=False)
    factors = [float(o) / float(i) for o, i in zip(out_shape, a.shape)]
    mode = "nearest" if order == 0 else "reflect"
    out = _zoom(a, zoom=factors, order=order, mode=mode, prefilter=(order > 0))
    return out.astype(np.float32, copy=False)


def _crop_to_common_shape(*arrays: np.ndarray | None) -> tuple[np.ndarray | None, ...]:
    present = [a for a in arrays if a is not None]
    z = min(a.shape[-3] for a in present)
    y = min(a.shape[-2] for a in present)
    x = min(a.shape[-1] for a in present)
    cropped: list[np.ndarray | None] = []
    for a in arrays:
        if a is None:
            cropped.append(None)
        elif a.ndim == 4:
            cropped.append(a[:, :z, :y, :x])
        else:
            cropped.append(a[:z, :y, :x])
    return tuple(cropped)


def _patchify_case(
    inv2: np.ndarray,
    flair: np.ndarray,
    seg: np.ndarray,
    ring: np.ndarray,
    sp: np.ndarray | None,
    center: tuple[int, int, int],
    crop_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    augment: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    arrays = [inv2, flair, seg, ring]
    if sp is not None:
        arrays.extend([sp[i] for i in range(sp.shape[0])])

    cropped = _crop_at_center(_pad_to_patch(arrays, crop_size), center, crop_size)
    inv2_c = _resize_to_shape(cropped[0], patch_size, order=1)
    flair_c = _resize_to_shape(cropped[1], patch_size, order=1)
    seg_c = _resize_to_shape(cropped[2], patch_size, order=0)
    ring_c = _resize_to_shape(cropped[3], patch_size, order=0)
    sp_c = None
    if sp is not None:
        sp_c = np.stack([_resize_to_shape(a, patch_size, order=1) for a in cropped[4:]], axis=0)

    if augment:
        aug_arrays = [inv2_c, flair_c, seg_c, ring_c]
        if sp_c is not None:
            aug_arrays.extend([sp_c[i] for i in range(sp_c.shape[0])])
        aug_arrays = list(_augment(*aug_arrays))
        inv2_c, flair_c, seg_c, ring_c = aug_arrays[:4]
        if sp_c is not None:
            sp_c = np.stack(aug_arrays[4:], axis=0)

    inv2_c = _force_shape(inv2_c, patch_size)
    flair_c = _force_shape(flair_c, patch_size)
    seg_c = _force_shape(seg_c, patch_size)
    ring_c = _force_shape(ring_c, patch_size)
    if sp_c is not None:
        sp_c = np.stack([_force_shape(a, patch_size) for a in sp_c], axis=0)
    return inv2_c, flair_c, seg_c, ring_c, sp_c


def find_nnunet_preprocessed_case(root: Path, cid: str) -> Path | None:
    root = Path(root)
    for suffix in (".b2nd", ".npy", ".npz"):
        p = root / f"{cid}{suffix}"
        if p.is_file():
            return p
    for suffix in (".b2nd", ".npy", ".npz"):
        hits = sorted(root.rglob(f"{cid}{suffix}"))
        if hits:
            return hits[0]
    return None


def _normalize_channel_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _find_dataset_json(preprocessed_root: Path) -> Path | None:
    root = Path(preprocessed_root)
    for p in (
        root / "dataset.json",
        root.parent / "dataset.json",
        root.parent.parent / "dataset.json",
    ):
        if p.is_file():
            return p
    return None


def infer_nnunet_channel_indices(
    preprocessed_root: Path,
    num_image_channels: int = 2,
    preferred_channel_names: Sequence[str] = ("INV2", "FLAIR"),
) -> tuple[list[int], dict | None]:
    dataset_json_path = _find_dataset_json(preprocessed_root)
    meta: dict | None = None
    if dataset_json_path is None:
        return list(range(num_image_channels)), None

    try:
        dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        meta = {"dataset_json_path": str(dataset_json_path), "warning": f"failed_to_read: {exc}"}
        return list(range(num_image_channels)), meta

    channel_names = dataset_json.get("channel_names", {})
    ordered = sorted(
        ((int(k), str(v)) for k, v in channel_names.items()),
        key=lambda item: item[0],
    )
    ordered_names = [name for _, name in ordered]
    meta = {
        "dataset_json_path": str(dataset_json_path),
        "ordered_channel_names": ordered_names,
        "channel_names": channel_names,
    }
    if len(ordered) < num_image_channels:
        meta["warning"] = f"dataset_json_has_only_{len(ordered)}_channels"
        return list(range(num_image_channels)), meta

    name_to_idx = {_normalize_channel_name(name): idx for idx, name in ordered}
    indices: list[int] = []
    for name in preferred_channel_names[:num_image_channels]:
        idx = name_to_idx.get(_normalize_channel_name(name))
        if idx is None:
            meta["warning"] = f"could_not_match_{name}"
            return list(range(num_image_channels)), meta
        indices.append(idx)

    meta["selected_channel_indices"] = indices
    meta["selected_channel_names"] = [ordered_names[i] for i in indices]
    return indices, meta


def load_nnunet_preprocessed_case(
    path: Path,
    num_image_channels: int = 2,
    channel_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    path = Path(path)
    properties = None

    if path.suffix == ".b2nd":
        if blosc2 is None:
            raise RuntimeError(f"{path} is a .b2nd file but blosc2 is not available")
        image = np.asarray(blosc2.open(str(path), mode="r"))
        seg_path = path.with_name(path.stem + "_seg.b2nd")
        if not seg_path.is_file():
            raise FileNotFoundError(f"Missing segmentation file for {path}: {seg_path}")
        seg = np.asarray(blosc2.open(str(seg_path), mode="r"))
        prop_path = path.with_suffix(".pkl")
        if prop_path.is_file():
            with prop_path.open("rb") as f:
                properties = pickle.load(f)
    elif path.suffix == ".npy":
        image = np.load(path, mmap_mode="r")
        seg_path = path.with_name(path.stem + "_seg.npy")
        if seg_path.is_file():
            seg = np.load(seg_path, mmap_mode="r")
        else:
            npz_path = path.with_suffix(".npz")
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing segmentation file for {path}")
            with np.load(npz_path) as f:
                if "seg" not in f:
                    raise KeyError(f"{npz_path} does not contain a 'seg' array")
                seg = f["seg"]
        prop_path = path.with_suffix(".pkl")
        if prop_path.is_file():
            with prop_path.open("rb") as f:
                properties = pickle.load(f)
    else:
        with np.load(path) as f:
            if "data" not in f:
                raise KeyError(f"{path} does not contain a 'data' array")
            image = f["data"]
            if "seg" not in f:
                raise KeyError(f"{path} does not contain a 'seg' array")
            seg = f["seg"]
        prop_path = path.with_suffix(".pkl")
        if prop_path.is_file():
            with prop_path.open("rb") as f:
                properties = pickle.load(f)

    image = np.asarray(image, dtype=np.float32)
    seg = np.asarray(seg, dtype=np.float32)

    if image.ndim != 4:
        raise ValueError(f"Expected nnUNet preprocessed image array [C,Z,Y,X], got {image.shape} from {path}")
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim != 3:
        raise ValueError(f"Expected nnUNet preprocessed seg array [Z,Y,X], got {seg.shape} from {path}")
    if image.shape[0] < num_image_channels:
        raise ValueError(
            f"{path} has {image.shape[0]} image channels, expected at least {num_image_channels}"
        )
    if channel_indices is None:
        channel_indices = list(range(num_image_channels))
    else:
        channel_indices = [int(i) for i in channel_indices][:num_image_channels]
    if len(channel_indices) < num_image_channels:
        raise ValueError(
            f"{path} channel_indices has length {len(channel_indices)}, expected at least {num_image_channels}"
        )
    if max(channel_indices) >= image.shape[0]:
        raise ValueError(
            f"{path} has {image.shape[0]} image channels, cannot select indices {channel_indices}"
        )

    return image[channel_indices].astype(np.float32, copy=False), (seg > 0.5).astype(np.float32, copy=False), properties


class JointDetSegDataset(Dataset):
    def __init__(
        self,
        inv2_root: Path,
        flair_root: Path,
        labels_root: Path,
        case_ids: List[str],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        samples_per_case: int = 8,
        fg_ratio: float = 0.6,
        center_jitter: int = 16,
        augment: bool = True,
        flair_dropout_prob: float = 0.5,
        heatmap_sigma: float = 3.0,
        label_min_voxels: int = 3,
        boundary_fg_prob: float = 0.35,
        patch_scales: Tuple[float, ...] = (1.0,),
        ring_dilate_voxels: int = 5,
        sp_root: Path | None = None,
        sp_suffix: str = "sp",
        sp_channels: int = 3,
    ):
        self.patch_size = patch_size
        self.samples_per_case = samples_per_case
        self.fg_ratio = fg_ratio
        self.center_jitter = center_jitter
        self.augment = augment
        self.flair_dropout_prob = flair_dropout_prob
        self.heatmap_sigma = heatmap_sigma
        self.label_min_voxels = label_min_voxels
        self.boundary_fg_prob = boundary_fg_prob
        self.ring_dilate_voxels = int(ring_dilate_voxels)
        self.sp_root = Path(sp_root) if sp_root else None
        self.sp_suffix = str(sp_suffix)
        self.sp_channels = int(sp_channels)
        self.patch_scales = tuple(float(s) for s in patch_scales if float(s) > 0)
        if not self.patch_scales:
            self.patch_scales = (1.0,)

        inv2_root = Path(inv2_root)
        flair_root = Path(flair_root)
        labels_root = Path(labels_root)

        self.cases: List[Tuple[Path, Path | None, Path, Path | None, str]] = []
        for cid in case_ids:
            inv2 = find_image(inv2_root, cid, "INV2")
            flair = find_image(flair_root, cid, "FLAIR")
            seg = _find_seg(labels_root, cid)
            if inv2 is None or seg is None:
                print(f"[JointDetSegDataset] WARN: {cid} missing INV2/label, skipped")
                continue
            if flair is None:
                print(f"[JointDetSegDataset] WARN: {cid} missing FLAIR, using zeros")
            sp_path = None
            if self.sp_root is not None:
                sp_path = find_sp_case(self.sp_root, cid, self.sp_suffix)
                if sp_path is None:
                    raise FileNotFoundError(
                        f"[JointDetSegDataset] {cid} missing SP in {self.sp_root} "
                        f"with suffix={self.sp_suffix!r}"
                    )
            self.cases.append((inv2, flair, seg, sp_path, cid))
        if not self.cases:
            raise RuntimeError("JointDetSegDataset has 0 valid cases")
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_case

    def _load(self, case_idx: int):
        if case_idx in self._cache:
            return self._cache[case_idx]
        inv2_path, flair_path, seg_path, sp_path, cid = self.cases[case_idx]
        inv2 = _zscore(_read_nifti(inv2_path))
        flair = _zscore(_read_nifti(flair_path)) if flair_path is not None else None
        seg = (_read_nifti(seg_path) > 0.5).astype(np.float32)
        sp = None
        if sp_path is not None:
            sp = load_sp_case(
                sp_path.parent,
                cid,
                self.sp_suffix,
                expected_shape=inv2.shape,
                channels=self.sp_channels,
            )
        if flair is None:
            inv2, seg, sp = _crop_to_common_shape(inv2, seg, sp)
            flair = np.zeros_like(inv2, dtype=np.float32)
        else:
            inv2, flair, seg, sp = _crop_to_common_shape(inv2, flair, seg, sp)
        comps = _label_components(seg)
        boundary = _boundary_mask(seg)
        ring = _ring_mask(seg, self.ring_dilate_voxels)
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        self._cache[case_idx] = (inv2, flair, seg, ring, sp, comps, boundary, cid)
        return inv2, flair, seg, ring, sp, comps, boundary, cid

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
            inv2, flair, seg, ring, sp, center, crop_size, self.patch_size, self.augment
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


class JointDetSegNNUNetPreprocessedDataset(Dataset):
    def __init__(
        self,
        preprocessed_root: Path,
        case_ids: List[str],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        samples_per_case: int = 8,
        fg_ratio: float = 0.6,
        center_jitter: int = 16,
        augment: bool = True,
        flair_dropout_prob: float = 0.5,
        heatmap_sigma: float = 3.0,
        label_min_voxels: int = 3,
        boundary_fg_prob: float = 0.35,
        patch_scales: Tuple[float, ...] = (1.0,),
        num_image_channels: int = 2,
        ring_dilate_voxels: int = 5,
        sp_root: Path | None = None,
        sp_suffix: str = "sp",
        sp_channels: int = 3,
    ):
        self.preprocessed_root = Path(preprocessed_root)
        self.patch_size = patch_size
        self.samples_per_case = samples_per_case
        self.fg_ratio = fg_ratio
        self.center_jitter = center_jitter
        self.augment = augment
        self.flair_dropout_prob = flair_dropout_prob
        self.heatmap_sigma = heatmap_sigma
        self.label_min_voxels = label_min_voxels
        self.boundary_fg_prob = boundary_fg_prob
        self.ring_dilate_voxels = int(ring_dilate_voxels)
        self.num_image_channels = int(num_image_channels)
        self.sp_root = Path(sp_root) if sp_root else None
        self.sp_suffix = str(sp_suffix)
        self.sp_channels = int(sp_channels)
        self.channel_indices, self.channel_meta = infer_nnunet_channel_indices(
            self.preprocessed_root,
            self.num_image_channels,
            preferred_channel_names=("INV2", "FLAIR"),
        )
        if self.channel_meta and self.channel_meta.get("selected_channel_names"):
            print(
                f"[JointDetSegNNUNetPreprocessedDataset] channel_order={self.channel_meta['selected_channel_names']} "
                f"indices={self.channel_indices} dataset_json={self.channel_meta['dataset_json_path']}",
                flush=True,
            )
        elif self.channel_meta and self.channel_meta.get("warning"):
            print(
                f"[JointDetSegNNUNetPreprocessedDataset] WARN: {self.channel_meta['warning']} "
                f"dataset_json={self.channel_meta.get('dataset_json_path')}, using leading channels {self.channel_indices}",
                flush=True,
            )
        else:
            print(
                f"[JointDetSegNNUNetPreprocessedDataset] WARN: dataset.json not found, using leading channels {self.channel_indices}",
                flush=True,
            )
        self.patch_scales = tuple(float(s) for s in patch_scales if float(s) > 0)
        if not self.patch_scales:
            self.patch_scales = (1.0,)

        self.cases: List[Tuple[Path, Path | None, str]] = []
        for cid in case_ids:
            case_path = find_nnunet_preprocessed_case(self.preprocessed_root, cid)
            if case_path is None:
                print(f"[JointDetSegNNUNetPreprocessedDataset] WARN: {cid} missing preprocessed case, skipped")
                continue
            sp_path = None
            if self.sp_root is not None:
                sp_path = find_sp_case(self.sp_root, cid, self.sp_suffix)
                if sp_path is None:
                    raise FileNotFoundError(
                        f"[JointDetSegNNUNetPreprocessedDataset] {cid} missing SP in {self.sp_root} "
                        f"with suffix={self.sp_suffix!r}"
                    )
            self.cases.append((case_path, sp_path, cid))
        if not self.cases:
            raise RuntimeError("JointDetSegNNUNetPreprocessedDataset has 0 valid cases")
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_case

    def _load(self, case_idx: int):
        if case_idx in self._cache:
            return self._cache[case_idx]
        case_path, sp_path, cid = self.cases[case_idx]
        image, seg, _ = load_nnunet_preprocessed_case(
            case_path,
            self.num_image_channels,
            channel_indices=self.channel_indices,
        )
        if image.shape[0] < 1:
            raise ValueError(f"{case_path} has no image channels")
        inv2 = image[0]
        flair = image[1] if image.shape[0] > 1 else np.zeros_like(inv2, dtype=np.float32)
        sp = None
        if sp_path is not None:
            sp = load_sp_case(
                sp_path.parent,
                cid,
                self.sp_suffix,
                expected_shape=inv2.shape,
                channels=self.sp_channels,
            )
        inv2, flair, seg, sp = _crop_to_common_shape(inv2, flair, seg, sp)
        comps = _label_components(seg)
        boundary = _boundary_mask(seg)
        ring = _ring_mask(seg, self.ring_dilate_voxels)
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        self._cache[case_idx] = (inv2, flair, seg, ring, sp, comps, boundary, cid)
        return inv2, flair, seg, ring, sp, comps, boundary, cid

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
            inv2, flair, seg, ring, sp, center, crop_size, self.patch_size, self.augment
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
