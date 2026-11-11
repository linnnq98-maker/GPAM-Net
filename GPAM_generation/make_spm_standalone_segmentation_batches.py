"""Create one SPM standalone batch .m file per INV2 case.

SPM standalone `spm25.exe batch file.m` expects a job file that defines
`matlabbatch`; it does not execute arbitrary MATLAB scripts with loops. This
helper generates standalone-compatible job files.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def _case_id(path: Path) -> str:
    for part in path.name.split("_"):
        if part.startswith("sub-"):
            return part
    stem = path.name.replace(".nii.gz", "").replace(".nii", "")
    return stem.split("_")[0]


def _matlab_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _prepare_nii(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and not overwrite:
        return
    if src.name.lower().endswith(".nii.gz"):
        with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy2(src, dst)


def _write_batch(batch_path: Path, image_path: Path, tpm_path: Path) -> None:
    img = _matlab_path(image_path)
    tpm = _matlab_path(tpm_path)
    text = f"""matlabbatch = [];

matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{'{img},1'}};
matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = 18;
matlabbatch{{1}}.spm.spatial.preproc.channel.write = [1 1];

matlabbatch{{1}}.spm.spatial.preproc.tissue(1).tpm = {{'{tpm},1'}};
matlabbatch{{1}}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{{1}}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(1).warped = [0 0];

matlabbatch{{1}}.spm.spatial.preproc.tissue(2).tpm = {{'{tpm},2'}};
matlabbatch{{1}}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{{1}}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(2).warped = [0 0];

matlabbatch{{1}}.spm.spatial.preproc.tissue(3).tpm = {{'{tpm},3'}};
matlabbatch{{1}}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{{1}}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(3).warped = [0 0];

matlabbatch{{1}}.spm.spatial.preproc.tissue(4).tpm = {{'{tpm},4'}};
matlabbatch{{1}}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{{1}}.spm.spatial.preproc.tissue(4).native = [0 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(4).warped = [0 0];

matlabbatch{{1}}.spm.spatial.preproc.tissue(5).tpm = {{'{tpm},5'}};
matlabbatch{{1}}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{{1}}.spm.spatial.preproc.tissue(5).native = [0 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(5).warped = [0 0];

matlabbatch{{1}}.spm.spatial.preproc.tissue(6).tpm = {{'{tpm},6'}};
matlabbatch{{1}}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{{1}}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{{1}}.spm.spatial.preproc.tissue(6).warped = [0 0];

matlabbatch{{1}}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{{1}}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{{1}}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{{1}}.spm.spatial.preproc.warp.samp = 2;
matlabbatch{{1}}.spm.spatial.preproc.warp.write = [0 0];
"""
    batch_path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SPM standalone segmentation job files.")
    p.add_argument("--input-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--tpm", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    tpm = Path(args.tpm)
    if not tpm.is_file():
        raise SystemExit(f"TPM not found: {tpm}")

    files = sorted(input_root.glob("*INV2*.nii")) + sorted(input_root.glob("*INV2*.nii.gz"))
    if not files:
        raise SystemExit(f"No *INV2*.nii/.nii.gz files found in {input_root}")

    batch_list = out_root / "batch_files.txt"
    batch_paths = []
    for src in files:
        cid = _case_id(src)
        case_out = out_root / cid
        dst_img = case_out / f"{cid}_INV2.nii"
        batch = case_out / f"{cid}_segment_batch.m"
        _prepare_nii(src, dst_img, args.overwrite)
        _write_batch(batch, dst_img, tpm)
        batch_paths.append(batch)
        print(f"[SPMBatch] {cid}: {batch}")

    out_root.mkdir(parents=True, exist_ok=True)
    batch_list.write_text("\n".join(str(p) for p in batch_paths) + "\n", encoding="utf-8")
    print(f"[SPMBatch] wrote {batch_list}")


if __name__ == "__main__":
    main()
