# GPAM Generation and Utility Scripts

This directory contains the low-level preprocessing, GPA generation, validation, resampling, environment-checking, and inference utilities used by GPAM-Net.

For the standard workflow, prefer the parameterized wrappers in `../powershell/`:

- `run_spm12_pipeline.ps1`: SPM12 tissue segmentation, brain-mask creation, and GPA generation.
- `run_sp_prediction.ps1`: GPAM-Net inference with a trained best checkpoint.
- `run_sp_training.ps1`: GPAM-Net training.

The scripts in this directory are useful when an individual step must be run, inspected, or debugged directly.

> **Compatibility note:** historical script names, command-line options, NPZ keys, and checkpoint fields use `sp`. In this release, `sp` means the geometric positional prior (GPA). Keep these compatibility names unless the data files and checkpoints are migrated together.

## Script Overview

| Script | Purpose | Typical use |
| --- | --- | --- |
| `check_sp_inference_env.py` | Checks CUDA, PyTorch, and `mamba_ssm` | Run before training or inference |
| `make_spm_standalone_segmentation_batches.py` | Creates standalone SPM batch files for INV2 images | Optional SPM standalone workflow |
| `make_whole_brain_mask_from_spm_c123.py` | Converts SPM c1/c2/c3 maps into refined brain masks | Recommended brain-mask utility |
| `spm12_c123_to_brain_mask.py` | Older compatible c1/c2/c3-to-mask implementation | Compatibility only |
| `resample_brain_masks_to_reference.py` | Resamples binary masks to reference grids | Repair mask/reference geometry mismatches |
| `generate_polar_sp_from_brain_mask.py` | Generates three-channel subject-specific GPA | Main GPA generation utility |
| `validate_mask_reference_geometry.py` | Checks mask/reference shape and affine | Run before GPA generation and inference |
| `validate_polar_sp.py` | Checks GPA shape, channels, values, and case coverage | Run after GPA generation |
| `visualize_sp.py` | Creates an RGB GPA quality-control image | Optional visual QC |
| `resample_to_fixed_spacing.py` | Resamples images and optional labels to fixed spacing | Optional experiment-specific preprocessing |
| `predict_sp_fold4_best_brainmasked.py` | Runs direct GPAM-Net sliding-window inference | Low-level alternative to the prediction wrapper |

## Setup

The examples below use PowerShell and generic placeholder paths:

```powershell
$repo = '<package-root>\GPAM-NET'
$tools = "$repo\GPAM_generation"
$python = '<python-env>\python.exe'
```

The Python environment requires the packages needed by the selected script. Across the complete workflow these include:

```text
numpy
scipy
SimpleITK
nibabel
matplotlib
torch
mamba_ssm
```

The PyTorch, CUDA, and `mamba_ssm` builds must be compatible with one another and with the trained checkpoint.

## Input Conventions

Use one consistent case ID, such as `sub001`, across every modality and generated file.

### Inference images

```text
<data-root>/cases/
|-- sub001/
|   |-- T1.nii.gz
|   `-- FLAIR.nii.gz
`-- sub002/
    |-- T1.nii.gz
    `-- FLAIR.nii.gz
```

The predictor accepts `.nii` or `.nii.gz`. The first modality must match the contrast used to train the checkpoint. If the checkpoint was trained with INV2, the image placed at the predictor's required `T1.nii[.gz]` path must contain the aligned INV2 contrast.

### SPM outputs

```text
<spm-root>/
|-- sub001/
|   |-- c1*.nii
|   |-- c2*.nii
|   `-- c3*.nii
`-- sub002/
    |-- c1*.nii
    |-- c2*.nii
    `-- c3*.nii
```

### Brain masks, references, and GPA

```text
<brain-mask-root>/
|-- sub001_brain.nii.gz
`-- sub002_brain.nii.gz

<reference-root>/
|-- sub001_T1.nii.gz
`-- sub002_T1.nii.gz

<gpa-root>/
|-- sub001_sp.npz
|-- sub001_sp.json
|-- sub002_sp.npz
`-- sub002_sp.json
```

When GPA NIfTI output is enabled, the additional quality-control channels are stored as:

```text
<gpa-root>/sub001/sub001_sp_r.nii.gz
<gpa-root>/sub001/sub001_sp_theta.nii.gz
<gpa-root>/sub001/sub001_sp_phi.nii.gz
```

A case-ID file is UTF-8 text with one case per line:

```text
sub001
sub002
```

## Recommended Workflow

### 1. Check the inference environment

```powershell
& $python -u "$tools\check_sp_inference_env.py"
```

The check must report `CUDA available: True` and successfully import `mamba_ssm`. It exits with an error if either requirement is unavailable.

### 2. Create brain masks from SPM c1/c2/c3 maps

```powershell
& $python -u "$tools\make_whole_brain_mask_from_spm_c123.py" `
  --spm-root '<data-root>\spm12_c123' `
  --out-root '<data-root>\brain_masks' `
  --output-suffix brain `
  --threshold 0.3 `
  --fill-holes `
  --closing-iter 1 `
  --keep-largest `
  --overwrite
```

The mask is created as:

```text
brain = (c1 + c2 + c3) > 0.3
brain = fill holes -> 3x3x3 closing -> largest connected component
```

The `0.3` value is the tissue-probability threshold used to construct a brain mask. It is not the lesion-prediction threshold.

### 3. Resample masks only when their grids do not match

Create `<data-root>\case_ids.txt`, then run:

```powershell
& $python -u "$tools\resample_brain_masks_to_reference.py" `
  --mask-root '<data-root>\brain_masks_original' `
  --reference-root '<data-root>\references' `
  --out-root '<data-root>\brain_masks_aligned' `
  --case-ids-file '<data-root>\case_ids.txt' `
  --mask-suffix brain `
  --reference-suffix T1 `
  --overwrite
```

This utility uses nearest-neighbor interpolation and writes binary masks. Do not use continuous interpolation for masks or labels.

### 4. Validate mask geometry

```powershell
& $python -u "$tools\validate_mask_reference_geometry.py" `
  --mask-root '<data-root>\brain_masks_aligned' `
  --reference-root '<data-root>\references' `
  --case-ids sub001 sub002 `
  --mask-suffix brain `
  --reference-suffix T1
```

The command fails if a mask is missing, contains fewer than 100 foreground voxels, or differs from its reference in shape or affine.

### 5. Generate GPA

```powershell
& $python -u "$tools\generate_polar_sp_from_brain_mask.py" `
  --wm-root '<data-root>\brain_masks_aligned' `
  --out-root '<data-root>\gpa' `
  --wm-suffix brain `
  --reference-root '<data-root>\references' `
  --reference-suffix T1 `
  --brain-mask-root '<data-root>\brain_masks_aligned' `
  --brain-mask-suffix brain `
  --radial-map centroid `
  --angle-encoding raw `
  --coordinate-space physical `
  --write-nifti `
  --overwrite
```

Although the historical argument is named `--wm-root`, the GPAM-Net workflow supplies the refined whole-brain mask. The recommended configuration writes three channels:

```text
r      normalized physical radial distance from the mask centroid
theta  normalized azimuth in [-1, 1]
phi    normalized elevation in [-1, 1]
```

The NPZ output contains the array `sp` with shape `[3, Z, Y, X]`. By default, values outside the aligned brain/reference region are zeroed.

Case selection can also be controlled with one of the following:

```text
--case-ids sub001 sub002
--case-ids-file <case-ids.txt>
--splits-json <splits_final.json> --fold 0 --split-key all
```

### 6. Validate generated GPA

```powershell
& $python -u "$tools\validate_polar_sp.py" `
  --sp-root '<data-root>\gpa' `
  --reference-root '<data-root>\references' `
  --case-ids-file '<data-root>\case_ids.txt' `
  --reference-suffix T1 `
  --expected-channels 3
```

The validator checks file presence, channel count, array shape, floating-point type, finite values, and nonzero occupancy. A successful run ends with `SP_VALIDATION_OK`.

### 7. Run direct inference

For routine use, prefer `../powershell/run_sp_prediction.ps1`. The direct Python entry point is:

```powershell
& $python -u "$tools\predict_sp_fold4_best_brainmasked.py" `
  --data-root '<data-root>\cases' `
  --ckpt '<checkpoint-root>\stage1_best.pth' `
  --sp-root '<data-root>\gpa' `
  --brain-mask-root '<data-root>\brain_masks_aligned' `
  --out-root '<output-root>\masks' `
  --prob-root '<output-root>\probabilities' `
  --ambiguity-root '<output-root>\ambiguity' `
  --threshold 0.50 `
  --min-size 10 `
  --patch-size 128 `
  --step 96 `
  --batch-size 1 `
  --sp-suffix sp
```

Outputs:

```text
<output-root>/masks/<case>_mask_thr0.50_min10.nii.gz
<output-root>/probabilities/<case>_prob.nii.gz
<output-root>/ambiguity/<case>_ambiguity.nii.gz
```


## Optional Utilities

### Create SPM standalone batch files

This tool searches for `*INV2*.nii` and `*INV2*.nii.gz` in a flat input directory:

```powershell
& $python -u "$tools\make_spm_standalone_segmentation_batches.py" `
  --input-root '<data-root>\INV2_flat' `
  --out-root '<data-root>\spm_batches' `
  --tpm '<SPM12-root>\tpm\TPM.nii' `
  --overwrite
```

It creates one prepared `.nii` image and one SPM `.m` job per case, plus `batch_files.txt`. Run those jobs with a compatible SPM standalone installation.

### Resample images and labels to fixed spacing

Input images must use one subject directory per case:

```text
<image-root>/sub001/sub001_INV2.nii.gz
<label-root>/sub001/sub001_seg.nii.gz
```

Run:

```powershell
& $python -u "$tools\resample_to_fixed_spacing.py" `
  --image-root '<data-root>\images' `
  --label-root '<data-root>\labels' `
  --out-image-root '<data-root>\resampled\images' `
  --out-label-root '<data-root>\resampled\labels' `
  --image-suffix INV2 `
  --label-suffix seg `
  --spacing 1 1 1 `
  --workers 4 `
  --overwrite
```

Images use B-spline interpolation. Labels use nearest-neighbor interpolation and are written on the resampled image grid. Use this tool only when the experiment explicitly requires fixed spacing; it is not a replacement for multimodal registration.


### Compatibility brain-mask converter

`spm12_c123_to_brain_mask.py` performs the same core c1/c2/c3 thresholding and refinement as `make_whole_brain_mask_from_spm_c123.py`, but it is retained as an older compatibility entry point. New workflows should use `make_whole_brain_mask_from_spm_c123.py`, which performs stricter input and completion checks.

## Geometry Requirements

For each subject, the first modality, FLAIR, lesion label, brain mask, and GPA must use the same voxel grid. They must agree in:

- Shape and voxel order.
- Voxel spacing.
- Physical origin.
- Direction and affine.

Similar voxel spacing alone is not sufficient. Register and resample continuous MRI images to the selected reference grid before GPA generation. Use nearest-neighbor interpolation for binary masks and labels. Do not crop, reorient, or resample only one input after GPA has been generated.

## Training

Training is implemented in `../src/gpam_net/` and launched through:

```powershell
& "$repo\powershell\run_sp_training.ps1" <required arguments>
```

See the root `../README.md` for the full training layout and command. The `sp_*` checkpoint fields and command-line options are retained to load existing GPAM-Net best weights correctly.
