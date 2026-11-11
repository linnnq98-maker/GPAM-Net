# GPAM-Net
## 1. How to Use Our Trained Model

### 1.1 Requirements

Prepare the following:

- A compatible GPAM-Net checkpoint, normally named `stage1_best.pth`.
- A Python 3.10 environment with CUDA-enabled PyTorch, `mamba_ssm`, NumPy, SciPy, SimpleITK, and nibabel.
- One aligned T1/INV2 volume and one FLAIR volume for each subject.
- One binary brain mask and one three-channel GPA file for each subject.
- MATLAB and SPM12 only if the brain masks and GPA files have not already been generated.

The PyTorch, CUDA, and `mamba_ssm` builds must be mutually compatible. The Linux wheel included in this repository is only suitable for the Python, PyTorch, CUDA, and ABI versions encoded in its file name.

### 1.2 Input layout

The inference wrapper expects one directory per subject:

```text
<data-root>/cases/
|-- sub001/
|   |-- T1.nii.gz
|   `-- FLAIR.nii.gz
|-- sub002/
|   |-- T1.nii.gz
|   `-- FLAIR.nii.gz
`-- ...
```

Brain masks and GPA files use the same subject IDs:

```text
<data-root>/brain_masks/
|-- sub001_brain.nii.gz
|-- sub002_brain.nii.gz
`-- ...

<data-root>/gpa/
|-- sub001_sp.npz
|-- sub002_sp.npz
`-- ...
```

Each GPA NPZ file must contain an array named `sp` with shape `[3, Z, Y, X]`.

The predictor requires the first modality to be named `T1.nii.gz` or `T1.nii`. If the checkpoint was trained with INV2 as its first modality, place the aligned INV2 volume at this required predictor path. Do not replace INV2 with a different T1 contrast merely by renaming the file; the image contrast must match the checkpoint's training data.

### 1.3 Generate brain masks and GPA when needed

If valid brain masks and GPA files already exist, skip this step. Otherwise, place the reference images in a flat directory using names such as `sub001_T1.nii.gz`, then run:

```powershell
$repo = '<package-root>\GPAM-NET'
$python = '<python-env>\python.exe'

& "$repo\powershell\run_spm12_pipeline.ps1" `
  -InputRoot '<data-root>\t1_for_gpa' `
  -ReferenceRoot '<data-root>\t1_for_gpa' `
  -InputSuffix T1 `
  -ReferenceSuffix T1 `
  -SpmRoot '<SPM12-root>' `
  -Matlab '<MATLAB-root>\bin\matlab.exe' `
  -Python $python `
  -RunRoot '<data-root>\gpa_preparation' `
  -WriteNifti
```

The required outputs are:

```text
<data-root>/gpa_preparation/brain_masks/
<data-root>/gpa_preparation/brain_polar_sp_centroid/
```

SPM12 segmentation and GPA generation preserve the reference image grid. They do not register FLAIR to T1/INV2.

### 1.4 Run inference

Use the parameterized PowerShell wrapper:

```powershell
$repo = '<package-root>\GPAM-NET'
$python = '<python-env>\python.exe'

& "$repo\powershell\run_sp_prediction.ps1" `
  -DataRoot '<data-root>\cases' `
  -SpRoot '<data-root>\gpa' `
  -BrainMaskRoot '<data-root>\brain_masks' `
  -Checkpoint '<checkpoint-root>\stage1_best.pth' `
  -Python $python `
  -RunRoot '<output-root>\gpam_net_predictions' `
  -GpuId 0 `
  -PatchSize 128 `
  -Step 96 `
  -BatchSize 1 `
  -Threshold 0.50 `
  -MinSize 10
```


The output directory contains:

```text
gpam_net_predictions/
|-- masks/          # final binary lesion masks
|-- probabilities/  # lesion probability maps
|-- ambiguity/      # boundary-focused ambiguity maps
|-- environment.log
`-- prediction.log
```

## 2. How to Train from Scratch

### 2.1 Training data layout

The training wrapper accepts either flat files or one directory per subject. The following flat layout is the simplest:

```text
<training-root>/
|-- INV2/
|   |-- sub001_INV2.nii.gz
|   `-- sub002_INV2.nii.gz
|-- FLAIR/
|   |-- sub001_FLAIR.nii.gz
|   `-- sub002_FLAIR.nii.gz
|-- labels/
|   |-- sub001_seg.nii.gz
|   `-- sub002_seg.nii.gz
|-- gpa/
|   |-- sub001_sp.npz
|   `-- sub002_sp.npz
`-- splits_final.json
```

The case IDs in all file names and in `splits_final.json` must match exactly. The split file is a JSON list with one object per fold:

```json
[
  {
    "train": ["sub001", "sub002"],
    "val": ["sub003"]
  }
]
```

For five-fold training, provide five such objects and run folds `0` through `4` separately. Generate each subject's GPA from the same aligned first-modality grid used for training.

### 2.2 Start training

The following command starts a new fold from random initialization and builds the disk cache before training:

```powershell
$repo = '<package-root>\GPAM-NET'
$python = '<python-env>\python.exe'
$fold = 0  # Set the desired zero-based fold index.

& "$repo\powershell\run_sp_training.ps1" `
  -Inv2Root '<training-root>\INV2' `
  -FlairRoot '<training-root>\FLAIR' `
  -LabelsRoot '<training-root>\labels' `
  -SpRoot '<training-root>\gpa' `
  -SplitsJson '<training-root>\splits_final.json' `
  -CacheRoot '<training-root>\cache' `
  -OutputRoot '<training-root>\outputs' `
  -Python $python `
  -Fold $fold `
  -GpuId 0 `
  -Epochs 300 `
  -BatchSize 1 `
  -NumWorkers 4 `
  -BuildCache
```

The wrapper does not load pretrained weights, so this is training from scratch. It writes the following files under `outputs/fold_N/`, where `N` is the selected fold index:

```text
stage1_best.pth        # best full-validation checkpoint
checkpoint_latest.pth # latest complete training state
training.log
environment.log
```

The wrapper intentionally refuses to start if a best or latest checkpoint already exists in the selected fold directory. Use a new output directory for a new experiment. Resuming an interrupted run requires invoking `gpam_net.train_stage1` with `--resume <checkpoint_latest.pth>` and the same data, split, model, and optimization settings.

## 3. Important Notes

### Image geometry and resolution

T1/INV2 and FLAIR do not merely need similar resolutions. For every subject, T1/INV2, FLAIR, the brain mask, the lesion label, and the GPA channels must represent the same voxel grid and agree in:

- Array shape and voxel order.
- Voxel spacing.
- Physical origin.
- Image direction and affine transformation.

If FLAIR is not already aligned, register and resample it to the T1/INV2 reference grid using an interpolation method suitable for continuous MRI intensities. Resample binary masks and lesion labels with nearest-neighbor interpolation only. After resampling, verify both array shape and physical geometry.

### Modality consistency

Use the same first-modality contrast, orientation, preprocessing, intensity convention, and GPA construction method that were used to train the checkpoint. A geometrically aligned conventional T1 image is not automatically equivalent to an INV2 image. Large domain differences can substantially reduce performance even when file dimensions match.

### GPA consistency

Generate GPA from the subject's own aligned reference grid. Do not reuse another subject's GPA, generate GPA before final registration, or crop only one input. GPA is supplied as model input and is not a substitute for registration. Historical `sp` names in scripts, options, NPZ keys, and checkpoint metadata must remain unchanged for compatibility.

### Checkpoints and software environment

Use the same model source and compatible PyTorch, CUDA, and `mamba_ssm` versions as the checkpoint. Do not silently ignore missing or unexpected checkpoint keys, and do not replace Mamba layers with a fallback implementation. The wrappers run an environment check before prediction or training and write the result to `environment.log`.
