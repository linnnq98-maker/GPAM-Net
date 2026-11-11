param(
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][string]$SpRoot,
    [Parameter(Mandatory=$true)][string]$BrainMaskRoot,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$RunRoot,
    [int]$GpuId = 0,
    [int]$Step = 96,
    [double]$Threshold = 0.50,
    [int]$MinSize = 10,
    [int]$PatchSize = 128,
    [int]$BatchSize = 1
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Package = Split-Path -Parent $PSScriptRoot
$Predictor = Join-Path $Package 'GPAM_generation\predict_sp_fold4_best_brainmasked.py'
$EnvCheck = Join-Path $Package 'GPAM_generation\check_sp_inference_env.py'
$srcRoot = Join-Path $Package 'src'

foreach ($path in @($DataRoot, $SpRoot, $BrainMaskRoot, $Checkpoint, $Python, $Predictor, $EnvCheck, $srcRoot)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
if (-not (Test-Path -LiteralPath (Join-Path $RunRoot 'masks'))) {
    New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
}
$MaskRoot = Join-Path $RunRoot 'masks'
$ProbRoot = Join-Path $RunRoot 'probabilities'
$AmbiguityRoot = Join-Path $RunRoot 'ambiguity'
$LogFile = Join-Path $RunRoot 'prediction.log'
New-Item -ItemType Directory -Force -Path $MaskRoot, $ProbRoot, $AmbiguityRoot | Out-Null

$cases = @(Get-ChildItem -LiteralPath $DataRoot -Directory | Sort-Object Name)
if ($cases.Count -eq 0) { throw "No case directories found under $DataRoot" }
foreach ($case in $cases) {
    foreach ($name in @('T1.nii.gz','FLAIR.nii.gz','T1.nii','FLAIR.nii')) {
        # Each modality may be .nii or .nii.gz; the predictor resolves either.
    }
    $hasT1 = (Test-Path -LiteralPath (Join-Path $case.FullName 'T1.nii.gz')) -or (Test-Path -LiteralPath (Join-Path $case.FullName 'T1.nii'))
    $hasFlair = (Test-Path -LiteralPath (Join-Path $case.FullName 'FLAIR.nii.gz')) -or (Test-Path -LiteralPath (Join-Path $case.FullName 'FLAIR.nii'))
    if (-not $hasT1 -or -not $hasFlair) { throw "$($case.Name): requires T1.nii[.gz] and FLAIR.nii[.gz]" }
}

$env:CUDA_VISIBLE_DEVICES = [string]$GpuId
$env:PYTHONPATH = "$srcRoot;$env:PYTHONPATH"
Write-Host '===== GPAM-Net fold-4 best prediction =====' -ForegroundColor Cyan
Write-Host "Cases:       $($cases.Count)"
Write-Host "Data:        $DataRoot"
Write-Host "SP:          $SpRoot"
Write-Host "Brain masks: $BrainMaskRoot"
Write-Host "Checkpoint:  $Checkpoint"
Write-Host "Threshold:   $Threshold"
Write-Host "Min size:    $MinSize"
Write-Host "Output:      $RunRoot"

& $Python -u $EnvCheck 2>&1 | Tee-Object -FilePath (Join-Path $RunRoot 'environment.log')
if ($LASTEXITCODE -ne 0) { throw "CUDA/mamba_ssm preflight failed: $LASTEXITCODE" }

$thresholdText = $Threshold.ToString('0.00', [Globalization.CultureInfo]::InvariantCulture)
$args = @(
    '-u', $Predictor,
    '--data-root', $DataRoot,
    '--ckpt', $Checkpoint,
    '--sp-root', $SpRoot,
    '--brain-mask-root', $BrainMaskRoot,
    '--out-root', $MaskRoot,
    '--prob-root', $ProbRoot,
    '--ambiguity-root', $AmbiguityRoot,
    '--threshold', $thresholdText,
    '--min-size', [string]$MinSize,
    '--patch-size', [string]$PatchSize,
    '--step', [string]$Step,
    '--batch-size', [string]$BatchSize,
    '--sp-suffix', 'sp'
)
& $Python @args 2>&1 | Tee-Object -FilePath $LogFile
if ($LASTEXITCODE -ne 0) { throw "SP prediction failed: $LASTEXITCODE. Log: $LogFile" }

$maskFiles = @(Get-ChildItem -LiteralPath $MaskRoot -File -Filter '*.nii.gz')
$probFiles = @(Get-ChildItem -LiteralPath $ProbRoot -File -Filter '*.nii.gz')
$ambiguityFiles = @(Get-ChildItem -LiteralPath $AmbiguityRoot -File -Filter '*.nii.gz')
$emptyFiles = @($maskFiles + $probFiles + $ambiguityFiles | Where-Object { $_.Length -eq 0 })
Write-Host "Masks:       $($maskFiles.Count)/$($cases.Count)"
Write-Host "Probability: $($probFiles.Count)/$($cases.Count)"
Write-Host "Ambiguity:   $($ambiguityFiles.Count)/$($cases.Count)"
Write-Host "Empty files: $($emptyFiles.Count)"
if ($maskFiles.Count -ne $cases.Count -or $probFiles.Count -ne $cases.Count -or $ambiguityFiles.Count -ne $cases.Count -or $emptyFiles.Count -ne 0) {
    throw 'Prediction output validation failed'
}
Write-Host "COMPLETE: $RunRoot" -ForegroundColor Green

