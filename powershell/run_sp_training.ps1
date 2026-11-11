param(
    [Parameter(Mandatory=$true)][string]$Inv2Root,
    [Parameter(Mandatory=$true)][string]$FlairRoot,
    [Parameter(Mandatory=$true)][string]$LabelsRoot,
    [Parameter(Mandatory=$true)][string]$SpRoot,
    [Parameter(Mandatory=$true)][string]$SplitsJson,
    [Parameter(Mandatory=$true)][string]$CacheRoot,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [Parameter(Mandatory=$true)][string]$Python,
    [ValidateRange(0, 4)][int]$Fold = 4,
    [int]$GpuId = 0,
    [int]$Epochs = 300,
    [int]$BatchSize = 4,
    [int]$NumWorkers = 4,
    [switch]$BuildCache,
    [switch]$CacheOverwrite,
    [switch]$DataParallel
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Package = Split-Path -Parent $PSScriptRoot
$srcRoot = Join-Path $Package 'src'
$EnvCheck = Join-Path $Package 'GPAM_generation\check_sp_inference_env.py'

foreach ($path in @($Inv2Root, $FlairRoot, $LabelsRoot, $SpRoot, $SplitsJson, $Python, $srcRoot, $EnvCheck)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}

$FoldRoot = Join-Path $OutputRoot "fold_$Fold"
$Best = Join-Path $FoldRoot 'stage1_best.pth'
$Latest = Join-Path $FoldRoot 'checkpoint_latest.pth'
$Log = Join-Path $FoldRoot 'training.log'
New-Item -ItemType Directory -Force -Path $FoldRoot, $CacheRoot | Out-Null

$env:CUDA_VISIBLE_DEVICES = [string]$GpuId
$env:PYTHONPATH = "$srcRoot;$env:PYTHONPATH"
$env:PYTHONUNBUFFERED = '1'

& $Python -u $EnvCheck 2>&1 | Tee-Object -FilePath (Join-Path $FoldRoot 'environment.log')
if ($LASTEXITCODE -ne 0) { throw "CUDA/mamba_ssm preflight failed: $LASTEXITCODE" }

if ((Test-Path -LiteralPath $Best) -or (Test-Path -LiteralPath $Latest)) {
    throw "Training output already exists under $FoldRoot. Use a new OutputRoot, or explicitly resume outside this wrapper."
}

# Parameters below are recovered from the verified fold_4 best checkpoint.
$args = @(
    '-u', '-m', 'gpam_net.train_stage1',
    '--inv2-root', $Inv2Root,
    '--flair-root', $FlairRoot,
    '--labels-root', $LabelsRoot,
    '--sp-root', $SpRoot,
    '--sp-suffix', 'sp',
    '--sp-channels', '3',
    '--sp-fusion-mode', 'early',
    '--sp-adapter-channels', '16',
    '--sp-gate-max', '0.15',
    '--sp-warmup-epochs', '50',
    '--sp-rampup-epochs', '120',
    '--sp-dropout', '0.1',
    '--cache-root', $CacheRoot,
    '--cache-keep-in-memory',
    '--splits-json', $SplitsJson,
    '--fold', [string]$Fold,
    '--save', $Best,
    '--save-latest', $Latest,
    '--epochs', [string]$Epochs,
    '--batch-size', [string]$BatchSize,
    '--patch-size', '128',
    '--samples-per-case', '8',
    '--fg-ratio', '0.75',
    '--center-jitter', '12',
    '--patch-scales', '0.75,1.0,1.25',
    '--flair-dropout', '0.0',
    '--boundary-fg-prob', '0.45',
    '--lesion-weight', '0.8',
    '--tversky-alpha', '0.15',
    '--tversky-beta', '0.85',
    '--tversky-weight', '0.4',
    '--det-weight', '0.0',
    '--uncertainty-head-weight', '0.03',
    '--uncertainty-boundary-weight', '0.5',
    '--mamba-block', 'simplified',
    '--boundary-blur-kernel', '3',
    '--full-val-every', '1',
    '--full-val-thresholds', '0.50',
    '--full-val-min-sizes', '10',
    '--full-val-mode', 'raw',
    '--full-val-raw-threshold', '0.50',
    '--full-val-raw-min-size', '10',
    '--full-val-step', '96',
    '--full-val-batch-size', '16',
    '--num-workers', [string]$NumWorkers,
    '--val-num-workers', '0'
)
if ($BuildCache) { $args += '--build-cache' }
if ($CacheOverwrite) { $args += '--cache-overwrite' }
if ($DataParallel) { $args += '--data-parallel' }

Write-Host '===== GPAM-Net training =====' -ForegroundColor Cyan
Write-Host "Fold:       $Fold"
Write-Host "SP:         $SpRoot"
Write-Host "Cache:      $CacheRoot"
Write-Host "Checkpoint: $Best"

& $Python @args 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { throw "GPAM-Net training failed: $LASTEXITCODE. Log: $Log" }
if (-not (Test-Path -LiteralPath $Best)) { throw "Training finished without best checkpoint: $Best" }
Write-Host "COMPLETE: $Best" -ForegroundColor Green


