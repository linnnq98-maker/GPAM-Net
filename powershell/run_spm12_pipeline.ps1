param(
    [Parameter(Mandatory=$true)][string]$InputRoot,
    [Parameter(Mandatory=$true)][string]$SpmRoot,
    [Parameter(Mandatory=$true)][string]$Matlab,
    [Parameter(Mandatory=$true)][string]$RunRoot,
    [Parameter(Mandatory=$true)][string]$Python,
    [string]$InputSuffix = 'T1',
    [string]$ReferenceRoot = '',
    [string]$ReferenceSuffix = 'T1',
    [int]$ExpectedCases = 0,
    [double]$BrainThreshold = 0.3,
    [int]$ClosingIter = 1,
    [switch]$NoFillHoles,
    [switch]$NoKeepLargest,
    [switch]$WriteNifti,
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Package = Split-Path -Parent $PSScriptRoot
$SpmOut = Join-Path $RunRoot 'spm12_c123'
$BrainOut = Join-Path $RunRoot 'brain_masks'
$SpOut = Join-Path $RunRoot 'brain_polar_sp_centroid'
$LogOut = Join-Path $RunRoot 'logs'
if ([string]::IsNullOrWhiteSpace($ReferenceRoot)) { $ReferenceRoot = $InputRoot }

foreach ($path in @($InputRoot, $ReferenceRoot, $SpmRoot, $Matlab, $Python, (Join-Path $SpmRoot 'spm.m'), (Join-Path $SpmRoot 'tpm\TPM.nii'))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
New-Item -ItemType Directory -Force -Path $RunRoot, $LogOut | Out-Null

$stable = Join-Path $PSScriptRoot 'run_spm12_segment_original_stable.ps1'
if (-not (Test-Path -LiteralPath $stable)) { throw "Missing packaged runner: $stable" }

Write-Host '===== 1. SPM12 native-grid c1/c2/c3 =====' -ForegroundColor Cyan
$spmArgs = @('-Matlab', $Matlab, '-SpmRoot', $SpmRoot, '-InRoot', $InputRoot, '-OutRoot', $SpmOut, '-InputSuffix', $InputSuffix)
if ($ExpectedCases -gt 0) { $spmArgs += @('-ExpectedCases', [string]$ExpectedCases) }
& $stable @spmArgs
if ($LASTEXITCODE -ne 0) { throw "SPM12 segmentation failed: $LASTEXITCODE" }

Write-Host '===== 2. c1+c2+c3 -> refined whole-brain mask =====' -ForegroundColor Cyan
$maskArgs = @((Join-Path $Package 'GPAM_generation\make_whole_brain_mask_from_spm_c123.py'), '--spm-root', $SpmOut, '--out-root', $BrainOut, '--threshold', [string]$BrainThreshold, '--closing-iter', [string]$ClosingIter)
if (-not $NoFillHoles) { $maskArgs += '--fill-holes' } else { $maskArgs += '--no-fill-holes' }
if (-not $NoKeepLargest) { $maskArgs += '--keep-largest' } else { $maskArgs += '--no-keep-largest' }
if ($Overwrite) { $maskArgs += '--overwrite' }
& $Python @maskArgs
if ($LASTEXITCODE -ne 0) { throw "Brain mask generation failed: $LASTEXITCODE" }

Write-Host '===== 3. brain-centered physical polar SP [r,theta,phi] =====' -ForegroundColor Cyan
$spArgs2 = @(
    '-u', (Join-Path $Package 'GPAM_generation\generate_polar_sp_from_brain_mask.py'),
    '--wm-root', $BrainOut,
    '--out-root', $SpOut,
    '--wm-suffix', 'brain',
    '--reference-root', $ReferenceRoot,
    '--reference-suffix', $ReferenceSuffix,
    '--brain-mask-root', $BrainOut,
    '--brain-mask-suffix', 'brain',
    '--wm-threshold', '0.5',
    '--nonzero-threshold', '0.0',
    '--radial-map', 'centroid',
    '--angle-encoding', 'raw',
    '--coordinate-space', 'physical'
)
if ($WriteNifti) { $spArgs2 += '--write-nifti' }
if ($Overwrite) { $spArgs2 += '--overwrite' }
& $Python @spArgs2
if ($LASTEXITCODE -ne 0) { throw "Polar SP generation failed: $LASTEXITCODE" }

$expected = if ($ExpectedCases -gt 0) { $ExpectedCases } else {
    $suffixRegex = [Regex]::Escape($InputSuffix)
    @(Get-ChildItem -LiteralPath $InputRoot -File | Where-Object { $_.Name -match "^sub-?\d+.*(?:_|-)$suffixRegex\.nii(\.gz)?$" }).Count
}
$maskCount = @(Get-ChildItem -LiteralPath $BrainOut -File -Filter '*_brain.nii.gz').Count
$spCount = @(Get-ChildItem -LiteralPath $SpOut -File -Filter '*_sp.npz').Count
Write-Host "[CHECK] brain masks=$maskCount/$expected SP NPZ=$spCount/$expected"
if ($maskCount -ne $expected -or $spCount -ne $expected) {
    throw "Pipeline output incomplete: brain masks=$maskCount/$expected, SP NPZ=$spCount/$expected"
}

Write-Host "COMPLETE: $RunRoot" -ForegroundColor Green
Write-Host "SPM c1/c2/c3: $SpmOut"
Write-Host "Brain masks:   $BrainOut"
Write-Host "Polar SP:      $SpOut"
