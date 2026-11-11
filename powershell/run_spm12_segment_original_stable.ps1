param(
    [Parameter(Mandatory=$true)][string]$Matlab,
    [Parameter(Mandatory=$true)][string]$SpmRoot,
    [Parameter(Mandatory=$true)][string]$InRoot,
    [Parameter(Mandatory=$true)][string]$OutRoot,
    [string]$InputSuffix = 'T1',
    [int]$ExpectedCases = 0,
    [string[]]$CaseIds = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Package = Split-Path -Parent $PSScriptRoot
$RunnerM = Join-Path $Package 'matlab\spm12_segment_one_original.m'
foreach ($path in @($Matlab, $SpmRoot, $InRoot, $RunnerM)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" } }
New-Item -ItemType Directory -Force -Path $OutRoot, (Join-Path $OutRoot '_logs') | Out-Null

$suffixRegex = [Regex]::Escape($InputSuffix)
$files = @(
    Get-ChildItem -LiteralPath $InRoot -File |
    Where-Object { $_.Name -match "^sub-?\d+.*(?:_|-)$suffixRegex\.nii(\.gz)?$" } |
    Sort-Object Name
)
if ($ExpectedCases -gt 0 -and $files.Count -ne $ExpectedCases) { throw "Expected $ExpectedCases input files, found $($files.Count)" }
if ($files.Count -eq 0) { throw "No *$InputSuffix.nii[.gz] inputs found in $InRoot" }

$done = 0
$failed = 0
foreach ($file in $files) {
    if ($file.BaseName -match '\.nii$') { $stem = $file.BaseName.Substring(0, $file.BaseName.Length - 4) } else { $stem = $file.BaseName }
    $cid = if ($stem -match '^(sub-?\d+)') { $Matches[1] } else { $stem.Split('_')[0] }
    if ($CaseIds.Count -gt 0 -and $CaseIds -notcontains $cid) { continue }
    $caseOut = Join-Path $OutRoot $cid
    New-Item -ItemType Directory -Force -Path $caseOut | Out-Null
    $runner = Join-Path $caseOut 'run_spm12_segment.m'
    $log = Join-Path $OutRoot "_logs\$cid.log"
    $err = Join-Path $OutRoot "_logs\$cid.err"
    $inputF = $file.FullName.Replace('\','/')
    $outF = $caseOut.Replace('\','/')
    $spmF = $SpmRoot.Replace('\','/')
    $runnerDir = (Split-Path -Parent $RunnerM).Replace('\','/')
    @"
try
  addpath('$runnerDir');
  spm12_segment_one_original('$inputF','$outF','$spmF');
  exit(0);
catch ME
  fprintf(2,'%s\n',getReport(ME,'extended','hyperlinks','off'));
  exit(1);
end
"@ | Set-Content -LiteralPath $runner -Encoding ASCII
    Write-Host "[SPM12][START] $cid"
    # Keep the MATLAB -batch expression quoted as one Windows command-line
    # argument; case paths may contain spaces.
    $batchExpression = "run('$($runner.Replace('\','/'))')"
    $batchArgument = '"' + $batchExpression + '"'
    $proc = Start-Process -FilePath $Matlab -ArgumentList @('-batch', $batchArgument) -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
    $proc.WaitForExit()
    $c1 = @(Get-ChildItem -LiteralPath $caseOut -File -Filter 'c1*.nii' -ErrorAction SilentlyContinue)
    $c2 = @(Get-ChildItem -LiteralPath $caseOut -File -Filter 'c2*.nii' -ErrorAction SilentlyContinue)
    $c3 = @(Get-ChildItem -LiteralPath $caseOut -File -Filter 'c3*.nii' -ErrorAction SilentlyContinue)
    if ($c1.Count -gt 0 -and $c2.Count -gt 0 -and $c3.Count -gt 0) {
        $done++
        Write-Host "[SPM12][DONE] $cid"
    } else {
        $failed++
        Write-Host "[SPM12][FAILED] $cid log=$log err=$err"
    }
}
Write-Host "[SPM12] completed=$done failed=$failed requested=$($files.Count)"
if ($failed -gt 0 -or ($ExpectedCases -gt 0 -and $done -ne $ExpectedCases)) {
    throw "SPM12 segmentation incomplete: completed=$done failed=$failed expected=$ExpectedCases"
}

