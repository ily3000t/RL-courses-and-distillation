param(
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$TeacherRun,
    [string]$Config = 'configs/stage1.json',
    [string]$StudentConfig = 'configs/distillation.json',
    [string]$Python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe',
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [ValidateRange(0, [int]::MaxValue)][int]$CollectionSeed = 104730,
    [ValidateRange(1, [int]::MaxValue)][int]$StepsPerScene = 64,
    [ValidateSet('validation', 'test', 'stress')][string]$Split = 'validation',
    [Nullable[int]]$StressSeed,
    [ValidateRange(1, [int]::MaxValue)][Nullable[int]]$StressCount,
    [string]$Device = '',
    [string]$ReuseScratch = ''
)

$ErrorActionPreference = 'Stop'
if ($Seeds.Count -eq 0 -or ($Seeds | Sort-Object -Unique).Count -ne $Seeds.Count -or
    ($Seeds | Where-Object { $_ -lt 0 })) {
    throw 'Provide unique nonnegative student seeds.'
}
if (($Split -eq 'stress') -ne $PSBoundParameters.ContainsKey('StressSeed')) {
    throw 'Set -StressSeed exactly when using -Split stress.'
}
if ($PSBoundParameters.ContainsKey('StressCount') -and $Split -ne 'stress') {
    throw 'Set -Split stress when supplying -StressCount.'
}
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (Test-Path -LiteralPath $outputPath) { throw "Output directory already exists: $outputPath" }
$teacherPath = (Resolve-Path -LiteralPath $TeacherRun).Path
$configPath = (Resolve-Path -LiteralPath $Config).Path
$studentPath = (Resolve-Path -LiteralPath $StudentConfig).Path
$scratchRoot = $outputPath
if ($ReuseScratch) {
    $scratchRoot = (Resolve-Path -LiteralPath $ReuseScratch).Path
    foreach ($seed in $Seeds) {
        if (-not (Test-Path -LiteralPath (Join-Path $scratchRoot "scratch/seed-$seed/train/best.zip") -PathType Leaf)) {
            throw "Missing reusable scratch checkpoint for seed $seed."
        }
    }
}
$deviceArgs = @()
if ($Device) { $deviceArgs = @('--device', $Device) }
$stressArgs = @()
if ($Split -eq 'stress') { $stressArgs = @('--stress-seed', $StressSeed) }
if ($PSBoundParameters.ContainsKey('StressCount')) { $stressArgs += @('--stress-count', $StressCount) }

function Invoke-Policy([string[]]$CommandArgs) {
    & $Python -B -m rl_courses @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw "Stage-two command failed: $($CommandArgs[0])" }
}

Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $collection = Join-Path $outputPath 'collection'
    Invoke-Policy @('collect-teacher', '--run', $teacherPath, '--seed', $CollectionSeed,
                    '--steps-per-scene', $StepsPerScene, '--output', $collection)
    $evalArgs = @('evaluate', '--split', $Split, '--checkpoint', 'best', '--benchmark-actor') + $stressArgs
    Invoke-Policy ($evalArgs + @('--run', $teacherPath, '--output', (Join-Path $outputPath "teacher/$Split")))
    $trainArgs = @('train-student', '--config', $configPath, '--student-config', $studentPath) + $deviceArgs
    foreach ($seed in $Seeds) {
        foreach ($method in @('scratch', 'bc', 'soft')) {
            $job = Join-Path $outputPath "$method/seed-$seed"
            $train = Join-Path $job 'train'
            if ($method -eq 'scratch' -and $ReuseScratch) {
                $train = Join-Path $scratchRoot "scratch/seed-$seed/train"
            } else {
                $dataArgs = @()
                if ($method -ne 'scratch') { $dataArgs = @('--data', $collection) }
                Invoke-Policy ($trainArgs + @('--method', $method, '--seed', $seed, '--output', $train) + $dataArgs)
            }
            Invoke-Policy ($evalArgs + @('--run', $train, '--output', (Join-Path $job $Split)))
        }
    }
    Invoke-Policy @('report-distillation', '--suite', $outputPath, '--split', $Split,
                    '--output', (Join-Path $outputPath 'report'))
}
finally {
    Pop-Location
}
