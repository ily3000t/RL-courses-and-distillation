param(
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$TeacherRun,
    [string]$Config = 'configs/stage1.json',
    [string]$StudentConfig = 'configs/distillation.json',
    [string]$Python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe',
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [ValidateRange(0, [int]::MaxValue)][int]$CollectionSeed = 104730,
    [ValidateRange(1, [int]::MaxValue)][int]$StepsPerScene = 64,
    [ValidateSet('truncate', 'full-episode')][string]$CollectionMode = 'truncate',
    [ValidateSet('validation', 'test', 'stress')][string[]]$Split = @('validation'),
    [Nullable[int]]$StressSeed,
    [ValidateRange(1, [int]::MaxValue)][Nullable[int]]$StressCount,
    [string]$Device = '',
    [string]$ReuseScratch = '',
    [string]$ReuseDistilled = ''
)

$ErrorActionPreference = 'Stop'
if ($Seeds.Count -eq 0 -or ($Seeds | Sort-Object -Unique).Count -ne $Seeds.Count -or
    ($Seeds | Where-Object { $_ -lt 0 })) {
    throw 'Provide unique nonnegative student seeds.'
}
if ($Split.Count -eq 0 -or ($Split | Sort-Object -Unique).Count -ne $Split.Count) {
    throw 'Provide a nonempty list of distinct evaluation splits.'
}
if (($Split -contains 'stress') -ne $PSBoundParameters.ContainsKey('StressSeed')) {
    throw 'Set -StressSeed exactly when stress is one of the evaluation splits.'
}
if ($PSBoundParameters.ContainsKey('StressCount') -and $Split -notcontains 'stress') {
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
$distilledRoot = $outputPath
if ($ReuseDistilled) {
    $distilledRoot = (Resolve-Path -LiteralPath $ReuseDistilled).Path
    foreach ($seed in $Seeds) {
        foreach ($method in @('bc', 'soft')) {
            if (-not (Test-Path -LiteralPath (Join-Path $distilledRoot "$method/seed-$seed/train/best.zip") -PathType Leaf)) {
                throw "Missing reusable $method checkpoint for seed $seed."
            }
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

function Invoke-Evaluation([string]$Run, [string]$Job) {
    foreach ($partition in $Split) {
        $evalArgs = @('evaluate', '--split', $partition, '--checkpoint', 'best', '--benchmark-actor',
                      '--run', $Run, '--output', (Join-Path $Job $partition))
        if ($partition -eq 'stress') { $evalArgs += $stressArgs }
        Invoke-Policy $evalArgs
    }
}

Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $collection = Join-Path $outputPath 'collection'
    Invoke-Policy @('collect-teacher', '--run', $teacherPath, '--seed', $CollectionSeed,
                    '--steps-per-scene', $StepsPerScene, '--collection-mode', $CollectionMode,
                    '--output', $collection)
    if ($ReuseDistilled) {
        $dataset = Get-Content -LiteralPath (Join-Path $collection 'collection_summary.json') -Raw | ConvertFrom-Json
        foreach ($seed in $Seeds) {
            foreach ($method in @('bc', 'soft')) {
                $train = Join-Path $distilledRoot "$method/seed-$seed/train"
                $status = Get-Content -LiteralPath (Join-Path $train 'status.json') -Raw | ConvertFrom-Json
                $manifest = Get-Content -LiteralPath (Join-Path $train 'manifest.json') -Raw | ConvertFrom-Json
                if ($status.state -ne 'completed' -or $manifest.teacher_data.dataset_sha256 -ne $dataset.dataset_sha256) {
                    throw "Reusable $method student seed $seed differs from the freshly collected dataset."
                }
            }
        }
    }
    Invoke-Evaluation $teacherPath (Join-Path $outputPath 'teacher')
    $trainArgs = @('train-student', '--config', $configPath, '--student-config', $studentPath) + $deviceArgs
    foreach ($seed in $Seeds) {
        foreach ($method in @('scratch', 'bc', 'soft')) {
            $job = Join-Path $outputPath "$method/seed-$seed"
            $train = Join-Path $job 'train'
            if ($method -eq 'scratch' -and $ReuseScratch) {
                $train = Join-Path $scratchRoot "scratch/seed-$seed/train"
            } elseif ($method -ne 'scratch' -and $ReuseDistilled) {
                $train = Join-Path $distilledRoot "$method/seed-$seed/train"
            } else {
                $dataArgs = @()
                if ($method -ne 'scratch') { $dataArgs = @('--data', $collection) }
                Invoke-Policy ($trainArgs + @('--method', $method, '--seed', $seed, '--output', $train) + $dataArgs)
            }
            Invoke-Evaluation $train $job
        }
    }
    foreach ($partition in $Split) {
        $report = if ($Split.Count -eq 1 -or $partition -eq 'validation') { 'report' } else { "report-$partition" }
        Invoke-Policy @('report-distillation', '--suite', $outputPath, '--split', $partition,
                        '--output', (Join-Path $outputPath $report))
    }
}
finally {
    Pop-Location
}
