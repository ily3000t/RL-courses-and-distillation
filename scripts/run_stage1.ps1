param(
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$Config = 'configs/stage1.json',
    [string]$Python = 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe',
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [ValidateSet('baselines', 'curriculum', 'backoff')][string]$Comparison = 'baselines',
    [string]$Profiles = '',
    [ValidateSet('validation', 'test', 'stress')][string]$Split = 'validation',
    [Nullable[int]]$StressSeed,
    [string]$Device = '',
    [double]$SuccessThreshold = 0.90,
    [double]$CollisionThreshold = 0.05
)

$ErrorActionPreference = 'Stop'
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
$configPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Config)
if ($Seeds.Count -eq 0 -or (@($Seeds | Sort-Object -Unique)).Count -ne $Seeds.Count) {
    throw 'Provide at least one training seed and use each seed once.'
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Output directory already exists: $outputPath"
}
$deviceArgs = @()
if ($Device) {
    $deviceArgs = @('--device', $Device)
}
$methods = @{
    baselines = @('hard', 'uniform', 'fixed', 'shuffled', 'plr')
    curriculum = @('plr', 'density-progress', 'opportunity-progress')
    backoff = @('opportunity-progress', 'opportunity-progress-backoff')
}[$Comparison]
$profileArgs = @()
if ($Comparison -ne 'baselines') {
    if (-not $Profiles) { throw "Supply -Profiles for the $Comparison comparison." }
    $profilePath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Profiles)
    if (-not (Test-Path -LiteralPath $profilePath -PathType Leaf)) { throw "Missing profile: $profilePath" }
    $profileArgs = @('--profiles', $profilePath)
}
$holdoutArgs = @()
if ($PSBoundParameters.ContainsKey('StressSeed')) {
    if ($Split -ne 'stress') { throw 'Set -Split stress when supplying -StressSeed.' }
    $holdoutArgs = @('--stress-seed', $StressSeed)
}

Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    foreach ($method in $methods) {
        $methodArgs = @()
        if ($method -in @('opportunity-progress', 'opportunity-progress-backoff')) {
            $methodArgs = $profileArgs
        }
        foreach ($seed in $Seeds) {
            $job = Join-Path $outputPath "$method/seed-$seed"
            $train = Join-Path $job 'train'
            $evaluation = Join-Path $job $Split
            & $Python -B -m rl_courses train --config $configPath --method $method --seed $seed --output $train @deviceArgs @methodArgs
            if ($LASTEXITCODE -ne 0) { throw "Training failed for $method (seed $seed)." }
            & $Python -B -m rl_courses evaluate --run $train --split $Split --output $evaluation @holdoutArgs
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $method (seed $seed)." }
        }
    }
    & $Python -B -m rl_courses report --suite $outputPath --split $Split --comparison $Comparison --output (Join-Path $outputPath 'report') --success-threshold $SuccessThreshold --collision-threshold $CollisionThreshold
    if ($LASTEXITCODE -ne 0) { throw 'Generating the stage-one report failed.' }
}
finally {
    Pop-Location
}
