<#
.SYNOPSIS
    Downscale the 4K roadside clips to 1600x900 H.264, mirroring the folder
    structure. Keeps clips SEPARATE (no merging) and preserves frame counts,
    so (camera, clip, local-frame) addressing and the 1600x900 homography stay
    valid.

.NOTES
    Requires ffmpeg on PATH (CPU libx264 path -- no GPU needed).
    Re-runnable: existing outputs are skipped unless -Overwrite is given.

.EXAMPLE
    .\downscale_clips.ps1
    .\downscale_clips.ps1 -Crf 23 -Overwrite
#>
param(
    [string]$Src = "C:\Users\praba\PycharmProjects\AvaCar\OpenCV Practice\roadside_video",
    [string]$Dst = "C:\Users\praba\PycharmProjects\AvaCar\OpenCV Practice\roadside_video_1600x900",
    [int]$Crf = 20,                 # quality: lower = better/bigger (18-23 sensible)
    [switch]$Overwrite              # re-encode even if the output already exists
)

$cameras = "c0", "c1", "c2", "c3", "c4"

foreach ($cam in $cameras) {
    $inDir  = Join-Path $Src $cam
    $outDir = Join-Path $Dst $cam
    if (-not (Test-Path $inDir)) { Write-Host "skip (no source folder): $inDir"; continue }
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    # Sort so processing order is stable (not required for correctness).
    Get-ChildItem -Path $inDir -Filter *.MP4 | Sort-Object Name | ForEach-Object {
        $in  = $_.FullName
        $out = Join-Path $outDir ($_.BaseName + ".mp4")

        if ((Test-Path $out) -and -not $Overwrite) {
            Write-Host "exists, skipping: $out"
            return
        }

        Write-Host "encoding $($_.Name)  ->  $out"
        ffmpeg -hide_banner -loglevel warning -stats -y `
            -i $in `
            -vf scale=1600:900 `
            -c:v libx264 -crf $Crf -preset veryfast `
            -an `
            $out

        if ($LASTEXITCODE -ne 0) { Write-Warning "ffmpeg FAILED on $in" }
    }
}

Write-Host "Done."
