# Export mermaid .mmd files to SVG + PNG via mermaid.ink
param(
    [string[]]$MmdFiles
)

$ErrorActionPreference = "Stop"

function Export-Mermaid([string]$mmdPath) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($mmdPath)
    $dir = [System.IO.Path]::GetDirectoryName($mmdPath)
    $code = [System.IO.File]::ReadAllText($mmdPath, [System.Text.Encoding]::UTF8)
    $state = @{ code = $code; mermaid = @{ theme = "default"; themeVariables = @{ fontSize = "14px" } } } | ConvertTo-Json -Depth 5 -Compress
    $b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($state))
    # URL-safe base64 (base64url): mermaid.ink requires it, otherwise / + = break the path
    $b64url = $b64.TrimEnd('=').Replace('+', '-').Replace('/', '_')

    $svgOut = Join-Path $dir "$base.svg"
    $pngOut = Join-Path $dir "$base.png"

    # SVG
    $resp = Invoke-WebRequest -Uri "https://mermaid.ink/svg/$b64url" -TimeoutSec 120 -UseBasicParsing
    [System.IO.File]::WriteAllText($svgOut, $resp.Content, [System.Text.Encoding]::UTF8)
    Write-Host "SVG OK: $svgOut ($($resp.RawContentLength) bytes)"

    # PNG
    $resp2 = Invoke-WebRequest -Uri "https://mermaid.ink/img/$b64url" -TimeoutSec 120 -UseBasicParsing
    if ($resp2.RawContentStream) {
        $bytes = New-Object byte[] $resp2.RawContentLength
        $resp2.RawContentStream.Read($bytes, 0, $bytes.Length) | Out-Null
        [System.IO.File]::WriteAllBytes($pngOut, $bytes)
    } else {
        [System.IO.File]::WriteAllBytes($pngOut, [System.Text.Encoding]::UTF8.GetBytes($resp2.Content))
    }
    Write-Host "PNG OK: $pngOut ($($resp2.RawContentLength) bytes)"
}

foreach ($f in $MmdFiles) {
    Write-Host "=== $f ==="
    Export-Mermaid $f
}
