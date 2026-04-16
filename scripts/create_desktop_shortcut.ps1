$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$launcherPath = Join-Path $repoRoot 'launch_mvp.bat'
$desktopPath = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktopPath 'Mouse Vision MVP Launcher.lnk'

if (-not (Test-Path $launcherPath)) {
    throw "Launcher not found: $launcherPath"
}

$wshShell = New-Object -ComObject WScript.Shell
$shortcut = $wshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcherPath
$shortcut.WorkingDirectory = $repoRoot
$shortcut.IconLocation = "$env:SystemRoot\System32\SHELL32.dll,44"
$shortcut.Description = 'Launch Mouse Vision MVP tools'
$shortcut.Save()

Write-Host "Created shortcut: $shortcutPath"
