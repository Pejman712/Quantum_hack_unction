# One-command bootstrap for a fresh clone (Windows).
# Usage:  .\setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "Installing dependencies (uv sync --dev)..."
uv sync --dev

Write-Host "Enabling pre-push test hook..."
git config core.hooksPath .githooks

Write-Host "Setup complete. Tests will run automatically before every 'git push'."
