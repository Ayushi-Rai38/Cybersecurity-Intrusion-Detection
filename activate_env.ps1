# ── Quick-start helper for the project virtual environment ──────────────────
# Run this script to activate .venv and confirm the environment is correct.
# Usage (from the project root in PowerShell):
#   .\activate_env.ps1

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvActivate = Join-Path $ProjectDir ".venv\Scripts\Activate.ps1"

if (-Not (Test-Path $VenvActivate)) {
    Write-Host "ERROR: .venv not found. Run first:" -ForegroundColor Red
    Write-Host "    python -m venv .venv" -ForegroundColor Yellow
    Write-Host "    .venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& $VenvActivate

Write-Host ""
Write-Host "── Environment Summary ──────────────────────────────────" -ForegroundColor Green
python --version
python -c "import pyspark; print('PySpark:', pyspark.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "PySpark: NOT YET INSTALLED — run pip install -r requirements.txt" -ForegroundColor Yellow }
Write-Host ""
Write-Host "── Run commands ─────────────────────────────────────────" -ForegroundColor Green
Write-Host "  Train + evaluate:    python main.py" -ForegroundColor White
Write-Host "  Launch dashboard:    streamlit run dashboard.py" -ForegroundColor White
Write-Host "─────────────────────────────────────────────────────────" -ForegroundColor Green
