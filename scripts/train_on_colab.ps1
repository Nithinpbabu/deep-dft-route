# DeepDFT-Route: Run Training on Google Colab Pro via CLI
# Usage:
#   .\scripts\train_on_colab.ps1
# Or to use A100 GPU:
#   .\scripts\train_on_colab.ps1 -Gpu A100

param (
    [string]$Gpu = "T4",
    [int]$Epochs = 60
)

$adcPath = "\\wsl$\Ubuntu\home\nithin\.config\gcloud\application_default_credentials.json"

if (-not (Test-Path $adcPath)) {
    Write-Host "`n[Step 1/2] One-time Google Cloud Authentication required for Colab Pro..." -ForegroundColor Yellow
    Write-Host "Opening your browser to log in with your Jio Colab Pro account...`n" -ForegroundColor Cyan
    wsl /home/nithin/google-cloud-sdk/bin/gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
}

Write-Host "`n[Step 2/2] Launching Colab Pro Cloud Training Runner ($Gpu GPU)...`n" -ForegroundColor Green
wsl /home/nithin/.venv_ml/bin/python3 /mnt/e/projects/DFT/scripts/run_colab_training.py --gpu $Gpu --epochs $Epochs
