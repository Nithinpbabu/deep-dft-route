#!/usr/bin/env bash
# DeepDFT-Route: Run Training on Google Colab Pro via CLI

GPU=${1:-"T4"}
EPOCHS=${2:-30}

ADC_PATH="$HOME/.config/gcloud/application_default_credentials.json"

if [ ! -f "$ADC_PATH" ]; then
    echo -e "\n[Step 1/2] One-time Google Cloud Authentication required for Colab Pro..."
    echo -e "Opening your browser to log in with your Jio Colab Pro account...\n"
    $HOME/google-cloud-sdk/bin/gcloud auth application-default login \
        --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
fi

echo -e "\n[Step 2/2] Launching Colab Pro Cloud Training Runner ($GPU GPU)...\n"
python3 /mnt/e/projects/DFT/scripts/run_colab_training.py --gpu "$GPU" --epochs "$EPOCHS"
