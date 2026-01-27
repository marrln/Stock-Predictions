#!/usr/bin/env bash
set -euo pipefail

# make_venv_tf.sh
# Creates a Python virtual environment named "venv-tf" and installs
# TensorFlow with GPU support plus required packages for this project.

VENV_DIR="venv-tf"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[INFO] Creating/using virtual environment at ${VENV_DIR}"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  echo "[INFO] Created venv: ${VENV_DIR}"
else
  echo "[INFO] Venv already exists; reusing ${VENV_DIR}"
fi

echo "[INFO] Activating venv"
source "${VENV_DIR}/bin/activate"

echo "[INFO] Upgrading packaging tools"
pip install --upgrade pip setuptools wheel

echo "[INFO] Installing core packages (TensorFlow GPU + dependencies)"
pip install "tensorflow[and-cuda]==2.15.*" \
           "numpy>=1.24,<2.0" \
           "pandas>=2.0" \
           "scikit-learn>=1.3" \
           "matplotlib>=3.7"

echo "[INFO] Verifying TensorFlow and GPU availability"
python - <<'PY'
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)
if not gpus:
    print("WARNING: No GPUs detected by TensorFlow.")
    print("Ensure NVIDIA driver is installed and 'nvidia-smi' works.")
PY

echo ""
echo "[DONE] venv is ready: ${VENV_DIR}"
echo "To activate:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Then run your training script, for example:"
echo "  python tf_train.py --tickers \"AAPL,MSFT,NVDA\" --visualize"
