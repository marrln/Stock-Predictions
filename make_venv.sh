#!/usr/bin/env bash
set -euo pipefail

# make_venv.sh - create a Python virtualenv and install PyTorch
# Usage: ./make_venv.sh [-n VENV_DIR] [-p PYTHON_BIN] [--cpu | --cuda CUVER] [--yes]
# Examples:
#   ./make_venv.sh                     # creates ./venv and installs CPU or GPU build (auto-detected)
#   ./make_venv.sh --cuda cu118 -n .venv
#   ./make_venv.sh --cpu -p python3.10

VENV_DIR="venv"
PYTHON_BIN="python3"
CUDA_MODE="auto"   # "auto", "cpu", or e.g. "cu118"
ASSUME_YES=0

usage() {
    cat <<EOF
Usage: $0 [-n VENV_DIR] [-p PYTHON_BIN] [--cpu | --cuda CUVER] [--yes]

Options:
    -n, --name VENV_DIR      virtualenv directory (default: venv)
    -p, --python PYTHON_BIN  python executable to use (default: python3)
            --cpu                force CPU-only PyTorch
            --cuda CUVER         force specific CUDA build (e.g. cu118, cu117)
            --yes                don't prompt before overwriting existing venv
    -h, --help               show this help
EOF
}

# simple arg parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name) VENV_DIR="$2"; shift 2;;
        -p|--python) PYTHON_BIN="$2"; shift 2;;
        --cpu) CUDA_MODE="cpu"; shift;;
        --cuda) CUDA_MODE="$2"; shift 2;;
        --yes) ASSUME_YES=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown arg: $1"; usage; exit 2;;
    esac
done

# Ensure python exists
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python executable "%s" not found in PATH\n' "$PYTHON_BIN" >&2
    exit 2
fi

# Handle existing venv
if [[ -d "$VENV_DIR" ]]; then
    if [[ $ASSUME_YES -eq 0 ]]; then
        printf 'Virtualenv "%s" already exists. Recreate? [y/N]: ' "$VENV_DIR"
        read -r resp
        if [[ ! "$resp" =~ ^[Yy] ]]; then
            printf 'Using existing virtualenv "%s".\n' "$VENV_DIR"
            # Activate and continue to install/upgrade packages
        else
            rm -rf "$VENV_DIR"
            printf 'Removed existing "%s".\n' "$VENV_DIR"
        fi
    else
        rm -rf "$VENV_DIR"
    fi
fi

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    printf 'Created virtualenv at %s\n' "$VENV_DIR"
fi

# Determine activation script path
ACTIVATE="$VENV_DIR/bin/activate"
if [[ ! -f "$ACTIVATE" ]]; then
    # Windows / scripts folder
    if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
        ACTIVATE="$VENV_DIR/Scripts/activate"
    fi
fi

# shellcheck disable=SC1090
# shellcheck source=/dev/null
source "$ACTIVATE"

python -m pip install --upgrade pip setuptools wheel

# decide which PyTorch wheel to install
if [[ "$CUDA_MODE" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # default to cu118 for modern drivers; user can override with --cuda
        CUDA_MODE="cu118"
    else
        CUDA_MODE="cpu"
    fi
fi

case "$CUDA_MODE" in
    cpu)
        PIP_EXTRA="--index-url https://download.pytorch.org/whl/cpu"
        ;;
    cu*)
        PIP_EXTRA="--index-url https://download.pytorch.org/whl/$CUDA_MODE"
        ;;
    *)
        echo "Unsupported CUDA option: $CUDA_MODE" >&2
        exit 2
        ;;
esac

printf 'Installing PyTorch (%s) into %s ...\n' "$CUDA_MODE" "$VENV_DIR"
python -m pip install "torch" "torchvision" "torchaudio" $PIP_EXTRA

# Install commonly used project libraries
PACKAGES=(pandas numpy matplotlib seaborn scikit-learn tqdm transformers sentencepiece nltk sumy torchsummary)
printf 'Installing additional Python packages: %s\n' "${PACKAGES[*]}"
python -m pip install "${PACKAGES[@]}"

# Ensure minimal NLTK data (punkt tokenizer) is available for scripts that need it
python - <<'PY'
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
PY

printf '\nDone. To activate the environment run:\n'
if [[ "$ACTIVATE" == */activate ]]; then
    printf '  source %s\n' "$ACTIVATE"
else
    printf '  %s\n' "$ACTIVATE"
fi
printf 'Verify with: python -c "import torch, pandas, numpy, transformers, nltk; print(\"torch:\", torch.__version__, torch.cuda.is_available()); print(\"pandas:\", pandas.__version__)"\n'