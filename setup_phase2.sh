#!/bin/bash
# ============================================================
# ROBOSHELF AI — Fázis 2 telepítési script
# LeRobot + G1 MuJoCo támogatás + új csomagok
# ============================================================
#
# Használat:
#   chmod +x setup_phase2.sh
#   ./setup_phase2.sh
#
# Feltételek:
#   - conda (miniforge) telepítve
#   - MuJoCo 3.6.0 már telepítve (Fázis 1-ből)
#   - Python 3.11+ (3.13 is OK)
# ============================================================

set -e

echo ""
echo "============================================="
echo "  ROBOSHELF AI — Fázis 2 telepítés"
echo "============================================="
echo ""

# --- 1. Meglévő környezet ellenőrzése ---
echo "📋 1. Meglévő környezet ellenőrzése..."
python3 -c "import mujoco; print(f'  MuJoCo: {mujoco.__version__}')" || {
    echo "  ❌ MuJoCo nincs telepítve! Először futtasd: pip install mujoco"
    exit 1
}
python3 -c "import gymnasium; print(f'  Gymnasium: {gymnasium.__version__}')"
python3 -c "import stable_baselines3; print(f'  SB3: {stable_baselines3.__version__}')"
echo "  ✅ Fázis 1 csomagok rendben"
echo ""

# --- 2. LeRobot telepítés G1 támogatással ---
echo "📦 2. LeRobot telepítés..."
pip install lerobot 2>/dev/null && echo "  ✅ LeRobot alap telepítve" || {
    echo "  ⚠️  LeRobot pip telepítés sikertelen, forrásból próbálom..."
    cd ~/Documents/roboshelf-ai-dev/
    if [ ! -d "lerobot" ]; then
        git clone https://github.com/huggingface/lerobot.git
    fi
    cd lerobot
    pip install -e ".[unitree_g1]"
    echo "  ✅ LeRobot forrásból telepítve G1 támogatással"
}

# Opcionális G1 extras (ha elérhető)
pip install lerobot[unitree_g1] 2>/dev/null && echo "  ✅ G1 extras telepítve" || echo "  ℹ️  G1 extras nem elérhető (nem kritikus)"
echo ""

# --- 3. Új Fázis 2 csomagok ---
echo "📦 3. Fázis 2 csomagok telepítése..."

# HuggingFace Hub (modell letöltésekhez)
pip install -q huggingface_hub
echo "  ✅ huggingface_hub"

# Robosuite (RoboCasa kompatibilitáshoz)
pip install -q robosuite 2>/dev/null && echo "  ✅ robosuite" || echo "  ℹ️  robosuite nem elérhető (nem kritikus)"

# Weights & Biases (opcionális monitoring)
pip install -q wandb 2>/dev/null && echo "  ✅ wandb" || echo "  ℹ️  wandb nem elérhető"

# ONNX export (policy exporthoz)
pip install -q onnx onnxruntime 2>/dev/null && echo "  ✅ onnx" || echo "  ℹ️  onnx nem elérhető"

echo ""

# --- 4. MuJoCo Menagerie G1 modell ellenőrzés ---
echo "🤖 4. G1 modell ellenőrzése..."
G1_PATHS=(
    "/opt/homebrew/Caskroom/miniforge/base/lib/python3.13/site-packages/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1/"
    "$(python3 -c 'import mujoco_playground; import os; print(os.path.join(os.path.dirname(mujoco_playground.__file__), "external_deps/mujoco_menagerie/unitree_g1/"))' 2>/dev/null)"
    "$(python3 -c 'import mujoco_menagerie; import os; print(os.path.join(os.path.dirname(mujoco_menagerie.__file__), "unitree_g1/"))' 2>/dev/null)"
)

G1_FOUND=""
for path in "${G1_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "${path}g1.xml" ]; then
        G1_FOUND="$path"
        break
    fi
done

if [ -n "$G1_FOUND" ]; then
    echo "  ✅ G1 modell megtalálva: $G1_FOUND"
    echo "  📄 g1.xml: ${G1_FOUND}g1.xml"
else
    echo "  ⚠️  G1 modell nem található automatikusan."
    echo "  Kézi ellenőrzés szükséges. Keresés:"
    echo "    find / -name 'g1.xml' -path '*/unitree_g1/*' 2>/dev/null | head -5"
fi
echo ""

# --- 5. Projekt struktúra frissítése ---
echo "📁 5. Fázis 2 projekt struktúra..."
PROJ_DIR=~/Documents/roboshelf-ai-dev/roboshelf-ai

mkdir -p "$PROJ_DIR/src/envs"
mkdir -p "$PROJ_DIR/src/envs/assets"
mkdir -p "$PROJ_DIR/src/training"
mkdir -p "$PROJ_DIR/src/evaluation"
mkdir -p "$PROJ_DIR/configs"
mkdir -p "$PROJ_DIR/data/demonstrations"
mkdir -p "$PROJ_DIR/data/models"

echo "  ✅ Könyvtárak létrehozva:"
echo "    src/envs/         — MJCF környezetek és Gymnasium wrapperek"
echo "    src/envs/assets/  — MJCF XML fájlok, textúrák"
echo "    configs/          — Domain randomization konfigurációk"
echo "    data/demos/       — Demonstrációs adatok (később)"
echo "    data/models/      — Letöltött modellek (GR00T, SmolVLA)"
echo ""

# --- 6. Verzió-összefoglaló ---
echo "============================================="
echo "  Fázis 2 telepítés kész!"
echo "============================================="
echo ""
python3 -c "
import sys
print(f'  Python: {sys.version.split()[0]}')

import mujoco; print(f'  MuJoCo: {mujoco.__version__}')
import gymnasium; print(f'  Gymnasium: {gymnasium.__version__}')
import stable_baselines3; print(f'  SB3: {stable_baselines3.__version__}')
import torch; print(f'  PyTorch: {torch.__version__}')
import jax; print(f'  JAX: {jax.__version__}')
import numpy; print(f'  NumPy: {numpy.__version__}')

try:
    import lerobot; print(f'  LeRobot: {lerobot.__version__}')
except: print('  LeRobot: (forrásból telepítve)')

try:
    import huggingface_hub; print(f'  HF Hub: {huggingface_hub.__version__}')
except: pass
"
echo ""
echo "  Következő lépés:"
echo "    python3 src/envs/roboshelf_retail_scene.py"
echo ""
