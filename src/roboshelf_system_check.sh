#!/bin/bash
# ============================================================================
#  ROBOSHELF AI – Rendszerdiagnosztika
#  MacBook Air M2 előkészítés humanoid robot betanításhoz
#  Futtatás: chmod +x roboshelf_system_check.sh && ./roboshelf_system_check.sh
# ============================================================================

# Színek
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

PASS="${GREEN}✅ MEGVAN${NC}"
FAIL="${RED}❌ HIÁNYZIK${NC}"
WARN="${YELLOW}⚠️  FIGYELEM${NC}"
INFO="${CYAN}ℹ️ ${NC}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       ROBOSHELF AI – Rendszerdiagnosztika v1.0              ║"
echo "║       MacBook Air M2 – Humanoid Robot Training Setup        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Számlálók
INSTALLED=0
MISSING=0
WARNINGS=0

check_pass() {
    ((INSTALLED++))
    echo -e "  $PASS  $1"
}

check_fail() {
    ((MISSING++))
    echo -e "  $FAIL  $1"
}

check_warn() {
    ((WARNINGS++))
    echo -e "  $WARN  $1"
}

# ============================================================================
# 1. HARDVER ÉS OS
# ============================================================================
echo -e "${BOLD}━━━ 1. HARDVER ÉS OPERÁCIÓS RENDSZER ━━━${NC}"
echo ""

# macOS verzió
MACOS_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "N/A")
echo -e "  ${INFO}macOS verzió: ${BOLD}$MACOS_VERSION${NC}"

# Chip
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "N/A")
if echo "$CHIP" | grep -qi "Apple"; then
    check_pass "Apple Silicon chip: $CHIP"
else
    check_warn "Nem Apple Silicon chip: $CHIP (MJX-JAX limitált lehet)"
fi

# RAM
RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1073741824}')
echo -e "  ${INFO}RAM: ${BOLD}${RAM_GB} GB${NC} (unified memory)"
if [ "$RAM_GB" -ge 16 ]; then
    check_pass "RAM elegendő (≥16 GB)"
else
    check_warn "RAM kevés (<16 GB) – nagyobb modelleknél korlát lehet"
fi

# Szabad lemezterület
FREE_DISK=$(df -h / | tail -1 | awk '{print $4}')
echo -e "  ${INFO}Szabad lemezterület: ${BOLD}$FREE_DISK${NC}"

# GPU info (Apple Silicon)
GPU_CORES=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -i "Total Number of Cores" | head -1 | awk -F: '{print $2}' | tr -d ' ')
if [ -n "$GPU_CORES" ]; then
    echo -e "  ${INFO}GPU magok: ${BOLD}$GPU_CORES${NC}"
else
    echo -e "  ${INFO}GPU magok: (nem elérhető)"
fi

echo ""

# ============================================================================
# 2. PYTHON KÖRNYEZET
# ============================================================================
echo -e "${BOLD}━━━ 2. PYTHON KÖRNYEZET ━━━${NC}"
echo ""

# Python 3
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
    if [ "$PY_MINOR" -ge 11 ] 2>/dev/null; then
        check_pass "Python: $PY_VERSION (≥3.11, MuJoCo Playground kompatibilis)"
    elif [ "$PY_MINOR" -ge 10 ] 2>/dev/null; then
        check_pass "Python: $PY_VERSION (≥3.10, MuJoCo kompatibilis)"
        check_warn "MuJoCo Playground Python ≥3.11-et ajánl"
    else
        check_warn "Python: $PY_VERSION (MuJoCo Playground ≥3.11-et igényel)"
    fi
else
    check_fail "Python 3 nincs telepítve"
fi

# pip
if command -v pip3 &>/dev/null; then
    PIP_VERSION=$(pip3 --version 2>&1 | awk '{print $2}')
    check_pass "pip: v$PIP_VERSION"
else
    check_fail "pip3 nincs telepítve"
fi

# Virtuális környezet ellenőrzés
if [ -n "$VIRTUAL_ENV" ]; then
    check_pass "Virtuális környezet aktív: $VIRTUAL_ENV"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    check_pass "Conda környezet aktív: $CONDA_DEFAULT_ENV"
else
    check_warn "Nincs aktív virtuális környezet (ajánlott: venv vagy conda)"
fi

# conda
if command -v conda &>/dev/null; then
    CONDA_VER=$(conda --version 2>&1)
    check_pass "Conda: $CONDA_VER"
else
    echo -e "  ${INFO}Conda nincs telepítve (opcionális, venv is megfelelő)"
fi

echo ""

# ============================================================================
# 3. MUJOCO ÉS KAPCSOLÓDÓ CSOMAGOK
# ============================================================================
echo -e "${BOLD}━━━ 3. MUJOCO ÉS FIZIKAI SZIMULÁCIÓ ━━━${NC}"
echo ""

# MuJoCo
MUJOCO_VER=$(python3 -c "import mujoco; print(mujoco.__version__)" 2>/dev/null)
if [ -n "$MUJOCO_VER" ]; then
    check_pass "MuJoCo: v$MUJOCO_VER"
    # Verzió ellenőrzés (≥3.0 kell MJX-hez)
    MUJOCO_MAJOR=$(echo "$MUJOCO_VER" | cut -d. -f1)
    if [ "$MUJOCO_MAJOR" -ge 3 ] 2>/dev/null; then
        check_pass "MuJoCo verzió ≥3.0 (MJX kompatibilis)"
    else
        check_warn "MuJoCo verzió <3.0 – frissítés ajánlott MJX-hez"
    fi
else
    check_fail "MuJoCo Python csomag nincs telepítve"
fi

# MuJoCo MJX (JAX backend)
MJX_VER=$(python3 -c "from mujoco import mjx; print('OK')" 2>/dev/null)
if [ "$MJX_VER" = "OK" ]; then
    check_pass "MuJoCo MJX (JAX backend) elérhető"
else
    check_fail "MuJoCo MJX nincs telepítve (pip install mujoco-mjx)"
fi

# MuJoCo Playground
PLAYGROUND_VER=$(python3 -c "import mujoco_playground; print('OK')" 2>/dev/null)
if [ "$PLAYGROUND_VER" = "OK" ]; then
    check_pass "MuJoCo Playground telepítve"
else
    check_fail "MuJoCo Playground nincs telepítve (pip install playground)"
fi

# MuJoCo vizualizáció teszt
MUJOCO_RENDER=$(python3 -c "
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light diffuse=\"1 1 1\"/><geom type=\"sphere\" size=\"1\"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print('OK')
" 2>/dev/null)
if [ "$MUJOCO_RENDER" = "OK" ]; then
    check_pass "MuJoCo fizikai szimuláció működik"
else
    check_warn "MuJoCo alapvető szimuláció nem futtatható (telepítési probléma?)"
fi

echo ""

# ============================================================================
# 4. JAX ÉS ML KERETRENDSZEREK
# ============================================================================
echo -e "${BOLD}━━━ 4. JAX ÉS MACHINE LEARNING KERETRENDSZEREK ━━━${NC}"
echo ""

# JAX
JAX_VER=$(python3 -c "import jax; print(jax.__version__)" 2>/dev/null)
if [ -n "$JAX_VER" ]; then
    check_pass "JAX: v$JAX_VER"
    # Backend ellenőrzés
    JAX_BACKEND=$(python3 -c "import jax; print(jax.default_backend())" 2>/dev/null)
    echo -e "  ${INFO}JAX backend: ${BOLD}$JAX_BACKEND${NC}"
    if [ "$JAX_BACKEND" = "gpu" ] || [ "$JAX_BACKEND" = "metal" ]; then
        check_pass "JAX GPU/Metal backend aktív"
    elif [ "$JAX_BACKEND" = "cpu" ]; then
        check_warn "JAX CPU backend aktív (Apple Silicon-on a Metal backend gyorsabb lehet)"
    fi
    # JAX Metal plugin
    JAX_METAL=$(python3 -c "import jax_metal; print('OK')" 2>/dev/null)
    if [ "$JAX_METAL" = "OK" ]; then
        check_pass "jax-metal plugin telepítve (Apple GPU gyorsítás)"
    else
        echo -e "  ${INFO}jax-metal plugin nincs telepítve (opcionális: pip install jax-metal)"
    fi
else
    check_fail "JAX nincs telepítve (pip install jax jaxlib)"
fi

# PyTorch
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ -n "$TORCH_VER" ]; then
    check_pass "PyTorch: v$TORCH_VER"
    # MPS (Apple Silicon GPU) elérhetőség
    TORCH_MPS=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null)
    if [ "$TORCH_MPS" = "True" ]; then
        check_pass "PyTorch MPS (Apple GPU) backend elérhető"
    else
        check_warn "PyTorch MPS backend nem elérhető"
    fi
else
    check_fail "PyTorch nincs telepítve (szükséges StableBaselines3-hoz és GR00T-hoz)"
fi

# TensorFlow (opcionális)
TF_VER=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
if [ -n "$TF_VER" ]; then
    check_pass "TensorFlow: v$TF_VER (opcionális)"
else
    echo -e "  ${INFO}TensorFlow nincs telepítve (opcionális)"
fi

echo ""

# ============================================================================
# 5. REINFORCEMENT LEARNING KÖNYVTÁRAK
# ============================================================================
echo -e "${BOLD}━━━ 5. REINFORCEMENT LEARNING KÖNYVTÁRAK ━━━${NC}"
echo ""

# Gymnasium
GYM_VER=$(python3 -c "import gymnasium; print(gymnasium.__version__)" 2>/dev/null)
if [ -n "$GYM_VER" ]; then
    check_pass "Gymnasium: v$GYM_VER"
else
    check_fail "Gymnasium nincs telepítve (pip install gymnasium)"
fi

# Gymnasium MuJoCo environments
GYM_MUJOCO=$(python3 -c "import gymnasium; env = gymnasium.make('Humanoid-v5'); print('OK')" 2>/dev/null)
if [ "$GYM_MUJOCO" = "OK" ]; then
    check_pass "Gymnasium MuJoCo környezetek elérhetők (Humanoid-v5)"
else
    check_fail "Gymnasium MuJoCo env-ek nem elérhetők (pip install gymnasium[mujoco])"
fi

# Stable-Baselines3
SB3_VER=$(python3 -c "import stable_baselines3; print(stable_baselines3.__version__)" 2>/dev/null)
if [ -n "$SB3_VER" ]; then
    check_pass "Stable-Baselines3: v$SB3_VER"
else
    check_fail "Stable-Baselines3 nincs telepítve (pip install stable-baselines3)"
fi

# Brax (opcionális, MuJoCo Playground használja)
BRAX_VER=$(python3 -c "import brax; print(brax.__version__)" 2>/dev/null)
if [ -n "$BRAX_VER" ]; then
    check_pass "Brax: v$BRAX_VER"
else
    echo -e "  ${INFO}Brax nincs telepítve (a Playground automatikusan kezeli)"
fi

echo ""

# ============================================================================
# 6. VIZUALIZÁCIÓ ÉS MONITORING
# ============================================================================
echo -e "${BOLD}━━━ 6. VIZUALIZÁCIÓ ÉS MONITORING ━━━${NC}"
echo ""

# TensorBoard
TB_VER=$(python3 -c "import tensorboard; print(tensorboard.__version__)" 2>/dev/null)
if [ -n "$TB_VER" ]; then
    check_pass "TensorBoard: v$TB_VER"
else
    check_fail "TensorBoard nincs telepítve (pip install tensorboard)"
fi

# Weights & Biases (opcionális)
WANDB_VER=$(python3 -c "import wandb; print(wandb.__version__)" 2>/dev/null)
if [ -n "$WANDB_VER" ]; then
    check_pass "Weights & Biases: v$WANDB_VER"
else
    echo -e "  ${INFO}Weights & Biases nincs telepítve (opcionális: pip install wandb)"
fi

# Matplotlib
MPL_VER=$(python3 -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null)
if [ -n "$MPL_VER" ]; then
    check_pass "Matplotlib: v$MPL_VER"
else
    check_fail "Matplotlib nincs telepítve (pip install matplotlib)"
fi

# OpenCV (opcionális, videó rendereléshez)
CV2_VER=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null)
if [ -n "$CV2_VER" ]; then
    check_pass "OpenCV: v$CV2_VER"
else
    echo -e "  ${INFO}OpenCV nincs telepítve (opcionális, videó rendereléshez: pip install opencv-python)"
fi

echo ""

# ============================================================================
# 7. EGYÉB HASZNOS ESZKÖZÖK
# ============================================================================
echo -e "${BOLD}━━━ 7. EGYÉB ESZKÖZÖK ━━━${NC}"
echo ""

# Git
if command -v git &>/dev/null; then
    GIT_VER=$(git --version | awk '{print $3}')
    check_pass "Git: v$GIT_VER"
else
    check_fail "Git nincs telepítve"
fi

# Homebrew
if command -v brew &>/dev/null; then
    BREW_VER=$(brew --version | head -1 | awk '{print $2}')
    check_pass "Homebrew: v$BREW_VER"
else
    check_warn "Homebrew nincs telepítve (ajánlott csomagkezelő macOS-en)"
fi

# Node.js (opcionális, korábbi eszközökhöz)
if command -v node &>/dev/null; then
    NODE_VER=$(node --version)
    check_pass "Node.js: $NODE_VER (opcionális)"
else
    echo -e "  ${INFO}Node.js nincs telepítve (opcionális)"
fi

# GLFW (MuJoCo vizualizációhoz)
GLFW_CHECK=$(python3 -c "import glfw; print(glfw.get_version())" 2>/dev/null)
if [ -n "$GLFW_CHECK" ]; then
    check_pass "GLFW (ablakkezelő): $GLFW_CHECK"
else
    echo -e "  ${INFO}GLFW Python binding nincs telepítve (pip install glfw, opcionális vizualizációhoz)"
fi

# Webots (korábbi szimulátor)
if command -v webots &>/dev/null || [ -d "/Applications/Webots.app" ]; then
    check_pass "Webots telepítve (korábbi szimulációs munka)"
else
    echo -e "  ${INFO}Webots nincs telepítve (nem szükséges a MuJoCo pipeline-hoz)"
fi

echo ""

# ============================================================================
# 8. NVIDIA GR00T N1 KOMPATIBILITÁS
# ============================================================================
echo -e "${BOLD}━━━ 8. NVIDIA GR00T N1 KOMPATIBILITÁS ━━━${NC}"
echo ""

# Hugging Face Hub
HF_VER=$(python3 -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null)
if [ -n "$HF_VER" ]; then
    check_pass "Hugging Face Hub: v$HF_VER (GR00T modell letöltéshez)"
else
    check_fail "Hugging Face Hub nincs telepítve (pip install huggingface_hub)"
fi

# Transformers
TRANS_VER=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
if [ -n "$TRANS_VER" ]; then
    check_pass "Transformers: v$TRANS_VER"
else
    echo -e "  ${INFO}Transformers nincs telepítve (pip install transformers – GR00T fine-tuninghoz kell)"
fi

# Diffusers
DIFF_VER=$(python3 -c "import diffusers; print(diffusers.__version__)" 2>/dev/null)
if [ -n "$DIFF_VER" ]; then
    check_pass "Diffusers: v$DIFF_VER (GR00T diffúziós modellhez)"
else
    echo -e "  ${INFO}Diffusers nincs telepítve (pip install diffusers – később szükséges)"
fi

# LeRobot (GR00T adatformátum)
LEROBOT=$(python3 -c "import lerobot; print('OK')" 2>/dev/null)
if [ "$LEROBOT" = "OK" ]; then
    check_pass "LeRobot telepítve (GR00T adatformátum)"
else
    echo -e "  ${INFO}LeRobot nincs telepítve (később szükséges GR00T fine-tuninghoz)"
fi

echo -e "  ${INFO}Megjegyzés: A GR00T N1.6 inference NVIDIA GPU-t igényel (nem fut M2-n)."
echo -e "  ${INFO}A modell fine-tuning felhős GPU-n történik (Colab/Lambda Labs)."

echo ""

# ============================================================================
# ÖSSZEGZÉS
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      ÖSSZEGZÉS                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "  ${GREEN}Telepítve:${NC}  $INSTALLED komponens"
echo -e "  ${RED}Hiányzik:${NC}   $MISSING komponens"
echo -e "  ${YELLOW}Figyelem:${NC}   $WARNINGS figyelmeztetés"
echo ""

if [ $MISSING -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}🎉 Minden szükséges komponens telepítve van!${NC}"
    echo -e "  Készen állsz a Fázis 1 megkezdésére."
else
    echo -e "  ${YELLOW}${BOLD}📋 A hiányzó komponensek telepítéséhez futtasd az alábbi parancsot:${NC}"
    echo ""
    echo "  ─────────────────────────────────────────────────────"
    echo ""
    echo "  # 1. Virtuális környezet létrehozása (ha még nincs)"
    echo "  python3 -m venv ~/roboshelf-sim"
    echo "  source ~/roboshelf-sim/bin/activate"
    echo ""
    echo "  # 2. Alapcsomagok"
    echo "  pip install --upgrade pip"
    echo "  pip install mujoco mujoco-mjx"
    echo "  pip install jax jaxlib"
    echo ""
    echo "  # 3. MuJoCo Playground"
    echo "  pip install playground"
    echo ""
    echo "  # 4. RL és ML keretrendszerek"
    echo "  pip install gymnasium[mujoco]"
    echo "  pip install stable-baselines3"
    echo "  pip install torch torchvision"
    echo ""
    echo "  # 5. Monitoring és vizualizáció"
    echo "  pip install tensorboard matplotlib"
    echo ""
    echo "  # 6. GR00T előkészítés (opcionális most)"
    echo "  pip install huggingface_hub transformers"
    echo ""
    echo "  # 7. Opcionális de ajánlott"
    echo "  pip install wandb opencv-python glfw"
    echo ""
    echo "  # 8. Opcionális: JAX Metal plugin (Apple GPU)"
    echo "  # pip install jax-metal"
    echo ""
    echo "  ─────────────────────────────────────────────────────"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Roboshelf AI – Rendszerdiagnosztika kész."
echo "  Az eredményt érdemes elmenteni: ./roboshelf_system_check.sh > check_results.txt 2>&1"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
