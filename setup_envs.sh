#!/bin/bash
# 双环境安装脚本：vla-preprocess (Grounding DINO + SAM) 和 simplevla
# 核心策略：torch 单独先装，避免版本冲突

set -e
CONDA_ROOT="${HOME}/miniconda3"
# openvla-oft 根目录，可通过环境变量覆盖，例如: OPENVLA_ROOT=/你的路径 bash setup_envs.sh
OPENVLA_ROOT="${OPENVLA_ROOT:-${HOME}/16831pro_fine_tune/openvla-oft}"
echo "[INFO] openvla-oft 路径: $OPENVLA_ROOT (pip -e 仅链接，不覆盖源码)"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# 检测是否有 GPU
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    TORCH_EXTRA=""
    echo "[INFO] 检测到 GPU，安装 CUDA 版 PyTorch"
else
    TORCH_EXTRA="--index-url https://download.pytorch.org/whl/cpu"
    echo "[INFO] 未检测到 GPU，安装 CPU 版 PyTorch（更稳定）"
fi

# ========== 1. vla-preprocess (Grounding DINO + SAM) ==========
echo ""
echo "========== 创建 vla-preprocess 环境 =========="
conda create -n vla-preprocess python=3.9 -y
conda activate vla-preprocess

# 先单独装 torch，避免其他包拉错版本
echo "[vla-preprocess] 步骤 1/4: 安装 PyTorch（单独安装避免冲突）..."
pip install torch torchvision torchaudio $TORCH_EXTRA

echo "[vla-preprocess] 步骤 2/4: 安装基础依赖..."
pip install numpy scipy opencv-python pillow matplotlib pyyaml tqdm requests
pip install addict timm pycocotools supervision yapf
pip install "transformers>=4.33" "tokenizers" "huggingface-hub" "safetensors"

# TensorFlow（RLDS/libero 预处理用）
pip install "tensorflow>=2.14" "tensorflow-datasets" tensorboard

# Gripper 检测（mask_processor 用，需 ROBOFLOW_API_KEY）
pip install inference

# GroundingDINO：用项目内的（pip -e 只做链接，不覆盖源码）
GDINO_DIR="${OPENVLA_ROOT}/GroundingDINO"
if [ -d "$GDINO_DIR" ]; then
    echo "[vla-preprocess] 步骤 3/4: 安装 GroundingDINO..."
    pip install -e "$GDINO_DIR" --no-build-isolation
else
    echo "[WARN] GroundingDINO 目录不存在: $GDINO_DIR"
    echo "       请先确保 openvla-oft 已克隆，或手动: pip install -e /path/to/GroundingDINO"
fi

# Segment Anything：用项目内的（避免 git 安装的版本冲突）
SAM_DIR="${OPENVLA_ROOT}/segment-anything"
if [ -d "$SAM_DIR" ]; then
    echo "[vla-preprocess] 步骤 4/4: 安装 Segment Anything..."
    pip install -e "$SAM_DIR"
else
    echo "[WARN] segment-anything 目录不存在，尝试从 GitHub 安装..."
    pip install "git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"
fi

echo "[vla-preprocess] 环境安装完成"

# ========== 2. simplevla (OpenVLA-OFT 训练) ==========
echo ""
echo "========== 创建 simplevla 环境 =========="
conda deactivate
conda create -n simplevla python=3.10 -y
conda activate simplevla

echo "[simplevla] 步骤 1/2: 安装 PyTorch（单独安装避免冲突）..."
pip install torch torchvision torchaudio $TORCH_EXTRA

OVLA_DIR="${OPENVLA_ROOT}"
if [ -d "$OVLA_DIR" ]; then
    echo "[simplevla] 步骤 2/2: 安装 openvla-oft..."
    pip install -e "$OVLA_DIR"
else
    echo "[WARN] openvla-oft 目录不存在: $OVLA_DIR"
    echo "       请克隆后执行: pip install -e /path/to/openvla-oft"
fi

echo "[simplevla] 环境安装完成"
conda deactivate

echo ""
echo "========== 完成 =========="
echo "激活环境："
echo "  conda activate vla-preprocess   # Grounding DINO + SAM 预处理"
echo "  conda activate simplevla        # OpenVLA 训练"
