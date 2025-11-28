#!/bin/bash
# WSL Environment Verification Script
# Ensures all requirements are met before starting quantization experiments

set -e  # Exit on error

echo "================================================"
echo "Advanced Quantization Experiment - Environment Check"
echo "================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Linux/WSL
echo "🔍 Checking operating system..."
if [[ "$(uname -s)" == "Linux" ]]; then
    echo -e "${GREEN}✅ Running on Linux${NC}"
    if [[ -n "$WSL_DISTRO_NAME" ]]; then
        echo -e "${GREEN}   WSL Distribution: $WSL_DISTRO_NAME${NC}"
    fi
else
    echo -e "${RED}❌ NOT running on Linux!${NC}"
    echo -e "${RED}   Current OS: $(uname -s)${NC}"
    echo -e "${YELLOW}   Please run this in WSL (Windows Subsystem for Linux)${NC}"
    exit 1
fi
echo ""

# Check 2: Python
echo "🔍 Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Python installed: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi
echo ""

# Check 3: CUDA
echo "🔍 Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA drivers installed${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
        echo -e "${GREEN}   GPU: $line${NC}"
    done
else
    echo -e "${YELLOW}⚠️  nvidia-smi not found${NC}"
    echo -e "${YELLOW}   GPU quantization may not work. CPU fallback available.${NC}"
fi
echo ""

# Check 4: PyTorch CUDA
echo "🔍 Checking PyTorch CUDA support..."
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "error")
if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo -e "${GREEN}✅ PyTorch CUDA available${NC}"
    echo -e "${GREEN}   CUDA version: $CUDA_VERSION${NC}"
    echo -e "${GREEN}   GPU count: $GPU_COUNT${NC}"
elif [[ "$CUDA_AVAILABLE" == "False" ]]; then
    echo -e "${YELLOW}⚠️  PyTorch installed but CUDA not available${NC}"
    echo -e "${YELLOW}   Will use CPU (slower but functional)${NC}"
else
    echo -e "${RED}❌ PyTorch not installed${NC}"
    echo -e "${YELLOW}   Install with: pip install torch torchvision torchaudio${NC}"
    exit 1
fi
echo ""

# Check 5: Quantization libraries
echo "🔍 Checking quantization libraries..."

check_package() {
    if python3 -c "import $1" 2>/dev/null; then
        VERSION=$(python3 -c "import $1; print($1.__version__)" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✅ $1 installed (version: $VERSION)${NC}"
        return 0
    else
        echo -e "${RED}❌ $1 not installed${NC}"
        return 1
    fi
}

# Core libraries
check_package "transformers"
check_package "accelerate"
check_package "optimum"
echo ""

# Quantization libraries
echo "Quantization-specific libraries:"
check_package "auto_gptq" || echo -e "${YELLOW}   Install: pip install auto-gptq${NC}"
check_package "awq" || echo -e "${YELLOW}   Install: pip install autoawq${NC}"

# Check HQQ (may be in transformers)
if python3 -c "from transformers import HQQConfig" 2>/dev/null; then
    echo -e "${GREEN}✅ HQQ available in transformers${NC}"
else
    echo -e "${YELLOW}⚠️  HQQ not available${NC}"
    echo -e "${YELLOW}   May need: pip install hqq${NC}"
fi

check_package "neural_compressor" || echo -e "${YELLOW}   Install: pip install neural-compressor${NC}"
echo ""

# Check 6: Disk space
echo "🔍 Checking disk space..."
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available space: ${AVAILABLE_GB}GB"
if [[ $AVAILABLE_GB -lt 20 ]]; then
    echo -e "${RED}❌ Less than 20GB available${NC}"
    echo -e "${YELLOW}   Recommendation: Free up space before proceeding${NC}"
    echo -e "${YELLOW}   Models + quantized variants need ~15-20GB${NC}"
elif [[ $AVAILABLE_GB -lt 50 ]]; then
    echo -e "${YELLOW}⚠️  Less than 50GB available${NC}"
    echo -e "${YELLOW}   Should be okay, but monitor disk usage${NC}"
else
    echo -e "${GREEN}✅ Sufficient disk space (${AVAILABLE_GB}GB)${NC}"
fi
echo ""

# Check 7: Memory
echo "🔍 Checking system memory..."
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
AVAIL_MEM=$(free -g | awk '/^Mem:/{print $7}')
echo "Total RAM: ${TOTAL_MEM}GB"
echo "Available RAM: ${AVAIL_MEM}GB"
if [[ $AVAIL_MEM -lt 8 ]]; then
    echo -e "${YELLOW}⚠️  Less than 8GB available${NC}"
    echo -e "${YELLOW}   Some quantization methods may run slowly or fail${NC}"
else
    echo -e "${GREEN}✅ Sufficient memory${NC}"
fi
echo ""

# Check 8: Project structure
echo "🔍 Checking project structure..."
REQUIRED_DIRS=(
    "experiments/advanced-quantization"
    "Models"
    "Testing/metrics/advanced-quantization"
    "utils"
)

ALL_DIRS_OK=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        echo -e "${GREEN}✅ $dir exists${NC}"
    else
        echo -e "${RED}❌ $dir missing${NC}"
        ALL_DIRS_OK=false
    fi
done

if [[ "$ALL_DIRS_OK" == false ]]; then
    echo -e "${YELLOW}   Run from project root: /mnt/c/Users/AlbertoTC/Documents/code/LLM-Training${NC}"
fi
echo ""

# Summary
echo "================================================"
echo "Environment Check Summary"
echo "================================================"
echo ""
echo -e "${GREEN}✅ Ready to proceed${NC}"
echo ""
echo "Next steps:"
echo "  1. cd experiments/advanced-quantization"
echo "  2. python common/baseline.py  # Establish baseline"
echo "  3. python quantize_gptq.py --bits 4 --output ../../Models/Llama-3.2-1B-Instruct/gptq-4bit"
echo ""
echo "Or run all experiments:"
echo "  bash run_all_quantization.sh"
echo ""
