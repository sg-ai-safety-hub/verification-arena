#!/usr/bin/env bash
# Fresh box setup (pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel image, run from the repo root). ~10 min + model download.
set -euo pipefail
cd "$(dirname "$0")/../.."
pip install -q uv && uv pip install --system -q vllm==0.32.0 numba cryptography
uv pip install --system -q "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-cuda-runtime==13.0.*" "nvidia-cuda-cccl==13.0.*" ninja
P=$(python -c "import nvidia, os; print(os.path.join(os.path.dirname(nvidia.__path__[0]), 'nvidia', 'cu13'))")
ln -sf libcudart.so.13 $P/lib/libcudart.so; [ -e $P/lib64 ] || ln -s $P/lib $P/lib64
[ -e /usr/lib/x86_64-linux-gnu/libcuda.so ] || ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
echo "export CUDA_HOME=$P PATH=$P/bin:\$PATH VLLM_USE_FLASHINFER_SAMPLER=0" > /root/env.sh
pip install -q --no-deps -e .
. scripts/reproduce/reference.env
python -c "from schemes.memory_link_exhaustion.gpu.cuda import build; build.load(); print('EXT_OK')"
python -c "from schemes.memory_link_exhaustion.gpu import native; native.mod(); print('NATIVE_OK')"
mkdir -p "$(dirname "$VA_DATA")" "$(dirname "$VA_MODEL")"
python - <<PY
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download(repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered", filename="ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset", local_dir="$(dirname "$VA_DATA")")
snapshot_download("Qwen/Qwen2.5-32B-Instruct", local_dir="$VA_MODEL", allow_patterns=["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.json"], max_workers=8)
print("DATA_MODEL_OK")
PY
echo SETUP_DONE
