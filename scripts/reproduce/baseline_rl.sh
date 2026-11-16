#!/usr/bin/env bash
# Bare-GPU LoRA GRPO, the declared RL task: 512-token prompt (task prompt + pad) + 512 generated, 16 prompts x 4 rollouts.
# Expect: about 1.2 samples/s with the default rank-16 adapter on all projections (1.24 measured on one H200 NVL).
cd "$(dirname "$0")/../.."; . scripts/reproduce/reference.env
mkdir -p $VA_OUT; env -u VARENA_TRITON_GEMM python -m baselines.rl.rl --model $VA_MODEL --seconds 150 \
  --spec "{\"n_prompts\":16,\"group\":4,\"gen\":512,\"prompt_pad\":492${RL_LORA-,\"lora_modules\":\"all\",\"rank\":16}}" --out $VA_OUT/rl_bare${RL_TAG:-}.json
python -c "import json;r=json.load(open('$VA_OUT/rl_bare${RL_TAG:-}.json'));print('samples/s', round(r['samples_s'],3))"
# Default: rank-16 LoRA on all seven projections of all 64 layers (docs/training_rules.md). Variants:
#   rank 1 (sensitivity):  RL_LORA=',"lora_modules":"all","rank":1' RL_TAG=_all1 bash scripts/reproduce/baseline_rl.sh
#   legacy q/v rank 16:    RL_LORA='' RL_TAG=_qv bash scripts/reproduce/baseline_rl.sh
