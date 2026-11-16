#!/usr/bin/env bash
# Bare-GPU pretraining: shard slices and stacks of full Qwen2.5-32B layers; the best stack is the baseline.
# Expect: best ~5.7e13 param-tok/s (8 layers, micro-batch 2) = ~3.4e14 FLOP/s (6 x params x tokens, before attention).
# Also the deterministic (blocked-summation) cost: deterministic_cost.py default|det.
cd "$(dirname "$0")/../.."; . scripts/reproduce/reference.env
mkdir -p $VA_OUT; env -u VARENA_TRITON_GEMM python baselines/pretrain/bare_sweep.py $VA_OUT/pretrain_bare.json | tee $VA_OUT/pretrain_bare.log
