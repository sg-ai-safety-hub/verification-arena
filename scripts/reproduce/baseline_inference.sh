#!/usr/bin/env bash
# Stock vLLM deterministic mode (no plugin), 300 prompts of the VA_DATASET workload (ShareGPT by default); saves the tokens every honest run is compared against.
# Expect (H200 NVL): about 1,136 tok/s on ShareGPT (VA_DATASET=sharegpt, the default); 1,435-1,463 with VA_DATASET=random.
cd "$(dirname "$0")/../.."; . scripts/reproduce/reference.env
O=$VA_OUT/base_stockdet; mkdir -p $O
env -u VARENA_TRITON_GEMM -u VARENA_KV_HEADS -u VARENA_W_BFLY_L VLLM_PLUGINS=none MODEL=$VA_MODEL NB=$VA_NB NP=300 TOKCAP=$VA_TOKCAP DS="--dataset-path $VA_DATA --dataset $VA_DATASET" OUT=$O \
  bash schemes/memory_link_exhaustion/scripts/timing_runs.sh base
tail -1 $O/baseline.jsonl
