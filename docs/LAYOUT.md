# Repository layout

The top level follows the four steps in the README.

| README step | Folder |
|---|---|
| 1. Baseline workloads | `baselines/` (`inference/`, `pretrain/`, `rl/`) |
| 2. Verifier and passing inference | `schemes/<scheme>/` |
| 3. Passing training (the red team's attacks) | `schemes/<scheme>/attacks/<attack>/` |
| 4. Measurement | `scripts/reproduce/`, with the measured numbers in `results/README.md` |

```
README.md                 human-written overview
docs/
  RESULTS.md              the measured numbers; the first section is the final configuration
  REPRODUCE.md            how to rerun every result on one H200
  ITERATE.md              how to change the scheme or attack it, and the rules for counting a result
  LAYOUT.md               this file
  training_rules.md       what the hidden training is allowed to do
baselines/                stock inference benchmark, bare pretraining, bare LoRA RL
schemes/memory_link_exhaustion/
  inference/              vLLM plugin (entry point `varena`), kernel patches, honest prover
  verifier/               verifier, reference engine, page checks, pad sizing, floor scoring
  encodings/              butterfly keystream (weights, KV, filler), the earlier sloth encodings, CPU references
  link/                   PCIe noise streamer
  gpu/                    Triton kernels, native single-queue code, CUDA sources
  core/                   arena page layout, model, wire protocol (shared by both sides)
  attacks/                zero_weight_cheater (the cheater prover) and the hidden workloads it runs
  tools/                  h2d_route.c (routes small host-to-device copies into the noise queue), its CPU mock
  tests/                  CPU and GPU tests
  scripts/                timing_runs.sh (one verifier session), merge_calib.py
  docs/DESIGN.md          the scheme as built: arena, encoding, challenge, link noise, floors
scripts/
  bundle.sh               pack tracked files into a tarball for a rented box
  reproduce/              setup, host check, baselines, honest runs, floors, cheaters, verdict, CPU tests
results/README.md         the run record: every run behind a reported number, with its box
```

## Attacks

| Folder | What it is |
|---|---|
| `zero_weight_cheater/` | the cheater prover: answers challenges, paces fake serving, runs one hidden workload |
| `max_shard/` | hidden pretraining of one stage of a sharded model |
| `local_rl/` | hidden LoRA RL on one GPU |
| `gated_shard/` | shared pieces: the copy pacer and the challenge gate |
| `hidden_lora_rl/` | shared piece: the LoRA model builder (BflyLin) local_rl uses |
| `ring_reuse/` | regression cheater behind the noise-permutation test |

## What is not tracked

Raw run output, logs and arrays. `results/` is in `.gitignore` apart from its README. Runs write to
`results/reproduce/` on the box.

Names that stay fixed: the environment flags (`VARENA=1`, `VARENA_*`), the vLLM plugin name `varena`, and the shared
memory files `/dev/shm/varena_*`.
