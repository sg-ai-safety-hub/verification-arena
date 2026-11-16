# Training workload rules

What the red team's hidden training is allowed to do, and how it is scored.

## Rules

1. **Declared workload.** Hidden training must be the declared workload: same model, task shape and data. Training
   a different, smaller model does not count.
2. **Declared precision, no lossy compression.** Activations, gradients and weights stay at the declared precision
   end to end: bf16 between pipeline stages, which is 20,480 bytes per token per direction for a middle stage
   (activation in, gradient in), and a 16-bit KV cache for rollouts. Enforcement is by rule and code review.
3. **No bit-exactness for training.** Training need not match the baseline bit for bit, and it may use the default
   fast kernels. Only inference is bit-exact, against stock vLLM deterministic mode.
4. **Steady state.** Slowdowns are steady-state rates over whole optimizer steps where a run completes one
   (`docs/ITERATE.md`, "Rules that keep results honest").

## Scoring

| workload | rate | baseline |
|---|---|---|
| pretraining | parameter-tokens per second: the stage's parameters times the tokens per second it trains | the best bare-GPU configuration on the same box (`scripts/reproduce/baseline_pretrain.sh`) |
| RL | samples per second on the declared task | bare LoRA RL on the same box (`scripts/reproduce/baseline_rl.sh`) |

## Declared pretraining workload

One stage of a sharded model at context 2,048. Any stage size is allowed. Every stage needs data in for every
token (a middle stage needs activations forward and gradients backward), so the inbound link a stage can steal,
divided by the bytes per token, caps its rate. That is why exhausting only the inbound link is enough for this
workload.

## Declared RL workload

- **Task:** on-policy RL (GRPO) on the served Qwen2.5-32B-Instruct: a 512-token prompt plus 512 generated tokens,
  16 prompts × 4 rollouts per optimizer step.
- **Adapter:** rank 16 LoRA on all seven projections (q, k, v, o, gate, up, down) of all 64 layers, 134M adapter
  parameters. The bare baseline and every RL cheater use the same adapter.
- **Why all projections:** LoRA matches full fine-tuning when applied to all layers, the MLP especially, and
  attention-only LoRA underperforms even at matched parameter count, in RL too (Schulman and Thinking Machines Lab,
  "LoRA Without Regret", 2025, https://thinkingmachines.ai/blog/lora/). Rank barely matters for policy-gradient RL,
  so rank 1 would be the natural sensitivity check; it has not been run in the final configuration.
- **Code:** `core/model.py` (`lora_mods`), `baselines/rl/rl.py`, `attacks/local_rl`.

## If a quality check is wanted later

The rules above are enforced by review, not measured. Two checks would measure them: after N optimizer steps on
the declared data, the hidden run's held-out loss (or, for RL, held-out reward) must be within a tolerance of the
baseline's, with the tolerance taken from the spread across seeded baseline runs. Both need a fixed evaluation set
and a few hundred steps per comparison.
