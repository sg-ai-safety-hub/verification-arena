# Iterating on the arena

How to change the scheme or attack it, and how to tell whether a change helped. Read `docs/REPRODUCE.md` first for
the commands; this file is about the loop around them.

## The loop

1. **Change code, run the CPU tests.** `bash scripts/reproduce/cpu_tests.sh` takes one to two minutes and needs
   no GPU. Add a CPU test for anything you can check without a GPU: encodings against their reference, scoring
   rules, a cheater's bookkeeping, bitwise equality of a fast path with the declared path.
2. **Rent one H200 and check the host.** `setup.sh`, then `check_host.sh`. Do not use a host that prints
   `NOISE_PATH WARN`.
3. **Baselines on that box.** Stock inference, bare pretraining, bare RL. Every slowdown is a ratio against the
   same box.
4. **Honest runs and floors on that box.** `floors.sh h1 h2 h3`. Any change that touches the honest path needs a
   fresh honest run before it becomes the default: outputs 300/300 identical to stock, tok/s inside the band of
   earlier honest runs, no failed checks except the known artefacts.
5. **Idle control, then cheaters.** `cheater.sh idle_control_native`, then the cheaters you care about.
6. **Record the result with its source.** Add the number to `results/README.md` with the run tag, the
   configuration and the box. Do not commit raw run output.
7. **Destroy the box** when its queue ends.

## Rules that keep results honest

- **Validity before any verdict.** State whether the run was valid and whether the cheater was caught before
  quoting its speed.
- **Same-box comparisons only.** If a baseline or a floor comes from another box, label it cross-box.
- **A cheater is "caught" only if the idle control passes the same rule** on the same box.
- **Whole steps.** Quote a training rate from complete optimizer steps where you can. If a run ends before one
  completes, say so and quote the per-sample or per-token rate with that caveat.
- **A cheater limited by its own bug is an upper bound on the slowdown,** not a result for the defence. Label it
  "as coded" and say what limits it.
- **"Does not fit" needs the smallest sensible setting tried.** Otherwise write "does not fit as coded".
- **Models are labelled as models.** Anything not measured under the verifier is an estimate and is marked.
- **Declared precision.** The hidden training must be the declared workload at the declared precision. Lossy
  compression of activations, gradients or weights is outside the rules (`docs/training_rules.md`). Bit-exactness is
  required only of inference.

## Saving GPU time

A verified run costs 8 to 10 minutes even when the measurement itself is 100 seconds, because of model load,
sealing and post-checks. Plan around that.

- **Rehearse memory before a full run.** Set `VARENA_REHEARSAL_STEPS=N` in a cheater's `env` to do a real, shortened
  seal and N workload steps. It prints `REHEARSAL PASS` or `FAIL` with the minimum spare memory in a few minutes,
  which is how to find out that a workload does not fit without paying for a whole session.
- **Time a steady rate from a slice.** If a cheater's per-step time is flat, a few minutes of steps plus one
  update gives the same rate as a whole pass. Run a short separate session for the verdict. Label the rate as
  extrapolated from a steady rate.
- **Use two boxes for independent work only.** A second box cannot shorten cheater verdicts, because it needs its
  own baselines and floors first (about 50 minutes). It does help for honest-path debugging and for testing the
  reproduce scripts from a fresh clone.
- **Not built yet:** keeping one sealed server up across several verifier sessions, and running post-checks while
  the next run starts. Together they would roughly halve the cost of a set of honest runs.

## Adding or changing an attack

Attacks live in `schemes/memory_link_exhaustion/attacks/<name>/`. All of the final ones run inside the zero-weight
cheater prover (`attacks/zero_weight_cheater/cheater.py`), which answers the verifier's challenges, paces fake
serving, and runs a hidden workload on its own thread in whatever memory and link it can free.

A workload is a class with:

- `__init__(spec, budget, device)`: build tensors within `budget` bytes;
- `step()`: do one unit of hidden work and return how many units it did;
- `describe()`: return a dict of settings and counters, which ends up in the run's JSON and in the verdict;
- `close()`.

To add one:

1. Create `attacks/<name>/workload.py` and a `test_cpu.py` beside it.
2. Register the kind in `Cheater._workload` in `cheater.py`.
3. Add an entry to `scripts/reproduce/cheaters.json` with `mode`, `prompts`, `env`, `spec` and an `expect` line
   that will later quote the measured result with its run and box.
4. Add its suite to `scripts/reproduce/cpu_tests.sh`.
5. Rehearse on a box, then run it with `cheater.sh <name>`.

Things the existing attacks already provide and you can reuse:

- `attacks/gated_shard/`: the `Pacer` that paces host-to-device copies at a declared stolen rate and splits them
  into slices small enough to go through the routed queue, and the gate that pauses hidden work during a challenge.
- `attacks/max_shard/`: the gradient flusher that ships gradients off the GPU, and the micro-batch phase timer.
- `attacks/local_rl/`: sub-group generation, reentrant checkpoints, and the slice MLP that decodes each weight
  once per pass.

## Changing the scheme

The scheme is `schemes/memory_link_exhaustion/`:

| part | where |
|---|---|
| vLLM plugin, kernel patches, honest prover | `inference/` |
| verifier, reference engine, page checks, floor scoring | `verifier/` |
| butterfly keystream (and the earlier sloth64) with CPU references | `encodings/` |
| link noise streamer | `link/` |
| Triton kernels, native queue, CUDA sources | `gpu/` |
| page layout, model, wire protocol shared by both sides | `core/` |

Settings are environment variables read by the prover (`VARENA_*`) and flags on the verifier. The final values are
in `scripts/reproduce/reference.env` and `lib.sh`. Change a default there only after an honest run on a box
confirms it.

What the scheme does and why is in `schemes/memory_link_exhaustion/docs/DESIGN.md`.

### Code that is not part of the final configuration

The final configuration sets about 30 of the more than 100 `VARENA_*` names the code reads. The rest are debug knobs
(launch, step, allocator and stall logs, memory snapshots, profilers; all off by default) or belong to earlier
designs whose code is still interleaved with live code. These are the known leftovers; removing each needs an
honest run and the GPU tests afterwards:

| leftover | why it is still there |
|---|---|
| sloth64 encodings (`encodings/sloth64.py`, `gpu/cuda/slothgemm_torch.cu`) | the previous encoding, still selectable (`VARENA_WENC`, `VARENA_KVENC`, `VARENA_FILLER`) and used by the regeneration benchmark for comparison |
| fresh AES noise (`--noise-fresh`) and the schedule without a permutation table (`--noise-perm 0`) | interleaved with the live reveal loop in `link/pcie.py`; the echo, the window digests and the unscheduled Python stream were removed on 14 Nov |
| the cheater prover's activation pool, `VARENA_AE_S32` / `encodings/sloth32.py` and the pool code in `encodings/actenc2.py` | the zero-weight cheater used to declare a 1.81 GiB pool; the verifier's pool rule (`verifier/act_pool.py`) and its CPU test remain. Honest runs have no pool and no longer set the pool settings. `ae_add_norm` and `ae_silu` are unused wrappers |
| `VARENA_AE_PARK`, `VARENA_WDEC_FUSE` | inside kernel templates |
| `VARENA_ATT_GATE`, `VARENA_WDEC_GATE`, `VARENA_FUSE_YIELD` | opt-in stronger pause and fused yield (16 Nov), off by default |
| `--kv-table 1` / `2` and `--kv-tab-test` | kernel-validation modes |
| spot checks and page claims on the socket (`spot`, `claims`) | used by `--selftest` only |
| the prover's own block-table log | unread for honest runs (the verifier derives the table), but the verifier reads the declared table of a cheater that serves nothing |

Removed in the second cleanup batch: the non-event queue modes (`VARENA_SQ_MODE=streamer` / `legacy`), the honest
run's inert pool settings, secret noise-rate spikes (`--noise-spike-*`), and the flags calibrated runs never consumed
(`--allowance-margin-gib` and `--pad-mib` for cheater runs, `--ring-rate` under `--unified`).

A different verification scheme would be a sibling folder under `schemes/` with its own verifier, passing
inference implementation and attacks, measured against the same `baselines/`.

## Where the honest overhead goes

Honest inference runs at 1.89 times the stock time on ShareGPT (597 against 1,127 tok/s, Box W, 16 Nov). The steps
that got there, one run each on Box W, are in the first section of `docs/RESULTS.md`: the butterfly decode itself
is cheap (about 2 TB/s in bulk), and most of the cost is the decode-ahead's extra HBM traffic and the challenge
pauses. Fused `gate_up` decode and smoothed decode-ahead cost about 3% of throughput and raise the honest noise rate
by 0.38 GB/s; exit-and-resume costs about 1.5% more. The sloth64-era breakdown (modular squarings per byte, the
14 Nov ablations) is not included in this release.

Ideas not yet tried: fewer chunks per layer in the decode-ahead (the engine's captured graphs use the maximum
context of 64 blocks per sequence, so a 270 MiB scratch takes 67 sequences at a time; a batch of 160 goes in three or
four chunks, each a second attention launch); a hand-written attention kernel whose summation order does not depend
on how the decoded tile is laid out.

## Open problems

- **Noise dips.** Honest runs show zero to five pauses of 0.3 to 0.6 s in the link noise; each 0.5 s pause costs
  about 0.25 GB/s of a 100 s run's mean, and the number of pauses explains most of the spread between honest runs on
  one host. Correctable PCIe errors explain some (each froze the whole host for about half a second; the verifier now
  credits its own freezes). CPU load on the host lowers the noise (a CPU test suite during an honest run cost about
  1.9 GB/s), so run nothing else on a box during a verified run. A larger in-flight window raises the noise and
  lowers serving but does not remove the dips. The 13 to 14 Nov diagnostics (what was ruled out, Box K's drift) are
  not included in this release.
- **The floor margin is thin and the floor ages.** The end-of-session floor is the lowest of three honest runs
  minus 0.12 GB/s. Against a same-hour floor a theft of 0.3 GB/s sits on the floor (caught once, passed once) and
  0.15 GB/s was caught once by 0.02 GB/s; a floor taken four hours earlier let every cheater through (14 Nov,
  sloth64). On 16 Nov a noisy host's honest runs spanned 0.74 GB/s, so a fourth floor run lowered its floor by
  0.44 GB/s (`docs/RESULTS.md`, Box X).
- **A prover may stop serving.** No rule depends on how fast the prover computes. A prover that freezes its engine
  can keep up to the in-flight ceiling (512 MiB at settle lag 0) unverified by content for as long as it serves
  nothing.
- **Outbound link is not exhausted.** Only inbound traffic is saturated with noise. Gradients leave the GPU
  unpoliced.
- **Where the policy comes from.** The scheme slows on-policy RL because rollouts must be generated on the GPU.
  Off-policy RL or supervised LoRA fine-tuning on data generated elsewhere avoids most of that cost. This is
  analysis only and has not been measured.
- **Zero-knowledge proofs** are out of scope.

## Tried and dropped

Kept here so nobody repeats them without a new reason.

- sloth64 encoding for weights, KV and filler: replaced by the butterfly keystream on 16 Nov (cheap per dropped word,
  61 GB/s in bulk; `docs/RESULTS.md`, first section). Its code is still in `encodings/`.
- Stronger pause, fused yield, a yielding weight decode, 8-way sliced decode and fusing every matrix (16 Nov): at most
  0.18 ms off the honest answer for 8% of throughput, or worse on both; the first two remain as opt-in switches.
- Memory-borrow cheaters (restore borrowed filler before each answer) and the regenerate-on-probe cheater: measured
  on 16 Nov, then removed from the catalogue; the per-probe cost is kept as `tests/bench_regen_probe_gpu.py`.
- Encoded activations: unencoded activations are the reference; the unchecked memory is cut another way.
- Under sloth64, overlapping the weight decode with the matmul and a faster standalone decode kernel: no gain. (Under
  butterfly, fusing the decode into the `gate_up` matmul is part of the final configuration.)
- Give-back filler for torch's idle reserve, scrub on halt, per-host noise calibration: designed, not built. The
  arena is instead sized from the measured peak, and the floors are per host.
- Refilling freed KV blocks with filler: replaced by the stale-KV route, where the verifier recomputes stale pages.
- Syncing the GPU on non-decode steps to stop the noise dips: the dips remained and throughput fell 3%.
- Weight streaming as a cheater: does not fit as coded down to a 128-token context.
- 8-bit storage for the RL cheater's rollout KV: built, outside the declared-precision rule, never run.
- A separate ring challenge and KV chain challenge: folded into the unified challenge.
