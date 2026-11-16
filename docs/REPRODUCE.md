# Reproducing the memory + link exhaustion results

Everything runs from `scripts/reproduce/` on one rented H200. The final configuration lives in two files:
`reference.env` (engine, prover and verifier settings) and `lib.sh` (what every verified run adds on top: the routed
host-to-device queue for the saturating noise, the memory cut, the per-host floors). The cheaters are named entries in `cheaters.json`.

No raw run output is tracked in this repository. Runs write to `results/reproduce/` on the box. The numbers we
measured, with the run and box behind each one, are in `results/README.md`.

**What you are reproducing (final design, 16 Nov).** One verified challenge: a chain that hops through the encoded
weights, the KV cache, filler and every slot of the noise ring. Reads that land on memory in flight — KV being
written, ring slots being overwritten — return a short tag, and the verifier's replay takes the old or the new
content by it, so no sealed memory and no ring memory is left unread. The table of which request wrote which KV
position is rebuilt by the verifier from the engine scheduler's call order; sampled chains (4 per run by default) are replayed
from sources. The link is judged by two rules — the end-of-session rate and the head the prover returns with each
answer — against a floor from the host's own honest runs (lowest of three minus 0.12 GB/s). Honest inference decodes
butterfly-encoded weights and KV in whole blocks on the way into the stock kernels, and yields to a challenge at
tile boundaries. Expect about 1.9x on ShareGPT (1.89x on Box W, 16 Nov). `docs/RESULTS.md` has the
measured numbers.

## What you need

- **GPU:** one H200 (NVL or SXM, 141 GB) on PCIe Gen5 x16 at every hop. `check_host.sh` must show a raw
  host-to-device ceiling of about 55 GB/s or more with 32 MiB copies, and must print `NOISE_PATH OK`.
  Hosts that reach only 32 to 43 GB/s exist and cannot run the reference noise.
- **Host:** at least 256 GB RAM (the noise pool is 64 GiB in `/dev/shm`) and at least 200 GB disk.
- **Image:** `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`. `setup.sh` installs vLLM 0.32.0 and the CUDA 13.0
  compiler wheels our extensions need.
- **Model and data:** Qwen/Qwen2.5-32B-Instruct (about 64 GB) and ShareGPT V3. `setup.sh` downloads both.

Paths default to `/root/models/...`, `/root/data/...` and `results/reproduce`. Override `VA_MODEL`, `VA_DATA` and
`VA_OUT` in the environment before running any script.

The workload is ShareGPT (`VA_DATASET=sharegpt`, the default), which is what every published floor and verdict uses.
`VA_DATASET=random` runs random tokens instead; give it its own `VA_OUT`, because floors and
baselines from one workload do not apply to the other.

## Without a GPU

```bash
bash scripts/reproduce/cpu_tests.sh
```

This runs every CPU test in about one to two minutes: the attack suites (`max_shard`, `local_rl`, `gated_shard`,
`ring_reuse`) and the scheme's CPU checks (noise schedule, floor scoring, the reference flags, the block table and its
session checks, the batched chain replay with tagged in-flight reads). It needs Python 3.10+, PyTorch,
NumPy, Numba and pytest. Each attack suite prints one line `attack <name>: <name> passed N failed 0`, each scheme test
one line `test <name>: PASS` or `FAIL`, and the script exits non-zero if anything failed.

## On a box

Get the code onto the box either with `git clone` on the box, or from a tarball of tracked files:

```bash
bash scripts/bundle.sh /tmp/varena-bundle.tgz                         # on your machine, from the repository root
scp -P <port> /tmp/varena-bundle.tgz root@<box>:/tmp/                 # copy it to the box
mkdir -p /root/varena && tar xzf /tmp/varena-bundle.tgz -C /root/varena   # on the box
cd /root/varena                                                       # the repository root for every step below
```

Then run from the repository root, in this order. Times were measured on one H200 NVL on 2026-11-13 with the
wheels and the model already on disk.

| step | command | time | what to expect |
|---|---|---|---|
| setup | `bash scripts/reproduce/setup.sh` | ~5 min with the model download on a fast link, ~1 min when re-run | prints `EXT_OK`, `NATIVE_OK`, `DATA_MODEL_OK`, `SETUP_DONE` |
| host check | `bash scripts/reproduce/check_host.sh` | ~6 min (Box S, 15 Nov; 18 min there with the earlier two long noise sessions) | Gen5 x16 on every hop; raw H2D at 32 MiB about 56 to 57 GB/s; `NOISE_PATH OK` (see "If the host check warns") |
| inference baseline | `bash scripts/reproduce/baseline_inference.sh` | ~3 min | about 1,136 tok/s on ShareGPT (stock vLLM, deterministic mode); 1,435 to 1,463 with `VA_DATASET=random` |
| pretraining baseline | `bash scripts/reproduce/baseline_pretrain.sh` | ~6 min | best about 5.7e13 to 5.8e13 param-tok/s (the largest `param_tok_s` among the printed rows; usually `stack32_L8_mb2`) |
| RL baseline | `bash scripts/reproduce/baseline_rl.sh` | ~4 min | about 1.2 samples/s (rank 16 LoRA on all projections) |
| honest runs and floors | `bash scripts/reproduce/floors.sh h1 h2 h3` | ~27 to 33 min (pad probe 8 to 9 min, then 6.5 to 8 min per run; about 1 min 20 s less per run since the 8-request pre-run was removed on 16 Nov: 6 min 15 s on Box W) | each run about 670 to 680 tok/s on ShareGPT (15 Nov, one host), 300/300 outputs identical to stock, one chain per challenge replayed exactly, `kv_table_derived` passing; writes `floors.env` with the end-of-session floor `ENDF` and the timing calibration |
| GPU tests | `bash scripts/reproduce/gpu_tests.sh` | ~4 min | one `PASS` line per test (kernels, encodings, GEMM paths, matmul gate, chase and its tagged reads, noise stream). `test_h2d_route_gpu` is a known failure and is not in the default list (see "Known open items") |
| idle control | `bash scripts/reproduce/cheater.sh idle_control_native` | ~4.5 min | `NOT CAUGHT`; end-of-session rate above the floor |
| one cheater | `bash scripts/reproduce/cheater.sh <name>` | about 0.37 s per prompt plus ~2 min of start-up and checks: paced pretraining 900 prompts, paced RL 500 prompts (about 5 min, one call of 2 samples) | see the `expect` field of that entry in `cheaters.json` |
| everything | `bash scripts/reproduce/run_all.sh` | about 1 h after a host check, plus ~15 min with it (summed from the rows above, not re-timed since the catalogue was cut to three) | runs the rows above in one go instead of one by one: the host check (skipped if it already printed `NOISE_PATH OK` on this box; `FORCE_CHECK=1` repeats it), baselines, honest runs and floors, then the control and the two cheaters (GPU tests are separate) |

`honest.sh <tag>` runs a single honest run. The first honest run on a box is preceded by a pad probe (tag `pad`),
which measures how much memory serving really uses so the arena can be sized to leave as little as possible unchecked.

Every run appends one `WINX` summary line to `results/reproduce/arena.log` and then prints its verdict.

## Reading a verdict

`verdict.py` prints validity first, then the rate.

- **Honest run:** `HONEST PASS` with tok/s, how many outputs are identical to stock, and the noise rate.
- **Cheater run:** `CAUGHT by <checks>` or `NOT CAUGHT`, then the workload rate and the slowdown against this box's
  own bare baselines.

Three rules decide whether a result counts:

1. **Floors are per host and per hour.** A cheater is judged against the end-of-session floor computed from honest
   runs on the same box (`floors.sh`: the lowest of them minus 0.12 GB/s, `VA_END_MARGIN`). Hosts differ by about
   1 GB/s in achievable noise, and one host drifted by 0.3 GB/s within a day, both more than the margins being
   judged. Recalibrate shortly before the runs you want to judge.
2. **Idle control first.** A link catch counts only if `idle_control_native` passes under the same floors on the
   same box.
3. **`verifier_healthy` failing means the verifier's own host stalled; rerun.** The link rules are counted only
   when floors are set (`link_rules_counted` in the report); honest runs started with `honest.sh` report them
   without counting them.

A `WORKLOAD ERROR` line under a cheater verdict means the hidden job itself stopped (so far always out of memory).
The validity line still stands, but the printed rate and slowdown then average over the time after the job died, so
take the rate from the completed optimiser steps or calls listed on the `RATE` line, or rerun.

## If the host check warns

`NOISE_PATH WARN` means the verifier's noise could not get within 2 GB/s of the raw link. `LINK_ERRORS WARN` counts
new correctable PCIe errors on the hops to the GPU during a run; it is informational, but a host that prints it on
every run is one whose noise rate drifts.

One H200 NVL host printed both on 2026-11-13 (raw 57.2 GB/s, noise 54.3 and 54.9). What still held there, and what
did not:

- Honest runs passed (six of six: 300/300 outputs identical to stock, 743 to 753 tok/s, no wrong ring unit), the GPU
  tests gave the same results, and the idle control was not caught under floors from the first three honest runs.
- Cheater verdicts that depend on the noise floors were not reliable. Honest noise means moved from 53.7 to 55.1 GB/s
  to 55.8 to 56.1 GB/s within two hours on the same box and code. Under floors from the first three honest runs
  neither of two fixed-rate theft cheaters (0.45 and 0.61 GB/s) was caught. Under floors from three later honest runs both were, but
  `idle_control` itself then failed ring content in three of three runs (one with `verifier_healthy` also failing), so by rule 2 above those catches do not
  count.

So on a host that warns, use the honest path and the GPU tests, and do not report cheater verdicts
that turn on the floors.

## When the link clock starts

For honest runs the noise (and the link clock) starts after the untimed warm-up; for cheaters and the idle control
it starts when the prover reports ready. On 12 Nov, on one host and before the routed single queue, a late start froze
honest serving; on 16 Nov it was clean in seven honest runs on Box W. An honest run that shows `STALLS` above 0 in
`arena.log`, or `GUARD TRIPPED`, is worth reporting with the host it ran on.

## Known open items

- `test_h2d_route_gpu` fails and is left out of the default `gpu_tests.sh` list (`GPU_TESTS_KNOWN_FAIL=1` runs it, up
  to 6 min more): with the interposer preloaded, a blocking pageable copy on the default
  stream does not land within 15 s under the self-test's saturating noise, or the 3,000-copy loop outruns the 360 s
  phase timeout. It fails the same way at the commit before the queue-mode removal. Verified runs are not affected in
  any way we measured (honest runs reach the expected tok/s with identical outputs).
- The 0.12 GB/s floor margin is about one honest run-to-run spread, and the idle rate drifted by 0.21 GB/s in an hour
  on our host. Take the floors in the same hour as the cheater runs.
- A graph-captured RL cheater does not fit the sealed memory (capture needs the adapters resident; measured on
  14 Nov; that round is not included in this release).

## Converting a rate into a slowdown

| workload | slowdown |
|---|---|
| inference | stock tok/s ÷ honest tok/s |
| pretraining | bare param-tok/s ÷ (stage parameters × cheater tok/s) |
| RL | bare samples/s ÷ cheater samples/s |

Baselines must come from the same box as the run. If you compare against another box's baseline, say so.

## The final cheaters

| name | what it is |
|---|---|
| `idle_control_native` | not an attack: the cheater prover doing no hidden work. Run it first; a catch on this host counts only if the control passes |
| `max_shard_fitnh_loop_A_l05` | hidden pretraining of the largest stage that fits the memory left free after sealing (link allowance 0.5 GB/s, 2 host threads, 450 prompts), paced closed-loop: it trains while its own noise average is above the floor and rests when it drops |
| `local_rl_a16_g2_smooth_loop_nh` | hidden RL on real weights (rank 16 LoRA, two rollouts in flight), paced the same way |

These three are the whole catalogue. The fixed-rate theft runs, the activation-pool negative control, the
launch/graph storm diagnostics, the memory-borrow and ceiling variants and the regenerate-on-probe cheater that
produced the earlier (pre-16 Nov) numbers are no longer in the code and are not included in this release. The cost of regenerating dropped pages per probe is kept as a standalone benchmark,
`tests/bench_regen_probe_gpu.py`.

The paced cheaters aim at `VARENA_PACE_TARGET_GBS`, which `lib.sh` sets to the floor plus 0.03 GB/s unless you export it.
None of these cheaters sends the verifier a block table: a prover that serves nothing has exactly one legitimate table
(every block filler, nothing in flight), and the verifier builds it itself.

## Notes that save time

- **Pin to the GPU's NUMA cores.** `lib.sh` does this for you. It raises the noise rate by about 1.3 GB/s and
  cuts run-to-run drift.
- **One run at a time.** Each verified run seals about 138 GiB of GPU memory. Kill leftover GPU processes and
  remove `/dev/shm/varena_*` before starting another (the scripts do).
- **Where the time goes.** An honest run serves for about 93 s. Start-up is about 2.5 min (engine start, model load,
  encoding, sealing, warm-up). After serving, about 3 min go to the reference job, which starts the engine a second
  time (about 60 s) and prefills all 300 requests into a KV store (about 50 s) so the chains can be replayed; the
  replay itself is short. Honest and floor runs replay a sample of 4 chains by default (`honest.sh <tag> 300 full`
  replays one chain of every answer), which saves only about 20 s. Shortening honest runs further means removing the second engine start, which
  is not done. The first run in a new output folder also stalls for about a minute at its first CUDA-graph capture
  (a compile whose cache is keyed on the memory configuration, which differs between the pad probe and later runs).
  The pad probe is reused by any output folder with the same workload on the box (`FORCE_PAD=1` measures again).
- **The RL session is one or two calls.** `local_rl_a16_g2_smooth_loop_nh` runs 500 prompts, which fits one or two calls. A *sample*
  is one generated rollout (512 tokens continuing a 512-token task prompt) scored and backpropagated; a *call* is one
  task prompt with its group of 2 rollouts, generated together and then used for one gradient pass, so 2 samples; an
  optimiser step is 16 calls. The slowdown is bare samples/s over the cheater's samples/s in that call; with 2
  samples the figure is coarse. For measurement the
  weight update runs after every call (`opt_every_calls: 1`) and is timed apart from the call. Its cost does not
  depend on how many calls fed the gradient, so `verdict.py` also prints a modelled whole step: 16 calls plus one
  update. On Box W the update took 2.7 to 4.1 s, 0.10 to 0.15% of a step. (The
  session's "prompts" are something else: the verifier's inference requests, which set how long the session lasts.)
  Pass a larger prompt count as the second argument to `cheater.sh` for more calls.
- **Do not touch the box during a verified run.** `nvidia-smi`, `lspci` or a CPU job in the middle of a session shows up as a stall. Read logs only.
- **Shut the box down when the queue ends.** Tie any automatic teardown to the queue's end marker, not to a clock.
