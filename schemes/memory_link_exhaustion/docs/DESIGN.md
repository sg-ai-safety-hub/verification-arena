# Memory + link exhaustion: the scheme as built

This describes the final configuration only (`scripts/reproduce/reference.env` plus `scripts/reproduce/lib.sh`).
Measured numbers are in `docs/RESULTS.md`; what was tried and dropped is listed in `docs/ITERATE.md`.

## Idea

A GPU that serves declared inference should have nothing left for hidden training. The scheme takes away two
things:

1. **Memory.** Nearly all GPU memory is one arena whose contents the verifier can predict and which is slow to
   recreate at random. About 1.95 GiB stays unchecked, and a cheater that serves nothing finds about 1.1 GiB of
   it free.
2. **Inbound link.** The verifier streams random noise over PCIe as fast as the link carries it. Hidden work that
   needs data from the host has to steal link time, and the noise rate the verifier measures falls.

Hidden training then runs in the leftover memory on whatever link it can take below the noise floor.

## Threat model

- **Zero weights.** The worst case is a prover that declares an all-zero model and serves no inference
  (`--zero-weights`). Then every weight, KV and filler page is the encoding of zeros, so the encoding itself has to
  be costly to regenerate. Runs with real weights are also measured (hidden RL needs them).
- **The verifier is conceptually off-host.** It owns the keys and the noise. On a rented box it runs as a separate
  process on the same machine; nothing in the prover reads its secrets.
- **The prover may compute faster than the calibration GPU.** So no check relies on how long honest compute takes.
  The checks are content, link rate and challenge latency.
- **Declared precision.** Hidden training must be the declared workload at the declared precision
  (`docs/training_rules.md`).
- **Which RL counts.** RL with adapters on every layer. An adapter on the last layer or the output head alone
  leaves the KV cache identical and is out of scope.

## Arena

64 KiB pages in one index space, sealed once after the model loads:

| region | H200 size | holds |
|---|---|---|
| W | 61.03 GiB | every large weight matrix, encoded in place (the 1.45 GiB embedding table is in W but stays plain) |
| K | 40 GiB | vLLM's KV blocks, one page per layer per block, encoded on write |
| F | 34.83 GiB | filler: the encoding of empty memory |
| ring | 2 GiB | the most recent link noise, window i in slot i mod ring size |

That is 137.86 of 139.81 GiB (98.6%, Box W, 16 Nov). The unchecked rest (1.92 to 1.95 GiB across boxes) is the CUDA context
and graphs, torch's reserve, activations, the weight decode scratch (270 MiB) and sampler scratch. These buffers
are rewritten every step, so they cannot hold
predictable content; the only lever is to make them small.

**Sizing from the measured peak.** The first honest run on a box is a pad probe: it reports how much free memory
serving really used, and the arena for later runs leaves that much times 1.25 plus 64 MiB (`verifier/pad.py`).
The honest calibration records the bytes outside the arena plus a 1/16 GiB margin as the allowance, and every later
run on that box, cheaters included, is held to that allowance. A cheater prover sizes its own arena to exactly the
capacity minus the allowance. vLLM captures CUDA graphs at powers of two only, which saves about 0.6 GiB of
driver memory. An honest run that exceeds its pad dies with an out-of-memory error, so the probe and the runs must
use the same prompt shape.

## Encoding

Weights, KV and filler all use the **butterfly keystream** (`encodings/bfly_ref.py`, `gpu/cuda/bfly.cuh`,
`encodings/kvbfly.py`): stored bytes are the plaintext XOR a pad, and the pad of a block cannot be computed in part.

- **Blocks.** A block is 2^L units of 16 bytes. Weights: L = 11, a 128 x 128 bf16 tile of one matrix, 32 KiB. KV
  and filler: L = 8, 4 KiB (for KV, 16 tokens x 128 dims of one head).
- **Leaves.** Each unit starts as two Philox rounds over (block id, unit index, key).
- **Stages.** L stages wired as a butterfly network: stage l mixes unit i with unit i XOR 2^l (multiply-add on one
  half, rotate and XOR with a constant on the other, the order alternating by stage). After L stages every unit
  depends on every leaf of its block.
- **Feed-forward.** The state after stage L-4 is added to the final state, so the pad is a one-way function of the
  leaves and kept bytes of a block reveal nothing about its interior.

Whole blocks are cheap: the honest kernels read whole tiles anyway and the pad of a block is generated in one
pass, about 2,000 GB/s on an H200 (filler about 1,700 GB/s). Single reads are not: a probe into a block that was
not kept needs the whole block's network, in series with the rest of its chain. Measured on Boxes W and X
(`tests/bench_regen_probe_gpu.py`): dropping 0.1% of the weight pages adds about 18 ms to one answer, 1% adds
28 ms, 10% adds 279 ms, against an honest answer of 3.8 ms. Under sloth64, the encoding this replaced, 1% added
3.3 ms and bulk decode ran at 61 GB/s.

What the encoding does not stop is restoring borrowed memory in bulk just before an answer; the timing rules
below price that. The construction is custom, built for decode speed and rebuild cost; it has had no cryptographic
review.

- **Weights.** Two cases, both feeding the stock arithmetic:
  - `gate_up` (the largest matrix) below 512 tokens in the step: a fused kernel that generates the pad and
    multiplies (`VARENA_BFLY_FUSE_KINDS=gate_up`; one CTA per 256 rows above 128 tokens, so a tile's pad is
    generated once);
  - everything else: decoded ahead into one shared 270 MiB scratch (in row slices when the matrix is larger than
    the scratch), then multiplied by a copy of vLLM's persistent matmul whose CTAs exit at a challenge and resume
    after it (`VARENA_AHEAD_AE`, `VARENA_AE_PREEMPT`). The decode kernel is held to two blocks per SM
    (`VARENA_BFLY_XOR_SMEM_KB=45`), which smooths its bursts of memory traffic and is what lets the noise stream
    hold its rate while serving.
- **KV** is encoded in the cache write and decoded inside the attention kernel, a whole block per tile, each chunk
  looking up its own cache block. The cache itself stays encoded.
- **Stale KV.** A freed KV block keeps its old content instead of being refilled with filler, and stays claimable
  as the pages of the request that last wrote it.
- **Activations are not encoded.** They live in the unchecked memory, and there is no activation pool. A run that
  declares one in place of filler fails the `no_act_pool` check.

**Bit-exactness.** Every decode feeds the same kernel the same values that stock deterministic vLLM would, so
outputs are identical token for token (300 of 300 prompts in every honest run). The encoded kernels are generated
from the installed vLLM's own source, and each substitution must match exactly once, so a vLLM upgrade fails
loudly. The embedding table stays plain: replacing its lookup changed how the compiler fused later operations.

## The challenge

One kind of challenge, sent at random times, 4 per second (`verifier/verifier.py`, kernel in `gpu/cuda/chase.cu`,
native path in `gpu/native.py`): a chase over all of the sealed memory.

- **The chain.** 4,224 chains of 1,200 rounds, 8 reads of 128 bytes per round, over a nonce-chosen range of the
  weights W, the KV region K, the filler F and the noise ring. The next addresses depend on the bytes just read, so
  the reads cannot be batched or predicted. The round mixes every byte (`t = rotl(t ^ w, 23) * c`).
- **The ring inside the chain.** Ring granules are addressed relative to the head: logical slot q is the window q
  behind the newest completed one, and every one of the 64 slots is read. The 24 oldest are the slots incoming
  windows overwrite while the chase runs: a read there is tagged like an in-flight KV read, and the verifier takes
  the old window or the incoming one by the tag (`--ring-tag 1`). No ring memory is left unread. The prover returns
  the head it used with the answer. (The cheaters that serve nothing answer an older form that skips those 24
  slots, because the tags ride on the block-table path.)
- **KV in flight.** A KV position that a step in flight may be writing is read like any other, and the kernel also
  returns a 32-bit tag of the 128 bytes it read (up to 255 per chain, shared with the ring's tagged reads). The tag lets the verifier tell whether the
  position still held its old content or already the new one; the chain value is what checks the bytes.
- **The block table.** Which request wrote which KV position is a function of the engine scheduler's state. The
  engine records the order of its scheduler's calls; the verifier replays them on the engine's own scheduler code
  on a CPU (`verifier/schedreplay.py`), with its own requests and the tokens it received, and derives the table for
  every step. Each timed answer carries the version and digest of the table the kernel ran under, and that digest
  must equal the verifier's.
- **Replay.** After the session the verifier recomputes every request's KV once with a reference engine and
  replays sampled chains from sources only: by default one chain of each of 4 randomly chosen answers per run
  (about 9,600 probed chunks per chain); `MODE=full` replays one chain of every answer (`verifier/chainreplay.py`):
  declared weights, filler, its own noise pool at the returned head, and KV by the derived table. Integer
  arithmetic throughout. A chain that does not reproduce fails the run.
- **Timing.** Judged over the run against this host's honest calibration, by two rules. Tail: at most one answer
  beyond the host's multiple of T (T is the honest 99th percentile; the multiple comes from the host's honest runs
  and is at least 1.5). Mean: with that one slowest answer dropped, the mean of the rest within 5% of the honest
  mean taken the same way, on the slowest of the host's honest floor runs. The mean bounds the total restore work over a session however it is spread.
  An answer that overlaps one of the verifier's own
  freezes is not timed.

While a challenge runs, vLLM pauses scheduling (`VARENA_PAUSE_VLLM`), its matmul waits at tile boundaries
(`VARENA_MM_GATE`) and two SMs are reserved; the honest mean is about 3.8 ms on Box W against 3.16 ms on an idle
GPU. That gap, plus the mean rule's 5%, is the answer-time room a cheater has.

No rule depends on how fast the prover computes inference.

## Link noise

- The verifier fills a 64 GiB pool in shared memory with random bytes. A window is 32 MiB; the ring on the GPU
  holds the newest 64.
- The pool is walked in order (`--noise-perm 2`): the schedule is public. A prover that kept an old ring and waited
  for the walk to come round would have to hold its head still and then jump it a whole cycle, which the head rule
  forbids.
- **Saturating mode.** The next window is revealed as soon as fewer than four are outstanding
  (`VARENA_SAT_LEAD=4`), so copies run back to back at the link's own rate, about 56 GB/s on a good host.
- The verifier measures the rate from its own stamps of window completions, never from prover reports.
- Only the inbound direction carries noise.

**One queue for every host-to-device copy** (`tools/h2d_route.c`, `gpu/native.py`). With the link saturated, vLLM's
own small copies starve behind the noise. An `LD_PRELOAD` interposer hands every host-to-device copy of at most
1 MiB to a queue next to the noise stream. Each caller stream gets its own queue stream: the copy waits for an
event recorded on the caller's stream, and the caller's later work waits for the copy, so stream order is exactly
what a direct copy would give. The noise stream yields to copies that are ready. Two earlier designs deadlocked on
the H200 (a wait on a flag value holds a shared hardware queue; one queue stream for all callers lets a gated
stream block the rest), which is why the design is events only and per caller. A guard thread turns routing off,
loudly, if a routed copy does not land within 5 s of its stream being ready, or its stream is not ready after
30 s.

## Link rules and the floor

A cheater run is judged against a floor from honest runs on the same host (`scripts/reproduce/floors.sh`,
`verifier/floor.py`). Honest runs carry no floor; they set it.

| rule | what it checks |
|---|---|
| end of session | the bytes that had arrived by the last timed answer, over the elapsed time, at least the floor |
| head | the head returned with each answer never goes back and never advances faster than the link allows; the rate counted from the first to the last head is at least the floor |

The floor is the lowest end-of-session rate of three honest runs minus 0.12 GB/s. The verifier's own frozen time
(its noise process logs every stall of its own loop) is taken off the clock before either rate is computed.

The floor is only as good as the host and the hour: hosts differ by about 1 GB/s, and one host's honest rate
drifted by 0.3 GB/s within a day as its link's correctable-error rate changed. On a noisy host three runs do not
capture the spread: on Box X (16 Nov) four honest runs ended between 54.77 and 55.51 GB/s, and the fourth would
have failed the floor set by the first three. On Box W the three runs spanned 0.02 GB/s. A cheater counts as caught only if
the idle control (the cheater prover doing no hidden work) passes the same floor on the same host.

## What the scheme does not cover

- Outbound traffic is not exhausted, so gradients leave the GPU freely.
- NVLink between GPUs of one host carries no noise.
- Two GPUs on one physical host cannot both hold the full noise rate (28 to 38 GB/s each when two ran together).
- Host freezes, noisy hosts and the thin floor margin: `docs/RESULTS.md` (Box X) and "Open problems" in `docs/ITERATE.md`.
- The order of scheduler events is taken from the prover. A wrong order makes chains fail, but what else it could buy has not been analysed.

## How fast a cheater can go (models, not measurements)

A hidden matmul needs weights and activations together on the GPU. With leftover memory M (about 1.1 GiB) and
stolen inbound link S (about 0.24 GB/s on Box W's final floors; 0.3 to 0.6 on looser floors), a cheater can arrange them three
ways:

| arrangement | kept on the GPU | streamed in | rate scales as |
|---|---|---|---|
| weights resident (a pipeline stage; `attacks/max_shard`) | a slice of the parameters | the stage boundary, 20,480 bytes per token at bf16 (activation in, gradient in) | S × M |
| activations resident (weight streaming) | hidden state for a batch of tokens | every weight twice per step | S × M |
| both streamed (tiled matmul) | one tile of each | about 1/√M bytes per FLOP | S × √M |

The first is the best for the cheater and is what the measured pretraining attack does: its token rate equals the
link it steals divided by the bytes per token, so its slowdown is set by the smallest theft the floor catches.

For RL the served weights are already on the GPU, so the cost is elsewhere: every generated token needs every
layer's weights decoded, and the leftover memory holds the attention cache for about two rollouts. Spreading
layers across GPUs does not help, because the cache for all rollouts in flight still has to fit fleet-wide and each
rollout passes through the layers in series.
