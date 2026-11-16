# Results

The measured numbers for the final configuration (16 Nov 2026). Every number comes from one H200 NVL with
Qwen2.5-32B-Instruct on a rented host, named by letter in the order the hosts were rented. They are measurements on
those hosts, not guarantees; the runs behind them are listed in `results/README.md`.

## Final configuration: butterfly encoding (2026-11-16, Box W)

All-butterfly encoding (weights L = 11, KV and filler L = 8), fused decode for `gate_up`, smoothed decode-ahead,
exit-and-resume matmul; tail rule and mean rule; 4 sampled chains per run. Box W ran the floors and the first cheater
set at `4d0f79c` and the second cheater set at `6cf5a86` (the catalogue reduced to the two paced cheaters, no change
to the honest path). H200 NVL, one GPU, ShareGPT, 300 prompts for honest runs. A second host, Box X, turned out to
be noisy and shared; its runs are kept out of these results and recorded in their own section below.

### Honest inference

| | Box W |
|---|---|
| stock inference | 1,127.4 tok/s |
| honest floor runs, tok/s | 596.5, 597.9, 597.0 (1.89x) |
| outputs identical to stock | 300 of 300 in every run |
| honest end of session, GB/s | 56.68, 56.66, 56.67 |
| link floor (lowest minus 0.12) | 56.54 |
| mean answer without the slowest, ms | 3.84 (slowest run) |
| chase kernel alone, ms | 3.30 |
| tail multiple from the honest runs | 1.5 |
| memory outside the arena | 1.95 GiB of 139.81 (98.6% checked) |

- Answers are about 0.5 ms slower than the chase kernel alone; the verifier's own process froze 0.03 s per run.
- The raw link on Box W (`check_host.sh`, pinned copies, no vLLM) is 57.74 GB/s, and 57.48 at the 32 MiB window
  size. The floor is 97.9% of the raw rate. An idle run reaches 56.81, so about 0.27 GB/s lies between idle and
  the floor.

How the honest configuration was chosen (Box W, 300 prompts, one run each unless noted):

| configuration | tok/s | end GB/s | mean answer ms |
|---|---|---|---|
| all sloth64 (previous design) | 674 | 56.62 | |
| all butterfly, no kernel work (three runs) | 624 to 625 | 56.28 | 3.77 to 3.80 |
| + fused `gate_up` | 615.5 | 56.53 | 3.84 |
| + fused `gate_up` + smoothing | 606.1 | 56.64 | 3.84 |
| **+ exit-and-resume (the default; four runs)** | **596.5 to 597.9** | **56.66 to 56.68** | **3.79 to 3.84** |
| + stronger pause (opt-in) | 586.6 | 56.65 | 3.73 |
| + fused yield (opt-in) | 558.4 | 56.69 | 3.66 |
| every matrix fused | 476.3 | 56.74 | 4.13 |

The fused decode and smoothing raise the honest end-of-session rate by 0.38 GB/s, which raises the floor by the
same amount. The pause and yield options take at most 0.18 ms off a 0.68 ms gap to the idle answer (3.16 ms) for 8%
of throughput and are off by default.

### Cheaters

| | `4d0f79c` | `6cf5a86` |
|---|---|---|
| idle control | not caught, 56.81 GB/s, slowest answer 0.99 T | |
| paced pretraining, 283M stage, 450 prompts | not caught, 56.57 GB/s; 5,677 tok/s, one optimiser step of 182 s; 36.6x | not caught, 56.57 GB/s; 5,765 tok/s; 36.0x |
| paced RL, two calls of 2 samples | caught by the end-of-session rule by 0.01 GB/s (56.53 against 56.54); calls 190.9 and 130.1 s; 101x | not caught, 56.57 GB/s; calls 206.5 and 121.2 s; 103x |

- Read RL as about 100x, not caught: the two runs are the same cheater at the same speed on either side of the
  floor by 0.01 and 0.03 GB/s.
- Against the floors of the plain-butterfly configuration (56.16) the same cheaters measured 14.1x and 59x. The
  floor rose by 0.38 GB/s and the link left to a paced cheater fell from about 0.61 to about 0.24 GB/s. The
  pretraining rate fell by the same factor (2.6), as its model predicts. RL fell by 1.7: it moves little data, but
  its GPU work costs the noise stream rate and its pacer rests more when the headroom is smaller.
- The slowdowns are therefore a property of the host's idle-to-floor gap as much as of the scheme.
- The pretraining cheater moved 34.3 GB from the GPU to the host in its session. That direction is not charged.

### Regenerating dropped pages per probe

`tests/bench_regen_probe_gpu.py`: one answer's shape (4,224 chains, 1,200 rounds of 8 probes) with a share of the
weight pages regenerated per probe instead of read. Kernel time added to one answer:

| pages dropped | butterfly, Box X | butterfly, Box W | sloth64 |
|---|---|---|---|
| 0.1% | +18.4 ms | +18.9 ms | +0.5 ms |
| 0.3% | +18.9 ms | | +0.9 ms |
| 1% | +28.0 ms | +28.1 ms | +3.3 ms |
| 3% | +83 ms | | +11 ms |
| 10% | +279 ms | +279 ms | +39 ms |

Bulk restore runs at about 2,000 GB/s for butterfly and 61 GB/s for sloth64. The benchmark's own all-resident time
is 6.36 ms (it uses the regenerating kernel's read path), so the differences are the result, not the totals.

### Known gaps in this round

- Outbound traffic and NVLink are not exhausted.
- About 0.5 ms of the honest answer gap on Box W is not explained by any single kernel that could be paused.
- A cheater that borrows arena memory between challenges and restores it in bulk was measured only in part, on
  earlier floors (about 0.67 ms of restore per GiB); a variant that keeps its gradients on the GPU was not
  measured. Those cheaters are no longer in the catalogue.
- The butterfly construction has had no cryptographic review.

## A noisy, shared host (Box X, 2026-11-16): not part of the main results

Box X ran the same sequence at `6cf5a86`. Its link noise varied more between honest runs than any cheater moved
it, so its cheater slowdowns measure the host and are not counted above. It is recorded because it shows where the
floor rule stops working.

| | Box X | Box W, for comparison |
|---|---|---|
| honest floor runs, tok/s | 604.6, 604.5, 605.5, 605.7 (300 of 300 identical in every run) | 596.5 to 597.9 |
| honest end of session, GB/s | 55.21, 55.48, 55.51, 54.77 (spread 0.74) | 56.66 to 56.68 (spread 0.02) |
| worst one-second window inside an honest run, GB/s | 51.96 to 54.31 | 56.43 to 56.44 |
| link floor | 55.09 from three runs, 54.65 from four | 56.54 |
| mean answer without the slowest, ms | 5.26, 5.65, 5.62 (chase kernel alone 3.49 to 3.54) | 3.84 (3.30) |
| tail multiple | 1.98 from three runs, 2.04 from four | 1.5 |
| verifier freezes per run | 0.15 to 0.20 s | 0.03 s |
| idle control | not caught, 55.16 against the three-run floor; not run against the four-run floor | not caught, 56.81 |
| paced pretraining, 283M stage, 450 prompts | not caught, 55.27 GB/s; 10,426 tok/s, one optimiser step of 106 s; 19.9x | 36x |
| paced RL, two calls of 2 samples | not caught, 56.00 GB/s; calls 117.6 and 115.7 s; 73x | about 100x |

- **The fourth honest run would have failed the floor set by the first three** (54.77 against 55.09). Three floor
  runs do not capture a noisy host's spread, and the false-positive rate of the floor rule has not been measured
  beyond these few runs on either host.
- The host check passed (raw link 56.86 GB/s, no PCIe errors), but the host load average was about 26 during honest
  runs, on 192 cores, with nothing of ours running besides the session. Other tenants are the likely cause;
  that was not measured. The host check does not yet measure spread or outside load.
- With the floor 0.6 GB/s below most honest runs, neither cheater was limited by it. The RL cheater's session read
  a higher noise rate than every honest run on the host.
- The stronger pause (150-prompt runs, comparable with each other only) took 0.25 ms off 5.65 ms and the fused
  yield added nothing.

## Earlier rounds (sloth64 encoding, 13 to 15 Nov)

Before 16 Nov weights, KV and filler were encoded with sloth64, the catalogue held about twenty cheaters and the
timing rules were different; those verdicts and slowdowns do not carry over and are not included in
this release. What they established about hosts still holds:

- **Hosts differ by about 1 GB/s in the noise they sustain.** Directly attached GPUs (Boxes R and U: raw link about
  57.4 GB/s, honest end of session 55.5 to 56.3) ran faster noise than GPUs behind a PCIe switch (Boxes S and T:
  raw about 54.6 to 55.0, honest 53.5 to 53.9), and cheater slowdowns split by host with them.
- **The same host drifts.** One host's honest rate moved by 0.3 GB/s within a day and an idle control by 0.21 GB/s
  in an hour; correctable PCIe errors froze whole hosts for about half a second each. Floors are therefore taken
  from honest runs on the same host close in time to the runs they judge, and the verifier credits its own freezes.
- **The answer-time tail rule needs the host's own honest tail** (the multiple of T comes from the host's honest
  runs, at least 1.5).
- **A host with a CPU quota slows the cheaters' host-side work** (RL about 120x there against about 62x elsewhere).
