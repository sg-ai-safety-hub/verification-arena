# Run record

The runs behind the numbers in `docs/RESULTS.md`: run tag, box and what was measured. All runs use the reference
configuration (`scripts/reproduce/reference.env` plus `lib.sh`) on one rented H200 NVL with Qwen2.5-32B-Instruct,
ShareGPT, unless a row says otherwise. Raw run output is not tracked; a run tag names its output directory on the
box. A number is only compared with baselines and floors from the same box.

## Final configuration (16 Nov)

**Box W:** rented H200 NVL. Stock inference 1,127.4 tok/s (`base_stockdet`); raw link 57.74 GB/s
(`check_host`).

| run | code | what | result |
|---|---|---|---|
| `haf1`-`haf3` | `aca848a` | honest, all butterfly, no kernel work | 624-625 tok/s; end 56.28-56.29; floor 56.16 |
| `hh2`, `hs1`, `hc1` | `9128ee4`-`e753934` | honest: + fused `gate_up`; + smoothing; both | 615.5, 604.9, 606.1 tok/s; end 56.53, 56.50, 56.64 |
| `hq1`, `hp1`, `hw1`, `ha1`, `hy1`, `hs8`, `hm1` | `b8b18e2`-`fba56af` | honest answer-gap ladder (stronger pause, exit-and-resume, yielding decode, resume-N, fused yield, slices, every matrix fused) | see the configuration table in `docs/RESULTS.md` |
| `hr2` | `42dece9` | honest, reviewed defaults | 596.8 tok/s; 300/300; end 56.67; mean answer 3.79 ms |
| `hf1`-`hf3` | `4d0f79c` | honest floor runs | 596.5, 597.9, 597.0 tok/s; 300/300; end 56.68, 56.66, 56.67; floor 56.54, MEANREF 3.841 ms, tail 1.5 |
| `idle_control_native` | `4d0f79c` | idle control | not caught, 56.81 |
| `max_shard_fitnh_loop_A_l05` | `4d0f79c` | paced pretraining | not caught, 56.57; 36.6x |
| `local_rl_a16_g2_smooth_loop_nh` | `4d0f79c` | paced RL | caught by 0.01 (56.53); 101x |
| `max_shard_fitnh_loop_A_l05` | `6cf5a86` | paced pretraining | not caught, 56.57; 36.0x |
| `local_rl_a16_g2_smooth_loop_nh` | `6cf5a86` | paced RL | not caught, 56.57; 103x |
| `bench_regen_probe_gpu` | `6cf5a86` | regenerate-on-probe benchmark | +28.1 ms per answer at 1% of pages |

Against the `haf` floors (56.16) the same paced cheaters measured 14.1x (pretraining) and 59x (RL).

**Box X:** rented H200 NVL; raw link 56.86 GB/s at 32 MiB (`check_host`).

| run | code | what | result |
|---|---|---|---|
| `xc2`, `xp1`, `xy1` | `fba56af` | honest, 150 prompts (comparable with each other only): no pause; stronger pause and exit-and-resume; plus fused yield | mean answer 5.653, 5.408, 5.504 ms |
| `pad`, `xf1`-`xf3` | `6cf5a86` | pad probe; honest floor runs | 605.5, 604.6, 604.5, 605.5 tok/s; 300/300; end 54.24 (pad), 55.21, 55.48, 55.51; floor 55.09 |
| `idle_control_native` | `6cf5a86` | idle control against the three-run floor | not caught, 55.16 |
| `xf4` | `6cf5a86` | fourth floor run | 605.7 tok/s; end 54.77; floor from four runs 54.65, tail 2.04 |
| `max_shard_fitnh_loop_A_l05` | `6cf5a86` | paced pretraining, four-run floor | not caught, 55.27; 19.9x |
| `local_rl_a16_g2_smooth_loop_nh` | `6cf5a86` | paced RL, four-run floor | not caught, 56.00; 73x |
| `bench_regen_probe_gpu` | `6cf5a86` | regenerate-on-probe benchmark | +28.0 ms per answer at 1% of pages |

## Earlier runs

The runs of 13 to 15 Nov (Boxes I to U, sloth64 encoding, the earlier verifier and catalogue) are not
included in this release.
