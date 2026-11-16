#!/usr/bin/env bash
# All CPU tests (no GPU needed, ~2 min): the attack suites and the scheme's CPU checks.
#   bash scripts/reproduce/cpu_tests.sh            -> one PASS/FAIL line per suite, exit 1 on any failure
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= GSHARD_NO_GATE=1
rc=0
for m in max_shard local_rl gated_shard ring_reuse; do
  out=$(timeout 600 python scripts/reproduce/cpu_suite.py $m 2>&1 | grep -v PROGRESS | tail -1)
  echo "attack $m: $out"; grep -q "failed 0" <<<"$out" || rc=1
done
for t in test_noise_perm_cpu test_noise_seq_cpu test_ring_chase_cpu test_head_rule_cpu test_settled_kv_cpu test_chase_mix_cpu test_kvtable_cpu test_chain_replay_cpu test_floor_carry_cpu test_gpu_cpus_cpu test_freeze_credit_cpu test_replay_cpu test_replay_bfly_cpu test_kvclaim_cpu test_verifier_flags_cpu test_pad test_act_pool_cpu; do
  if timeout 600 python -m schemes.memory_link_exhaustion.tests.$t > /tmp/$t.out 2>&1; then echo "test $t: PASS"; else echo "test $t: FAIL (see /tmp/$t.out)"; rc=1; fi
done
exit $rc
