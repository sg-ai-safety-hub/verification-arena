"""Tagged in-flight reads, kernel vs reference (GPU): chase_honest_full with info_off bit 8 on a 4 GiB arena of random
bytes (half weights-like, half KV with a synthetic block table that masks about half of KV).

  1. chain values and [count, tags] of spot chains equal the reference walk over the same bytes (ref.ch_chain reading
     real memory in a band, ref.ch_tag for each in-band read);
  2. without the tag bit the kernel still gives the stand-in chain (ref with ch_mask_words);
  3. the cost of the tag output: the same chase with and without it, answers written to pinned host memory as in a
     verified run (the tags are written from the kernel straight into that buffer)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np, torch
from schemes.memory_link_exhaustion.encodings import ref
from schemes.memory_link_exhaustion.gpu.cuda import build as CB

ext = CB.load()
G = ref.CH_GRAN_PAGES
nb_kv, L = 1024, 32
nW, nK = 32 * G, L * nb_kv                                   # 2 GiB + 2 GiB
npg = nW + nK
g = torch.Generator(device="cuda"); g.manual_seed(7)
base = torch.randint(-2**31, 2**31 - 1, (npg * 16384,), dtype=torch.int32, device="cuda", generator=g)
gp = ref.ch_granules_mix(nW, nK, 0, 0, 0)
tb = len(gp) // ref.CH_GRAN_PER_BLOCK
assert tb == 1
ga = torch.tensor([base.data_ptr() + p * 65536 for p in gp], dtype=torch.int64, device="cuda")
gkv = torch.tensor([p - nW if nW <= p < nW + nK else -1 for p in gp], dtype=torch.int32, device="cuda")
tab = ref.kv_tab_test(7, nb_kv)
band = [(int(x) & 0xff, int(x) >> 8) for x in tab]
kvtab = torch.from_numpy(tab.view(np.int16).copy()).cuda()
info = torch.tensor([42, 4242], dtype=torch.int64, device="cuda")
Kc, S = 132 * 32, ref.CH_TAG_SLOTS
nonce = torch.tensor([12345, 0], dtype=torch.int32).pin_memory()
bh = base.cpu().numpy().view(np.uint32)
cw = lambda page, wo: bh[page * 16384 + wo: page * 16384 + wo + 32]
inband = lambda page, wo: nW <= page < nW + nK and ref.ch_masked(band[(page - nW) % nb_kv], wo)


def run(P, tagm, host=True):
    n = Kc + 2 + (Kc * S // 2 if tagm else 0)
    out = torch.zeros(n, dtype=torch.int64)
    out = out.pin_memory() if host else out.cuda()
    ext.chase_honest_full(ga, tb, 1, nonce.data_ptr(), Kc, P, out, 0, 1, gkv, kvtab, nb_kv, info.data_ptr(), 0x100 if tagm else 0)
    torch.cuda.synchronize()
    return out.cpu().numpy()


P = 16
o = run(P, True)
st, kvt = o[:Kc].view(np.uint64), o[Kc:Kc + 2]
tags = o[Kc + 2:].view(np.uint32).reshape(Kc, S)
ok1 = int(kvt[0]) == 42 and int(kvt[1]) == 4242
n_in = 0
for c in (0, 7, 1000, 4223):
    tg = []
    want = ref.ch_chain(12345, c, P, tb, 1, gp, cw,
                        on_probe=lambda page, wo, w: tg.append(ref.ch_tag(12345, page - nW, wo, w)) if inband(page, wo) else None)
    n_in += len(tg)
    good = int(st[c]) == want and int(tags[c, 0]) == len(tg) and [int(x) for x in tags[c, 1:1 + min(len(tg), S - 1)]] == tg[:S - 1]
    ok1 = ok1 and good
print(f"tagged kernel == reference (4 chains, P={P}, {n_in} in-band reads):", ok1)

o2 = run(P, False)
st2 = o2[:Kc].view(np.uint64)


def cw_mask(page, wo):
    return ref.ch_mask_words(12345, page - nW, wo) if inband(page, wo) else cw(page, wo)


ok2 = all(int(st2[c]) == ref.ch_chain(12345, c, P, tb, 1, gp, cw_mask) for c in (0, 7, 4223)) and o2.size == Kc + 2
print("stand-in kernel unchanged (3 chains):", ok2)

# 2b. tagged ring reads: a 64-slot ring (2 GiB) in the chase, every slot read, the 24 oldest logical slots tagged
NR, ZONE, HEAD = 64, 24, 1000
ringt = torch.randint(-2**31, 2**31 - 1, (NR << 23,), dtype=torch.int32, device="cuda", generator=g)
ringh = ringt.cpu().numpy().view(np.uint32)
gp4 = ref.ch_granules_mix(nW, nK, 0, 0, ref.ring_chase_granules(NR, ZONE, tagged=True))
tb4 = len(gp4) // ref.CH_GRAN_PER_BLOCK
ga4 = torch.tensor([(p - ref.CH_RA0) // G - ref.CH_RING_TAG if p >= ref.CH_RA0 else base.data_ptr() + p * 65536
                    for p in gp4], dtype=torch.int64, device="cuda")
gkv4 = torch.tensor([p - nW if nW <= p < nW + nK else -1 for p in gp4], dtype=torch.int32, device="cuda")
nonce4 = torch.tensor([12345, HEAD], dtype=torch.int32).pin_memory()
out4 = torch.zeros(Kc + 2 + Kc * S // 2, dtype=torch.int64).pin_memory()
ext.chase_honest_full(ga4, tb4, tb4, nonce4.data_ptr(), Kc, 16, out4, ringt.data_ptr(), NR, gkv4, kvtab, nb_kv,
                      info.data_ptr(), 0x100 | (ZONE << 16))
torch.cuda.synchronize()
o4 = out4.numpy()
st4, tags4 = o4[:Kc].view(np.uint64), o4[Kc + 2:].view(np.uint32).reshape(Kc, S)


def cw4(page, wo):
    if page >= ref.CH_RA0:
        q, w_in = ref.ring_chase_slot(page, wo)
        b = (((HEAD - q) & (NR - 1)) << 23) + w_in
        return ringh[b:b + 32]
    return cw(page, wo)


ok2b, n_ring_tags, n_kv_tags = True, 0, 0
for c in (0, 7, 1000, 4223):
    tg = []

    def probe(page, wo, w):
        global n_ring_tags, n_kv_tags
        if page >= ref.CH_RA0:
            q, w_in = ref.ring_chase_slot(page, wo)
            if q >= NR - ZONE:
                tg.append(ref.ch_tag_id(12345, ref.ring_tag_id(q, w_in), w)); n_ring_tags += 1
        elif inband(page, wo):
            tg.append(ref.ch_tag(12345, page - nW, wo, w)); n_kv_tags += 1
    want = ref.ch_chain(12345, c, 16, tb4, tb4, gp4, cw4, on_probe=probe)
    ok2b = ok2b and int(st4[c]) == want and int(tags4[c, 0]) == len(tg) and \
        [int(x) for x in tags4[c, 1:1 + min(len(tg), S - 1)]] == tg[:S - 1]
print(f"tagged ring kernel == reference (4 chains, {n_ring_tags} ring-zone reads, {n_kv_tags} in-band KV reads):", ok2b)
ok2 = ok2 and ok2b and n_ring_tags > 0
del ringt

# 3. cost at full length with a sparse table (about 1% of KV in a band, close to an honest run): counts, then time
kvtab = torch.from_numpy(ref.kv_tab_test(-7, nb_kv).view(np.int16).copy()).cuda()
o = run(1200, True)
cnt = o[Kc + 2:].view(np.uint32).reshape(Kc, S)[:, 0]
print(f"P=1200: in-band reads per chain mean {cnt.mean():.1f} max {int(cnt.max())} (slots {S - 1}); chains over the cap {(cnt > S - 1).sum()}")
for tagm in (False, True):
    for host in (True, False):
        n = Kc + 2 + (Kc * S // 2 if tagm else 0)
        out = torch.zeros(n, dtype=torch.int64)
        out = out.pin_memory() if host else out.cuda()
        ms = []
        for _ in range(6):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            ext.chase_honest_full(ga, tb, 1, nonce.data_ptr(), Kc, 1200, out, 0, 1, gkv, kvtab, nb_kv, info.data_ptr(), 0x100 if tagm else 0)
            e1.record(); e1.synchronize()
            ms.append(e0.elapsed_time(e1))
        print(f"  chase P=1200 tags={'on ' if tagm else 'off'} answers to {'pinned host' if host else 'device     '}: best {min(ms[1:]):.3f} ms, median {sorted(ms[1:])[2]:.3f} ms")
print("ALL OK" if (ok1 and ok2) else "FAIL")
sys.exit(0 if (ok1 and ok2) else 1)
