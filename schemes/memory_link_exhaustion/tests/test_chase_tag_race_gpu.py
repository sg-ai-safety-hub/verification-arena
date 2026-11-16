"""Can a tagged in-flight read see a half-written chunk? (GPU; passes when the tag set matches the kernel with no
writer and with an ordered writer. The torn-read counts of the unordered modes are measurements, not pass criteria:
14 Nov, one H200: 0 of 42 million with plain copies, 45 of 42 million with the encode kernel.)

A 4 GiB arena as in test_chase_tag_gpu (2 GiB weights-like, 2 GiB KV, 32 layers x 1,024 blocks). Blocks 0..31 are
marked fully in flight, so the chase tags every read that lands there (1,024 pages, 64 MiB). A second thread on its
own CUDA stream rewrites in-flight content while the chase runs on the default stream, with nothing ordering the two:

  mode 0  no writer                              every tag must be the tag of content A (validates the tag set)
  mode 1  plain copies, A <-> B, of all 64 MiB   a tag must be A's or B's; anything else is a torn read
  mode 2  the sloth encode kernel (the one that writes KV), re-encoding layer 0's 2 MiB from two plaintexts
  mode 3  mode 1 on the chase's own stream       ordered by the stream: no torn read possible

A tag is 32 bits over the chunk's 128 bytes, the chase nonce and the chunk's address, so a chunk holding part of A
and part of B gives a tag in neither set (up to a 2^-32 * set-size chance). Prints reads and torn reads per mode.
Run: python -m schemes.memory_link_exhaustion.tests.test_chase_tag_race_gpu [launches per mode, default 200]"""
import os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np, torch
from schemes.memory_link_exhaustion.encodings import ref, sloth64
from schemes.memory_link_exhaustion.gpu.cuda import build as CB

N_LAUNCH = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ext = CB.load()
G = ref.CH_GRAN_PAGES
nb_kv, L, NB = 1024, 32, 32
nW, nK = 32 * G, L * nb_kv
npg = nW + nK
g = torch.Generator(device="cuda"); g.manual_seed(11)
base = torch.randint(-2**31, 2**31 - 1, (npg * 16384,), dtype=torch.int32, device="cuda", generator=g)
kv = base[nW * 16384:].view(L, nb_kv, 16384)
A = kv[:, :NB].clone()
B = torch.randint(-2**31, 2**31 - 1, A.shape, dtype=torch.int32, device="cuda", generator=g)
gp = ref.ch_granules_mix(nW, nK, 0, 0, 0)
tb = len(gp) // ref.CH_GRAN_PER_BLOCK
ga = torch.tensor([base.data_ptr() + p * 65536 for p in gp], dtype=torch.int64, device="cuda")
gkv = torch.tensor([p - nW if nW <= p < nW + nK else -1 for p in gp], dtype=torch.int32, device="cuda")
lo = np.full(nb_kv, 16, dtype=np.uint16); lo[:NB] = 0
kvtab = torch.from_numpy((lo | np.uint16(16 << 8)).view(np.int16).copy()).cuda()
info = torch.tensor([1, 1], dtype=torch.int64, device="cuda")
Kc, S, P, NONCE = 132 * 32, ref.CH_TAG_SLOTS, 400, 12345
nonce = torch.tensor([NONCE, 0], dtype=torch.int32).pin_memory()
out = torch.zeros(Kc + 2 + Kc * S // 2, dtype=torch.int64).pin_memory()
M = np.uint64(0xFFFFFFFFFFFFFFFF)
LW = np.array([(31 ** bin(l).count("1")) & 0xFFFFFFFFFFFFFFFF for l in range(32)], dtype=np.uint64)


def mix(z):
    z = z ^ (z >> np.uint64(30)); z = z * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(27)); z = z * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def tags_of(content, layers):
    """Tags of every chunk of content [len(layers), NB, 16384] int32 (the in-flight pages of those layers)."""
    w = content.cpu().numpy().view(np.uint32).reshape(len(layers), NB, 512, 32).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = (w * LW).sum(-1, dtype=np.uint64)
        page = (np.asarray(layers, dtype=np.uint64)[:, None, None] * np.uint64(nb_kv)
                + np.arange(NB, dtype=np.uint64)[None, :, None])
        cid = page * np.uint64(512) + np.arange(512, dtype=np.uint64)[None, None, :]
        n64 = mix(np.array([NONCE], dtype=np.uint64))[0]
        t = mix(h ^ n64 ^ mix(np.uint64(ref.CH_TAG_SALT) ^ cid)) >> np.uint64(32)
    return t.astype(np.uint32).ravel()


def chase(stream=None):
    if stream is None:
        ext.chase_honest_full(ga, tb, 1, nonce.data_ptr(), Kc, P, out, 0, 1, gkv, kvtab, nb_kv, info.data_ptr(), 0x100)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            ext.chase_honest_full(ga, tb, 1, nonce.data_ptr(), Kc, P, out, 0, 1, gkv, kvtab, nb_kv, info.data_ptr(), 0x100)
        stream.synchronize()
    t = out.numpy()[Kc + 2:].view(np.uint32).reshape(Kc, S)
    cnt = np.minimum(t[:, 0], S - 1)
    return np.concatenate([t[c, 1:1 + cnt[c]] for c in range(Kc)]), int((t[:, 0] > S - 1).sum())


def run(mode):
    kv[:, :NB] = A
    torch.cuda.synchronize()
    stop, nwr = threading.Event(), [0]
    ws = torch.cuda.Stream()
    if mode == 2:
        seed = 0x5EED
        pa, pb = A[0].contiguous().view(torch.int64).view(-1), B[0].contiguous().view(torch.int64).view(-1)
        dst = kv[0, :NB].view(torch.int64).view(-1)            # layer 0, blocks 0..NB-1: contiguous
        ea, eb = torch.empty_like(pa), torch.empty_like(pb)
        sloth64.encode_to(pa, ea, seed); sloth64.encode_to(pb, eb, seed)
        torch.cuda.synchronize()
        dst.copy_(ea)
        rest = kv[1:, :NB]
        valid = np.unique(np.concatenate([tags_of(ea.view(torch.int32).view(1, NB, 16384), [0]),
                                          tags_of(eb.view(torch.int32).view(1, NB, 16384), [0]),
                                          tags_of(rest, list(range(1, L)))]))

        def writer():
            with torch.cuda.stream(ws):
                while not stop.is_set():
                    sloth64.encode_to(pb if nwr[0] % 2 == 0 else pa, dst, seed)
                    ws.synchronize(); nwr[0] += 1
    else:
        valid = np.unique(np.concatenate([tags_of(A, list(range(L))), tags_of(B, list(range(L)))]))

        def writer():
            with torch.cuda.stream(ws if mode == 1 else torch.cuda.default_stream()):
                while not stop.is_set():
                    kv[:, :NB] = B if nwr[0] % 2 == 0 else A
                    if mode == 1:
                        ws.synchronize()
                    else:
                        time.sleep(0.0002)                    # (same stream as the chase: let launches interleave)
                    nwr[0] += 1
    th = None
    if mode:
        th = threading.Thread(target=writer, daemon=True); th.start()
        time.sleep(0.2)
    reads = torn = over = 0
    t0 = time.time()
    for _ in range(N_LAUNCH):
        tg, ov = chase()
        reads += tg.size; over += ov
        torn += int((~np.isin(tg, valid)).sum())
    dt = time.time() - t0
    stop.set()
    if th is not None:
        th.join()
    torch.cuda.synchronize()
    name = {0: "no writer", 1: "plain copies, own stream", 2: "sloth encode kernel, own stream", 3: "plain copies, chase's stream"}[mode]
    print(f"mode {mode} ({name}): launches {N_LAUNCH}, tagged reads {reads:,}, torn {torn:,} "
          f"({torn / max(reads, 1):.2e} per read), rewrites during the test {nwr[0]:,}, chains over the tag cap {over}, "
          f"{dt:.1f} s", flush=True)
    return reads, torn


res = {m: run(m) for m in (0, 1, 2, 3)}
ok = res[0][1] == 0 and res[0][0] > 0 and res[3][1] == 0
print("BASELINE OK" if ok else "BASELINE FAIL (tag set does not match the kernel with no writer)")
sys.exit(0 if ok else 1)
