// CPU test of the native revealer (gpu/native.py) in saturating mode: a fake prover completes windows in order and
// the revealed values are written to stdout, one "i s" line per window, for comparison with the Python reference
// (tests/test_noise_perm_cpu.py). argv: n_windows ktab_file|- vkey.  Built by run_revealer.sh; no GPU or torch.
#include "native_src.cpp"
#include <cstdio>
int main(int argc, char** argv) {
  const int64_t nw = atoll(argv[1]), N = 1 << 12, STATS = 16, HIST = 64, DL = 1 << 16;
  std::vector<uint32_t> tab;
  if (strcmp(argv[2], "-") != 0) {
    FILE* f = fopen(argv[2], "rb"); fseek(f, 0, SEEK_END); tab.resize(ftell(f) / 4); fseek(f, 0, SEEK_SET);
    if (fread(tab.data(), 4, tab.size(), f) != tab.size()) return 2; fclose(f);
  }
  const int64_t vkey = atoll(argv[3]);
  std::vector<int64_t> a(2 * N + STATS + HIST + N, 0), dl(DL, -1);
  for (int64_t k = 0; k < 2 * N; k++) a[k] = -1;
  setenv("VARENA_SAT_LEAD", "4", 1);
  std::thread th([&] { revealer((int64_t)a.data(), N, vkey, 1e-3, 4e-3, rv::now(), 1.0, 0.0, STATS, HIST, 1e-4,
                                (int64_t)dl.data(), DL, 0, tab.empty() ? 0 : (int64_t)tab.data(), (int64_t)tab.size(), 0, 0); });
  volatile int64_t* rev = a.data(); volatile int64_t* rep = a.data() + N;
  for (int64_t i = 0; i < nw; i++) {
    while ((rev[i % N] >> 32) != i) std::this_thread::yield();
    const uint32_t s = (uint32_t)rev[i % N];
    printf("%lld %u\n", (long long)i, s);
    rep[i % N] = (i << 32) | s;
  }
  fflush(stdout);
  _Exit(0);    // the revealer loops until its parent exits
}
