#!/bin/bash
# CPU test of the single H2D queue (gpu/native.py) and tools/h2d_route.so on a mock CUDA runtime whose streams are
# worker threads: routing, stream order (write-after-read on the destination), pageable staging, queue and staging
# wrap-around, cudaMemcpy, pass-through before and after the streamer runs, shared hardware queues. No GPU or torch.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT
python3 - "$H/../../gpu/native.py" "$B/native_src.cpp" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = s.index('_SRC = r"""') + len('_SRC = r"""')
open(sys.argv[2], "w").write(s[a:s.index('"""\n\n_mod = None')])
PY
g++ -std=c++17 -O1 -g -shared -fPIC -o "$B/libcudart_mock.so" "$H/mockcuda.cpp" -lpthread
gcc -O2 -shared -fPIC -o "$B/h2d_route.so" "$H/../h2d_route.c" -ldl
g++ -std=c++17 -O1 -g -w -I"$B" -I"$H" -o "$B/test_sq" "$H/test_sq.cpp" -L"$B" -lcudart_mock -Wl,-rpath,"$B" -ldl -lpthread
# private hardware queues, then 1 and 4 shared ones (CUDA_DEVICE_MAX_CONNECTIONS-style false dependencies; the old
# wait-value design, removed, hung here)
for y in 0 1; do for k in ${MOCK_CONNECTIONS_LIST:-0 1 4}; do
  echo "--- VARENA_SQ_YIELD=$y MOCK_CONNECTIONS=$k"
  VARENA_SQ_YIELD=$y MOCK_CONNECTIONS=$k timeout 120 env LD_PRELOAD="$B/h2d_route.so" "$B/test_sq" \
    || { echo "FAIL (rc $?) at VARENA_SQ_YIELD=$y MOCK_CONNECTIONS=$k"; exit 1; }
done; done
