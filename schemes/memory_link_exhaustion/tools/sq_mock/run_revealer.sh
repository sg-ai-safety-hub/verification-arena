#!/bin/bash
# Build the native revealer harness (test_revealer.cpp) against the mock runtime; prints the binary's path.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
B=${1:-$(mktemp -d)}
python3 - "$H/../../gpu/native.py" "$B/native_src.cpp" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = s.index('_SRC = r"""') + len('_SRC = r"""')
open(sys.argv[2], "w").write(s[a:s.index('"""\n\n_mod = None')])
PY
[ -f "$B/libcudart_mock.so" ] || g++ -std=c++17 -O1 -shared -fPIC -o "$B/libcudart_mock.so" "$H/mockcuda.cpp" -lpthread
g++ -std=c++17 -O1 -w -I"$B" -I"$H" -o "$B/test_revealer" "$H/test_revealer.cpp" -L"$B" -lcudart_mock -Wl,-rpath,"$B" -ldl -lpthread
echo "$B/test_revealer"
