"""_same_socket (verifier): the socket of the GPU's CPUs comes from the kernel topology, not from CPU numbering.
Fake sysfs trees for the three host shapes we met. CPU only."""
import os, tempfile
from unittest import mock
from schemes.memory_link_exhaustion.verifier import verifier as V


def tree(pkgs):
    d = tempfile.mkdtemp()
    for c, p in enumerate(pkgs):
        os.makedirs(f"{d}/cpu{c}/topology")
        open(f"{d}/cpu{c}/topology/physical_package_id", "w").write(f"{p}\n")
    return d


def socket_of(pkgs, cpus):
    with mock.patch.object(V.os, "cpu_count", return_value=len(pkgs)):
        return V._same_socket(cpus, tree(pkgs))


# two sockets of 128 cores, SMT: cpus 0-127 and 256-383 on package 0, 128-255 and 384-511 on package 1 (Box R)
boxr = [0] * 128 + [1] * 128 + [0] * 128 + [1] * 128
gpu = list(range(128, 256)) + list(range(384, 512))
assert socket_of(boxr, gpu) == set(gpu), "GPU on socket 1: nothing from socket 0"
# two sockets of 64 cores, GPU on socket 0 (Box N)
boxn = [0] * 64 + [1] * 64 + [0] * 64 + [1] * 64
gpu = list(range(0, 64)) + list(range(128, 192))
assert socket_of(boxn, gpu) == set(gpu), "GPU on socket 0: nothing from socket 1"
# sub-NUMA clustering: the GPU's node is half of socket 1; the other half of that socket is added, socket 0 is not
snc = [0] * 32 + [1] * 32
gpu = list(range(32, 48))
assert socket_of(snc, gpu) == set(range(32, 64))
# unreadable topology: empty, so the caller keeps only the GPU's own CPUs
assert V._same_socket([0, 1], "/nonexistent") == set()
print("test gpu_cpus_cpu: PASS")
