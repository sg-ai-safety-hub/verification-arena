"""Newline-delimited JSON over a Unix socket."""

import json
import socket
import threading


class Conn:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.f = sock.makefile("rb")
        self.wlock = threading.Lock()

    def send(self, obj):
        data = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
        with self.wlock:
            self.sock.sendall(data)

    def recv(self):
        line = self.f.readline()
        if not line:
            return None
        return json.loads(line)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
