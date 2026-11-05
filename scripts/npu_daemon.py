#!/usr/bin/env python3
"""Keep a Coral NPU module loaded and run invocations over a unix socket.

The LiteRT dispatch library normally runs one `iree-run-module` process per
invocation.  That costs a module load, a new NPU context, and an upload of every
constant for every call — fine for a prefill-shaped call, hopeless for decoding
one token at a time, where the transformer's weights would be re-uploaded
hundreds of times per second.

This daemon loads the VMFB once, keeps the runtime and its device allocations
alive, and executes each request against them.  `coralnpu_executor_command.cc`
in the Coral NPU dispatch library speaks the protocol.

Protocol (little-endian, one request/reply per invoke, pipelined requests are
not supported):

    request: u64 module hash, u32 name length, name bytes, u32 num inputs,
             then per input: u32 element type, u32 rank, i32 dims[rank],
                             u64 byte size, byte size bytes of data
    reply:   u32 status, u32 num outputs, then per output: u64 byte size and
             that many bytes

The module hash is checked against the loaded VMFB so a daemon started with a
stale artifact fails loudly instead of producing plausible garbage.

Usage:

    npu_daemon.py --module partitions.vmfb --socket /tmp/coralnpu.sock \\
        --device torq

The daemon is single-connection-at-a-time and exits on SIGINT/SIGTERM.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import struct
import sys
import traceback

import numpy as np

# kLiteRtElementType* -> numpy dtype, for the types a Coral NPU partition can
# contain.  Packed 4-bit and fp8 values are deliberately absent: they have no
# numpy equivalent and silently widening them would be worse than failing.
DTYPES = {
    1: np.float32,
    2: np.int32,
    3: np.uint8,
    4: np.int64,
    6: np.bool_,
    7: np.int16,
    9: np.int8,
    10: np.float16,
    11: np.float64,
    13: np.uint64,
    16: np.uint32,
    17: np.uint16,
}

STATUS_OK = 0
STATUS_ERROR = 1
STATUS_MODULE_MISMATCH = 2


def fnv1a(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def read_exactly(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(min(remaining, 1 << 22))
        if not chunk:
            raise EOFError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class Connection:
    def __init__(self, conn: socket.socket):
        self.conn = conn

    def u32(self) -> int:
        return struct.unpack("<I", read_exactly(self.conn, 4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", read_exactly(self.conn, 8))[0]

    def i32(self) -> int:
        return struct.unpack("<i", read_exactly(self.conn, 4))[0]

    def raw(self, size: int) -> bytes:
        return read_exactly(self.conn, size)

    def send_u32(self, value: int) -> None:
        self.conn.sendall(struct.pack("<I", value))

    def send_u64(self, value: int) -> None:
        self.conn.sendall(struct.pack("<Q", value))


class Runner:
    """One loaded module plus the functions LiteRT asks for."""

    def __init__(self, module_path: str, device_name: str, torq_flags: str = ""):
        import iree.runtime as rt

        self.torq_flags = torq_flags
        with open(module_path, "rb") as fh:
            self.module_bytes = fh.read()
        self.module_hash = fnv1a(self.module_bytes)
        module = rt.VmModule.from_flatbuffer(rt.VmInstance(), self.module_bytes)
        # The torq driver consumes global parse flags at device-creation time
        # (mirrors torq.runtime.vmfb.VMFBInferenceRunner).
        flags = ["--task_topology_group_count=1"]
        if device_name == "torq":
            flags += ["--torq_device_allocator=cpu", "--torq_hw_type=astra_machina"]
        if self.torq_flags:
            flags += shlex.split(self.torq_flags)
        rt.flags.parse_flags(*flags)
        # Config() expects a HalDevice object, not a name; resolve via
        # iree.runtime.get_device (system_setup) which caches per URI.
        device = rt.get_device(device_name)
        config = rt.Config(device=device)
        self.module = rt.load_vm_module(module, config)
        self.functions = {}
        print("npu_daemon: loaded %s on %s" % (module_path, device_name),
              flush=True)

    def function(self, name: str):
        if name not in self.functions:
            fn = getattr(self.module, name, None)
            if fn is None:
                raise KeyError("%s is not exported by the module" % name)
            self.functions[name] = fn
        return self.functions[name]

    def invoke(self, name: str, args: list[np.ndarray]) -> list[np.ndarray]:
        results = self.function(name)(*args)
        if not isinstance(results, (list, tuple)):
            return [np.asarray(results)]
        return [np.asarray(r) for r in results]


def serve_connection(conn: socket.socket, runner: Runner) -> None:
    channel = Connection(conn)
    while True:
        try:
            module_hash = channel.u64()
        except EOFError:
            return
        name_len = channel.u32()
        name = channel.raw(name_len).decode()
        count = channel.u32()
        if module_hash != runner.module_hash:
            channel.send_u32(STATUS_MODULE_MISMATCH)
            print("npu_daemon: module hash mismatch for %s" % name, flush=True)
            return

        args = []
        for _ in range(count):
            element_type = channel.u32()
            rank = channel.u32()
            shape = [channel.i32() for _ in range(rank)]
            size = channel.u64()
            data = bytearray(channel.raw(size))
            dtype = DTYPES.get(element_type)
            if dtype is None:
                raise SystemExit(
                    "npu_daemon: element type %d has no numpy representation"
                    % element_type)
            array = np.frombuffer(data, dtype=dtype)
            expected = int(np.prod(shape)) if shape else 1
            if expected != array.size:
                raise SystemExit(
                    "npu_daemon: %s got %d values for shape %r" %
                    (name, array.size, shape))
            args.append(array.reshape(shape))

        try:
            outputs = runner.invoke(name, args)
        except Exception:  # noqa: BLE001 - the reply must carry the failure
            traceback.print_exc()
            channel.send_u32(STATUS_ERROR)
            continue

        channel.send_u32(STATUS_OK)
        channel.send_u32(len(outputs))
        for output in outputs:
            data = np.ascontiguousarray(output).tobytes()
            channel.send_u64(len(data))
            conn.sendall(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", default=os.environ.get("CORALNPU_DAEMON_MODULE"),
                    required=os.environ.get("CORALNPU_DAEMON_MODULE") is None)
    ap.add_argument("--socket", default=os.environ.get("CORALNPU_DAEMON_SOCKET",
                                                       "/tmp/coralnpu.sock"))
    ap.add_argument("--device", default=os.environ.get("CORALNPU_DEVICE", "torq"))
    ap.add_argument("--torq-flags", default="",
                    help="extra torq parse flags, e.g. --torq-flags "
                         "--torq_device_allocator=cpu")
    args = ap.parse_args()

    runner = Runner(args.module, args.device, args.torq_flags)

    path = args.socket
    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(4)
    print("npu_daemon: listening on %s" % path, flush=True)

    def shutdown(*_):
        server.close()
        try:
            os.unlink(path)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        conn, _ = server.accept()
        try:
            serve_connection(conn, runner)
        except EOFError:
            pass
        except Exception:  # noqa: BLE001 - one bad client must not kill the daemon
            traceback.print_exc()
        finally:
            conn.close()


if __name__ == "__main__":
    main()
