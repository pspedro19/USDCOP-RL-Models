"""Tests for the `with_server.py` module that this repo ADOPTED from the marketplace.

Why these exist: `tests/regression/test_quant_library_gate.py` refuses to let an adopted
skill ship executable code that CI cannot verify. `.claude` sits in `norecursedirs`, so a
default pytest run never imports this module — only the specs-gate reaches
`.claude/skills/*/scripts/tests` explicitly. Without a test here, `with_server.py` was
tracked code that nothing could exercise.

The two behaviours pinned below are the ones that can silently ruin a run:

* `is_server_ready` deciding a dead port is alive would make every downstream automation
  fail against a server that was never up, with a misleading error.
* the `--server`/`--port` arity check: if it stopped firing, `zip()` would silently DROP
  the extra server, and the command would run against a partially-started stack.

Both are asserted against real sockets and a real subprocess — no mocks that could keep
passing after the logic underneath them changed.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "with_server.py"


def _load_module():
    """Import with_server.py by path (its directory is not an importable package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("with_server_under_test", MODULE_PATH)
    assert spec and spec.loader, f"cannot load {MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def with_server():
    assert MODULE_PATH.is_file(), f"the adopted skill lost its module: {MODULE_PATH}"
    return _load_module()


def _free_port() -> int:
    """A port number nothing is listening on (bind, read it back, release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_is_server_ready_detects_a_real_listening_socket(with_server):
    """True must mean 'something actually accepted a connection', not 'time passed'."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = int(listener.getsockname()[1])

        assert with_server.is_server_ready(port, timeout=5) is True


def test_is_server_ready_reports_false_on_a_dead_port(with_server):
    """A closed port must yield False — and must not hang past its own timeout."""
    port = _free_port()

    started = time.monotonic()
    ready = with_server.is_server_ready(port, timeout=2)
    elapsed = time.monotonic() - started

    assert ready is False
    # The loop polls every 0.5s; allow generous slack for slow CI but still catch a
    # timeout that is ignored entirely.
    assert elapsed < 10, f"is_server_ready ignored its timeout (took {elapsed:.1f}s)"


def test_mismatched_server_and_port_counts_are_rejected():
    """Two servers with one port must ABORT, never silently drop the extra server."""
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--server", "echo one",
            "--server", "echo two",
            "--port", str(_free_port()),
            "--", sys.executable, "-c", "pass",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1, (
        "arity check did not abort; zip() would have dropped a server silently. "
        f"rc={result.returncode} stdout={result.stdout!r}"
    )
    assert "must match" in result.stdout


def test_missing_command_is_rejected():
    """No command after `--` is a usage error, not a no-op success."""
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--server", "echo one",
            "--port", str(_free_port()),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "No command specified" in result.stdout
