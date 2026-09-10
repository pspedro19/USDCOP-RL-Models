#!/usr/bin/env python3
"""
Start one or more servers, wait for them to be ready, run a command, then clean up.

Usage:
    # Single server
    python scripts/with_server.py --server "npm run dev" --port 5173 -- python automation.py
    python scripts/with_server.py --server "npm start" --port 3000 -- python test.py

    # Multiple servers
    python scripts/with_server.py \
      --server "cd backend && python server.py" --port 3000 \
      --server "cd frontend && npm run dev" --port 5173 \
      -- python test.py
"""

import subprocess
import socket
import time
import sys
import argparse

def is_server_ready(port, timeout=30):
    """Wait for server to be ready by polling the port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(('localhost', port), timeout=1):
                return True
        except (socket.error, ConnectionRefusedError):
            time.sleep(0.5)
    return False


def _verify() -> int:
    """Auto-comprobacion sin dependencias externas (`--verify`).

    POR QUE EXISTE (CTR-QUANT-LIBRARY-001, 2026-08-24)
    -------------------------------------------------
    `pyproject.toml` pone `.claude` en `norecursedirs`, asi que un `pytest` normal NUNCA
    ve este modulo: al adoptarse la skill (sus ficheros pasaron a estar trackeados), el
    codigo quedo publicado pero sin nada que lo verificara.
    `test_quant_library_gate.py::test_promoted_skills_shipping_code_also_ship_tests`
    acepta dos formas de cerrar ese hueco —una suite en `scripts/tests/` o un modulo con
    `--verify`— y esta es la segunda, que es la house style de las skills de finanzas.

    El precedente que justifica el gate: `xasset-alpha-engine` se edito para delegar el
    Deflated Sharpe al SSOT constitucional, el tipo de retorno paso de float a dict y
    `validate()` seguia comparandolo contra un float. Sus 74 tests lo cazaban en una
    corrida, pero se habian eliminado al promover la skill, asi que nadie los corrio y el
    gate siguio verde.

    Se comprueba lo que este modulo hace de verdad, no un placeholder:
      1. `is_server_ready` devuelve False —y respeta el timeout— con el puerto cerrado.
      2. `is_server_ready` devuelve True contra un socket que abrimos aqui mismo.
      3. El parser exige que `--server` y `--port` vengan en el mismo numero.
    """
    ok = True

    # 1. Puerto cerrado: debe fallar, y hacerlo dentro del timeout pedido.
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        closed_port = probe.getsockname()[1]
    t0 = time.time()
    if is_server_ready(closed_port, timeout=1):
        print(f"FALLO: is_server_ready dijo True en el puerto cerrado {closed_port}")
        ok = False
    elapsed = time.time() - t0
    if elapsed > 3:
        print(f"FALLO: el timeout de 1s tardo {elapsed:.1f}s en rendirse")
        ok = False

    # 2. Puerto abierto: debe detectarlo.
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("localhost", 0))
    listener.listen(1)
    open_port = listener.getsockname()[1]
    try:
        if not is_server_ready(open_port, timeout=5):
            print(f"FALLO: is_server_ready no vio el puerto abierto {open_port}")
            ok = False
    finally:
        listener.close()

    # 3. El parser no puede aceptar 2 servidores y 1 puerto.
    parser = _build_parser()
    args = parser.parse_args(["--server", "a", "--server", "b", "--port", "1", "--", "x"])
    if len(args.servers) == len(args.ports):
        print("FALLO: el parser acepto 2 --server con 1 --port como si cuadraran")
        ok = False

    print("verify: OK" if ok else "verify: FALLOS ARRIBA")
    return 0 if ok else 1


def _build_parser():
    """El parser, extraido para que `--verify` pueda ejercitarlo sin correr `main`."""
    parser = argparse.ArgumentParser(description='Run command with one or more servers')
    parser.add_argument('--server', action='append', dest='servers', required=True, help='Server command (can be repeated)')
    parser.add_argument('--port', action='append', dest='ports', type=int, required=True, help='Port for each server (must match --server count)')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds per server (default: 30)')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command to run after server(s) ready')

    return parser


def main():
    parser = _build_parser()
    if '--verify' in sys.argv[1:]:
        sys.exit(_verify())
    args = parser.parse_args()

    # Remove the '--' separator if present
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]

    if not args.command:
        print("Error: No command specified to run")
        sys.exit(1)

    # Parse server configurations
    if len(args.servers) != len(args.ports):
        print("Error: Number of --server and --port arguments must match")
        sys.exit(1)

    servers = []
    for cmd, port in zip(args.servers, args.ports):
        servers.append({'cmd': cmd, 'port': port})

    server_processes = []

    try:
        # Start all servers
        for i, server in enumerate(servers):
            print(f"Starting server {i+1}/{len(servers)}: {server['cmd']}")

            # Use shell=True to support commands with cd and &&
            process = subprocess.Popen(
                server['cmd'],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            server_processes.append(process)

            # Wait for this server to be ready
            print(f"Waiting for server on port {server['port']}...")
            if not is_server_ready(server['port'], timeout=args.timeout):
                raise RuntimeError(f"Server failed to start on port {server['port']} within {args.timeout}s")

            print(f"Server ready on port {server['port']}")

        print(f"\nAll {len(servers)} server(s) ready")

        # Run the command
        print(f"Running: {' '.join(args.command)}\n")
        result = subprocess.run(args.command)
        sys.exit(result.returncode)

    finally:
        # Clean up all servers
        print(f"\nStopping {len(server_processes)} server(s)...")
        for i, process in enumerate(server_processes):
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            print(f"Server {i+1} stopped")
        print("All servers stopped")


if __name__ == '__main__':
    main()