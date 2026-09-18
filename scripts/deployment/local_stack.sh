#!/usr/bin/env bash
# Local (dockerless) stack for the USDCOP trading system.
#   Postgres 16 + TimescaleDB · Redis · SignalBridge API (8085)
#   trading API (8000, optional) · Next.js dashboard (5000)
#   nginx terminates TLS on 443 for 174.138.70.157.sslip.io (systemd-managed, enabled at
#   boot, Let's Encrypt cert auto-renewed by certbot.timer) and proxies to :5000.
# Usage: local_stack.sh {start|stop|status|logs}
set -uo pipefail

ROOT=/root/USDCOP-RL-Models
LOGDIR=${STACK_LOGDIR:-/var/log/usdcop-stack}
PIDDIR=/run/usdcop-stack
mkdir -p "$LOGDIR" "$PIDDIR"

start_infra() {
  pg_isready -h 127.0.0.1 -q || pg_ctlcluster 16 main start || true
  redis-cli ping >/dev/null 2>&1 || service redis-server start >/dev/null 2>&1 || true
}

start_signalbridge() {
  [ -f "$PIDDIR/sb.pid" ] && kill -0 "$(cat "$PIDDIR/sb.pid")" 2>/dev/null && return
  cd "$ROOT/services/signalbridge_api" || return 1
  setsid nohup /root/.venvs/sb/bin/python -m uvicorn app.main:app \
    --host 0.0.0.0 --port 8085 --log-level warning \
    > "$LOGDIR/signalbridge.log" 2>&1 < /dev/null &
  echo $! > "$PIDDIR/sb.pid"
}

start_trading_api() {
  [ -x /root/.venvs/api/bin/python ] || return 0
  [ -f "$PIDDIR/tapi.pid" ] && kill -0 "$(cat "$PIDDIR/tapi.pid")" 2>/dev/null && return
  cd "$ROOT/services" || return 1
  set -a; [ -f "$ROOT/services/.env" ] && . "$ROOT/services/.env"; set +a
  setsid nohup /root/.venvs/api/bin/python -m uvicorn trading_api_realtime:app \
    --host 0.0.0.0 --port 8000 --log-level warning \
    > "$LOGDIR/trading-api.log" 2>&1 < /dev/null &
  echo $! > "$PIDDIR/tapi.pid"
}

start_dashboard() {
  [ -f "$PIDDIR/dash.pid" ] && kill -0 "$(cat "$PIDDIR/dash.pid")" 2>/dev/null && return
  cd "$ROOT/usdcop-trading-dashboard" || return 1
  local cmd="run dev"
  [ -d .next/BUILD_ID ] || [ -f .next/BUILD_ID ] && cmd="run start"
  PATH=/root/.local/bin:$PATH setsid nohup /root/.local/bin/npm $cmd \
    > "$LOGDIR/dashboard.log" 2>&1 < /dev/null &
  echo $! > "$PIDDIR/dash.pid"
}

stop_one() { [ -f "$1" ] || return 0; local p; p=$(cat "$1"); pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; rm -f "$1"; }

case "${1:-start}" in
  start)  start_infra; start_signalbridge; start_trading_api; start_dashboard; sleep 18; "$0" status ;;
  stop)   stop_one "$PIDDIR/dash.pid"; stop_one "$PIDDIR/tapi.pid"; stop_one "$PIDDIR/sb.pid"
          ss -ltnp 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u | while read -r p; do
            case "$(ps -p "$p" -o args= 2>/dev/null)" in *next-server*|*uvicorn*) kill "$p" 2>/dev/null;; esac; done ;;
  status) printf '%-14s %s\n' nginx-tls "$(curl -s -o /dev/null -m 8 -w '%{http_code}' https://174.138.70.157.sslip.io/ 2>/dev/null || echo DOWN)"
          printf '%-14s %s\n' postgres "$(pg_isready -h 127.0.0.1 -q && echo UP || echo DOWN)"
          printf '%-14s %s\n' redis "$(redis-cli ping 2>/dev/null || echo DOWN)"
          for s in "signalbridge http://127.0.0.1:8085/health" "trading-api http://127.0.0.1:8000/" "dashboard http://127.0.0.1:5000/api/health"; do
            set -- $s; printf '%-14s %s\n' "$1" "$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$2" 2>/dev/null || echo DOWN)"; done ;;
  logs)   tail -n 40 "$LOGDIR"/*.log ;;
  *) echo "usage: $0 {start|stop|status|logs}"; exit 2 ;;
esac
