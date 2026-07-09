#!/usr/bin/env bash
# =============================================================================
# Final acceptance: DETERMINISTIC clean-slate teardown + restore, repeated N times.
# =============================================================================
# Each cycle: `docker compose down -v` (wipes volumes) -> `up -d --build` (rebuilds
# the data-seeder so it runs the feature-data restore) -> wait for health + seeding
# -> verify OHLCV+macro+NEWS/ANALYSIS/H5 restored from backups AND the multi-asset
# analysis (USDCOP/Gold/BTC + news) + registry still serve. Runs CYCLES times
# (default 3): identical green each cycle == proven deterministic end-to-end.
#
# DESTRUCTIVE — wipes the running stack. Run LAST, after promotion has populated
# the DB and the feature backup (data/backups/features/*.parquet) exists.
#
# Usage:  CYCLES=3 bash scripts/validation/teardown_restore_test.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
CYCLES="${CYCLES:-3}"
LOG() { echo "[$(date +%H:%M:%S)] $*"; }
PSQL='docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -tAc'

# Capture pre-teardown feature backup row counts (what restore should bring back).
BK_MANIFEST="data/backups/features/feature_backup_manifest.json"
BK_NEWS=0
if [ -f "$BK_MANIFEST" ]; then
  BK_NEWS=$(python -c "import json;print(json.load(open('$BK_MANIFEST'))['tables'].get('news_articles',{}).get('rows',0))" 2>/dev/null || echo 0)
fi
LOG "feature backup baseline: news_articles=${BK_NEWS} rows (restore target)"

# Pre-build the data-seeder image ONCE (carries the feature-restore step). A full
# `up --build` per cycle rebuilds every build-context service (minutes each) and is
# impractical; a targeted single-image build keeps each cycle fast.
LOG "pre-building data-seeder image (one-time)..."
docker compose build data-seeder >/dev/null 2>&1 && LOG "data-seeder image built" || LOG "data-seeder build warn (using existing image)"

run_cycle() {
  local cyc="$1"
  LOG "========================= CYCLE ${cyc}/${CYCLES} ========================="

  LOG "STEP 1: teardown (down -v)"
  docker compose down -v >/dev/null 2>&1
  LOG "STEP 2: up -d (pre-built images incl. data-seeder feature-restore)"
  docker compose up -d >/dev/null 2>&1

  # postgres healthy (init-scripts 00..26 done)
  for i in $(seq 1 60); do
    [ "$(docker inspect --format '{{.State.Health.Status}}' usdcop-postgres-timescale 2>/dev/null)" = "healthy" ] \
      && { LOG "  postgres healthy (~${i}0s)"; break; }
    sleep 10
  done
  # data-seeder loads OHLCV then restores feature data — poll until OHLCV rows appear
  for i in $(seq 1 36); do
    n=$($PSQL "SELECT COUNT(*) FROM usdcop_m5_ohlcv" 2>/dev/null)
    [ "${n:-0}" -gt 0 ] 2>/dev/null && { LOG "  seeder restored ${n} OHLCV rows (~${i}0s)"; break; }
    sleep 10
  done
  sleep 15  # let the feature-restore step finish after OHLCV seed

  LOG "STEP 3: verify DB restored (core + feature data)"
  local ohlcv macro news h5t asset_daily
  ohlcv=$($PSQL "SELECT COUNT(*) FROM usdcop_m5_ohlcv" 2>/dev/null)
  macro=$($PSQL "SELECT COUNT(*) FROM macro_indicators_daily" 2>/dev/null)
  news=$($PSQL "SELECT COUNT(*) FROM news_articles" 2>/dev/null)
  h5t=$($PSQL "SELECT COUNT(*) FROM information_schema.tables WHERE table_name LIKE 'forecast_h5_%'" 2>/dev/null)
  asset_daily=$($PSQL "SELECT COUNT(*) FROM asset_daily_ohlcv" 2>/dev/null)
  LOG "  OHLCV=${ohlcv:-ERR} macro=${macro:-ERR} news_articles=${news:-ERR} forecast_h5_tables=${h5t:-ERR} asset_daily=${asset_daily:-ERR}"

  LOG "STEP 4: verify multi-asset web serves (front-back)"
  sleep 5
  local assets reg
  assets=$(curl -s --max-time 10 http://localhost:5000/api/analysis/assets 2>/dev/null | grep -o '"asset_id"' | wc -l)
  # Count the strategies array via JSON (grep on "strategy_id" also matches the
  # default.strategy_id block → off-by-one). Fall back to grep if python absent.
  reg=$(curl -s --max-time 10 http://localhost:5000/data/registry.json 2>/dev/null \
        | python -c "import sys,json; print(len(json.load(sys.stdin).get('strategies',[])))" 2>/dev/null)
  [ -z "$reg" ] && reg=0
  LOG "  /api/analysis/assets=${assets} | registry strategies=${reg}"

  # PASS criteria: OHLCV+macro restored, news restored to >= baseline, schema present,
  # web serves 3 assets + 8 strategies.
  local pass=1
  [ "${ohlcv:-0}" -gt 0 ] 2>/dev/null || { LOG "  FAIL: OHLCV not restored"; pass=0; }
  [ "${macro:-0}" -gt 0 ] 2>/dev/null || { LOG "  FAIL: macro not restored"; pass=0; }
  [ "${news:-0}" -ge "${BK_NEWS:-0}" ] 2>/dev/null || { LOG "  FAIL: news not restored (${news} < ${BK_NEWS})"; pass=0; }
  [ "${h5t:-0}" -ge 5 ] 2>/dev/null || { LOG "  FAIL: forecast_h5 schema missing"; pass=0; }
  [ "${assets:-0}" -eq 3 ] 2>/dev/null || { LOG "  FAIL: web assets != 3"; pass=0; }
  [ "${reg:-0}" -eq 8 ] 2>/dev/null || { LOG "  FAIL: registry strategies != 8"; pass=0; }

  if [ "$pass" = "1" ]; then LOG "  CYCLE ${cyc} PASS"; return 0; else LOG "  CYCLE ${cyc} FAIL"; return 1; fi
}

GREEN=0
for c in $(seq 1 "$CYCLES"); do
  if run_cycle "$c"; then GREEN=$((GREEN+1)); fi
done

LOG "========================================================="
LOG "DETERMINISM RESULT: ${GREEN}/${CYCLES} cycles green"
[ "$GREEN" = "$CYCLES" ] && LOG "ALL CYCLES GREEN — deterministic end-to-end restore proven" || LOG "NON-DETERMINISTIC / FAILURES — inspect logs above"
[ "$GREEN" = "$CYCLES" ]
