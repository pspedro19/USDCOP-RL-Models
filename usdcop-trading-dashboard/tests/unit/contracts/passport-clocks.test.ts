/**
 * F-07 (CODEX) — el tercer reloj y su nota, contra el PAYLOAD real.
 * ==================================================================
 *
 * Dos defectos distintos, un solo origen (una nota escrita a mano):
 *
 *  1. El Passport llamaba `exec` al tercer reloj mientras el productor
 *     (`control__system_health`, `src/monitoring/system_health_contract.py::Clock`)
 *     publica `pnl`. Consecuencia: si `system_health.json` trae el reloj de PnL,
 *     el composer lo descarta y publica `unavailable`. Ya se estaba perdiendo dato.
 *  2. La nota mostrada al usuario decía literalmente "system_health publica data y
 *     model" — hardcodeada. Es una afirmación sobre el payload que el payload puede
 *     desmentir en cualquier momento (y lo desmiente en cuanto BL-25 publique `pnl`).
 *
 * Este archivo prueba lo único que cierra ambos: **la nota se DERIVA del payload**.
 * Se inyectan payloads sintéticos de `system_health.json` (mock de `fs`) y se exige
 * que ninguna nota afirme algo que el payload contradiga.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  HEALTH_CLOCKS,
  isAvailable,
  type HealthClock,
  type Sourced,
  type TowerClock,
} from '@/lib/contracts/passport.contract';

/** Payload sintético de `production/system_health.json` para la corrida en curso. */
const healthOverride = vi.hoisted(() => ({ value: null as unknown }));

vi.mock('fs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('fs')>();
  const readFile = async (p: unknown, enc: unknown) => {
    const norm = String(p).replace(/\\/g, '/');
    if (healthOverride.value !== null && norm.endsWith('production/system_health.json')) {
      return JSON.stringify(healthOverride.value);
    }
    return (actual.promises.readFile as (a: unknown, b: unknown) => Promise<string>)(p, enc);
  };
  const promises = { ...actual.promises, readFile };
  return { ...actual, promises, default: { ...actual, promises } };
});

import { composeControlTower } from '@/lib/passport/compose';

function clockPayload(name: string, signal = 'green') {
  return {
    clock: name, signal, actions: [], events: [], metrics: {},
    evaluated_at: '2026-07-28T00:00:00+00:00',
  };
}

async function clocksFor(published: string[]): Promise<Record<HealthClock, Sourced<TowerClock>>> {
  healthOverride.value = {
    contract: 'CTR-SYSTEM-HEALTH-001',
    version: '1.0.0',
    generated_at: '2026-07-28T00:00:00+00:00',
    clocks: Object.fromEntries(published.map((c) => [c, clockPayload(c)])),
  };
  const tower = await composeControlTower();
  return tower.data.clocks;
}

beforeEach(() => { healthOverride.value = null; });

describe('F-07 · el reloj de PnL se LEE cuando el payload lo trae', () => {
  it('con data+model+pnl publicados, los TRES quedan disponibles', async () => {
    const clocks = await clocksFor([...HEALTH_CLOCKS]);
    for (const c of HEALTH_CLOCKS) {
      expect(isAvailable(clocks[c]), `reloj ${c} descartado pese a venir en el payload`).toBe(true);
    }
  });

  it('un reloj presente en el payload JAMÁS se publica como unavailable', async () => {
    const clocks = await clocksFor(['data', 'pnl']);
    expect(isAvailable(clocks.data)).toBe(true);
    expect(isAvailable(clocks.pnl)).toBe(true);
    expect(isAvailable(clocks.model)).toBe(false);
  });
});

describe('F-07 · la nota se DERIVA del payload, no está escrita a mano', () => {
  it('la nota del reloj ausente enumera exactamente lo que el payload publica', async () => {
    const clocks = await clocksFor(['data', 'model']);
    const pending = clocks.pnl.source.pending ?? '';
    // La lista que la nota declara como PUBLICADA es exactamente la del payload…
    const declared = /publica:\s*([^)]*)/.exec(pending)?.[1] ?? '';
    expect(declared.split(',').map((s) => s.trim()).filter(Boolean)).toEqual(['data', 'model']);
    // …y el reloj ausente jamás aparece entre los publicados.
    expect(declared).not.toContain('pnl');
  });

  it('cambia con el payload: si solo hay `data`, la nota no puede citar `model`', async () => {
    const clocks = await clocksFor(['data']);
    const pendingModel = clocks.model.source.pending ?? '';
    const pendingPnl = clocks.pnl.source.pending ?? '';
    for (const pending of [pendingModel, pendingPnl]) {
      // La lista de "publicados" que la nota declara debe ser exactamente ['data'].
      const declared = /publica:\s*([^)]*)/.exec(pending)?.[1] ?? '';
      expect(declared.split(',').map((s) => s.trim()).filter(Boolean)).toEqual(['data']);
    }
  });

  it('sin system_health.json la nota lo dice, sin inventar relojes publicados', async () => {
    healthOverride.value = { clocks: {} };
    const tower = await composeControlTower();
    const clocks = tower.data.clocks;
    for (const c of HEALTH_CLOCKS) {
      const pending = clocks[c].source.pending ?? '';
      const declared = /publica:\s*([^)]*)/.exec(pending)?.[1] ?? '';
      expect(declared.split(',').map((s) => s.trim()).filter(Boolean)).toEqual([]);
    }
  });
});
