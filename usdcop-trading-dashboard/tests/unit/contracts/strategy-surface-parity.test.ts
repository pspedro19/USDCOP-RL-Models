/**
 * C-005 / BL-13 — paridad RUNTIME del validador de surface (exigencia CXD re-review:
 * "test runtime Vitest de ausencia/unknown/valid", no inspección por regex).
 * Espejo funcional de validate_surface() en src/contracts/strategy_manifest.py:
 * ausencia => 'action' (legacy), desconocido => rechazo, whitelist cerrada.
 */
import { describe, expect, it } from 'vitest';
import {
  STRATEGY_SURFACES,
  assertStrategySurface,
  isStrategySurface,
  validateStrategySurface,
} from '@/lib/contracts/strategy-manifest.contract';
import {
  STRATEGY_SURFACES as REEXPORTED_SURFACES,
  validateStrategySurface as reexportedValidate,
} from '@/lib/contracts/strategy.contract';

describe('surface whitelist runtime (C-005)', () => {
  it('whitelist cerrada exacta y una sola fuente re-exportada', () => {
    expect([...STRATEGY_SURFACES]).toEqual(['action', 'diagnostic']);
    expect(REEXPORTED_SURFACES).toBe(STRATEGY_SURFACES);
    expect(reexportedValidate).toBe(validateStrategySurface);
  });

  it('ausencia es legal (semantica legacy => action)', () => {
    expect(validateStrategySurface(undefined)).toEqual([]);
    expect(validateStrategySurface(null)).toEqual([]);
    expect(assertStrategySurface(undefined)).toBe('action');
    expect(assertStrategySurface(null)).toBe('action');
  });

  it('valores validos pasan', () => {
    for (const s of STRATEGY_SURFACES) {
      expect(validateStrategySurface(s)).toEqual([]);
      expect(assertStrategySurface(s)).toBe(s);
      expect(isStrategySurface(s)).toBe(true);
    }
  });

  it('desconocidos fallan CERRADO (condicion del ACK C-005)', () => {
    const invalid: unknown[] = ['banana', 'unknown_surface', 'ACTION', '', true, 1, {}, []];
    for (const v of invalid) {
      expect(validateStrategySurface(v).length, `debe rechazar ${JSON.stringify(v)}`).toBeGreaterThan(0);
      expect(isStrategySurface(v)).toBe(false);
      expect(() => assertStrategySurface(v)).toThrowError();
    }
  });
});
