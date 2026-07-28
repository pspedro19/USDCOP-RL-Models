/**
 * C-006/BL-20-UI (CXD-040) — tests de seguridad de las rutas de interpretabilidad.
 *
 * Cobertura 1:1 de los hallazgos de Codex:
 *  #1 artefactos fuera de public/ (única vía = API con admin:all)
 *  #2 validación runtime del artefacto (schema fail-closed, unknown-field STRIP,
 *     números finitos, size-cap 2MB) + errores SIEMPRE genéricos
 *  #4 traversal: whitelist + path.resolve + realpath-prefix (%2e%2e, absolutos, symlinks)
 *  #5 401/403 ANTES de tocar el filesystem (fs espiado; cero llamadas sin permiso)
 */
import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import os from 'node:os';
import realFs from 'node:fs';
import path from 'node:path';

// ── Espía de fs/promises: delega en el fs real pero cuenta llamadas (hallazgo #5).
const fsCalls = vi.hoisted(() => ({ list: [] as string[] }));
vi.mock('fs/promises', async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown> & { default?: Record<string, unknown> };
  const real = (actual.default ?? actual) as Record<string, (...a: unknown[]) => unknown>;
  const wrapped: Record<string, unknown> = { ...real };
  for (const k of ['readdir', 'readFile', 'stat', 'realpath', 'access', 'lstat', 'open']) {
    if (typeof real[k] === 'function') {
      wrapped[k] = (...a: unknown[]) => {
        fsCalls.list.push(k);
        return real[k](...a);
      };
    }
  }
  return { ...actual, ...wrapped, default: wrapped };
});

import { GET as indexGET } from '@/app/api/admin/interpretability/route';
import { GET as summaryGET } from '@/app/api/admin/interpretability/summary/route';

const ADMIN = { 'x-user-role': 'admin' };
const FREE = { 'x-user-role': 'free' };

const INDEX_URL = 'http://t/api/admin/interpretability';
const sumUrl = (qs: string) => `http://t/api/admin/interpretability/summary?${qs}`;
const q = (surface: string, asset: string, model: string, version: string) =>
  `surface=${surface}&asset=${asset}&model_id=${model}&version=${version}`;

// Artefacto lineal mínimo VÁLIDO según el schema compartido.
const VALID_LINEAR = {
  nota: 'SHAP explica el modelo, no el mercado; solo test-folds; diagnostico 0 trials',
  surface: 'zoo',
  asset: 'usdcop',
  model_id: 'ridge',
  model_type: 'linear',
  method: 'linear_shap_closed_form',
  attribution_not_shap: false,
  version: '2026-07-27',
  generated_at: '2026-07-28T00:00:00+00:00',
  scope: 'diagnostico del modelo congelado',
  fit: {
    scheme: 'walk-forward',
    origin: '2026-07-27',
    n_train: 100,
    horizon: 5,
    purge_days: 5,
    scaler: 'StandardScaler train-only',
    params: { alpha: 1.0, fit_intercept: true },
  },
  base_value: -0.0001,
  n_rows: 105,
  n_features: 1,
  top_features: [{ rank: 1, feature: 'return_10d', coef: 0.007, mean_abs_shap: 0.005, mean_shap: -0.00001 }],
  by_year: { '2026': [{ feature: 'return_10d', mean_abs_shap: 0.005, mean_shap: -0.00001 }] },
  kill_flags_sign_change_by_year: ['return_10d'],
};

let base: string; // raíz temporal de artefactos (INTERPRETABILITY_DATA_DIR)
let outside: string; // directorio HERMANO fuera del base (target de symlinks)
let symlinkOk = false;

function writeArtifact(segs: string[], content: string) {
  const dir = path.join(base, ...segs);
  realFs.mkdirSync(dir, { recursive: true });
  realFs.writeFileSync(path.join(dir, 'summary.json'), content);
}

beforeAll(() => {
  base = realFs.mkdtempSync(path.join(os.tmpdir(), 'interp-base-'));
  outside = realFs.mkdtempSync(path.join(os.tmpdir(), 'interp-outside-'));
  process.env.INTERPRETABILITY_DATA_DIR = base;

  writeArtifact(['zoo', 'usdcop', 'ridge', '2026-07-27'], JSON.stringify(VALID_LINEAR));
  writeArtifact(
    ['zoo', 'usdcop', 'extras', '2026-07-27'],
    JSON.stringify({ ...VALID_LINEAR, model_id: 'extras', ___evil_extra: '<script>alert(1)</script>' }),
  );
  writeArtifact(
    ['zoo', 'usdcop', 'nonfinite', '2026-07-27'],
    // 1e999 parsea a Infinity — JSON.parse NO lo rechaza; el schema runtime sí debe.
    JSON.stringify({ ...VALID_LINEAR, model_id: 'nonfinite' }).replace('"base_value":-0.0001', '"base_value":1e999'),
  );
  writeArtifact(
    ['zoo', 'usdcop', 'badschema', '2026-07-27'],
    JSON.stringify({ surface: 'zoo', asset: 'usdcop' }), // faltan required
  );
  writeArtifact(['zoo', 'usdcop', 'corrupt', '2026-07-27'], '{not json{{{');
  writeArtifact(
    ['zoo', 'usdcop', 'big', '2026-07-27'],
    JSON.stringify({ ...VALID_LINEAR, model_id: 'big', nota: 'A'.repeat(2 * 1024 * 1024 + 1024) }),
  );

  // Symlink/junction que apunta FUERA del base (hallazgo #4). En Windows la junction
  // de directorios no requiere privilegios; si el FS no lo permite, se omite ese caso.
  realFs.mkdirSync(path.join(outside, 'secretver'), { recursive: true });
  realFs.writeFileSync(path.join(outside, 'secretver', 'summary.json'), JSON.stringify(VALID_LINEAR));
  try {
    realFs.mkdirSync(path.join(base, 'zoo', 'usdcop', 'linkmodel'), { recursive: true });
    realFs.symlinkSync(
      path.join(outside, 'secretver'),
      path.join(base, 'zoo', 'usdcop', 'linkmodel', '2026-07-27'),
      'junction',
    );
    symlinkOk = true;
  } catch {
    symlinkOk = false;
  }
});

afterAll(() => {
  delete process.env.INTERPRETABILITY_DATA_DIR;
  realFs.rmSync(base, { recursive: true, force: true });
  realFs.rmSync(outside, { recursive: true, force: true });
});

// ─────────────────────────────────── #1 · artefactos fuera de public/
describe('#1 — los artefactos NO viven en public/ (cierre del bypass estático)', () => {
  it('public/data/interpretability ya no existe (git mv a <repo>/data/interpretability)', () => {
    expect(realFs.existsSync(path.join(process.cwd(), 'public', 'data', 'interpretability'))).toBe(false);
  });

  it('los artefactos reales viven en <repo>/data/interpretability', () => {
    const repoRoot = path.resolve(process.cwd(), '..');
    expect(
      realFs.existsSync(path.join(repoRoot, 'data', 'interpretability', 'zoo', 'usdcop', 'ridge', '2026-07-27', 'summary.json')),
    ).toBe(true);
  });
});

// ─────────────────────────────────── #5 · RBAC antes del filesystem
describe('#5 — 401/403 ANTES de tocar el filesystem', () => {
  it.each([
    ['index sin sesión', () => indexGET(new Request(INDEX_URL)), 401],
    ['index rol free', () => indexGET(new Request(INDEX_URL, { headers: FREE })), 403],
    ['summary sin sesión', () => summaryGET(new Request(sumUrl(q('zoo', 'usdcop', 'ridge', '2026-07-27')))), 401],
    ['summary rol free', () => summaryGET(new Request(sumUrl(q('zoo', 'usdcop', 'ridge', '2026-07-27')), { headers: FREE })), 403],
  ] as const)('%s ⇒ %i sin llamadas fs', async (_name, call, status) => {
    fsCalls.list.length = 0;
    const res = await call();
    expect(res.status).toBe(status);
    expect(fsCalls.list).toEqual([]);
  });
});

// ─────────────────────────────────── #4 · traversal / absolutos / symlinks
describe('#4 — traversal bloqueado (whitelist + realpath-prefix)', () => {
  it.each([
    ['.. literal', q('zoo', 'usdcop', 'ridge', '..')],
    ['%2e%2e codificado', q('zoo', 'usdcop', 'ridge', '%2e%2e')],
    ['%2e%2e%2fdir codificado', q('zoo', 'usdcop', '%2e%2e%2fridge', '2026-07-27')],
    ['separador /', 'surface=zoo&asset=usdcop&model_id=a%2Fb&version=2026-07-27'],
    ['separador \\', 'surface=zoo&asset=usdcop&model_id=a%5Cb&version=2026-07-27'],
    ['absoluto win', 'surface=zoo&asset=usdcop&model_id=ridge&version=C%3A%5CWindows'],
    ['absoluto posix', 'surface=zoo&asset=usdcop&model_id=ridge&version=%2Fetc%2Fpasswd'],
    ['null byte', 'surface=zoo&asset=usdcop&model_id=ridge&version=2026%002E'],
    ['segmento vacío', 'surface=zoo&asset=usdcop&model_id=ridge'],
    ['punto inicial', q('zoo', 'usdcop', 'ridge', '.hidden')],
  ])('%s ⇒ 400 genérico', async (_name, qs) => {
    const res = await summaryGET(new Request(sumUrl(qs), { headers: ADMIN }));
    expect(res.status).toBe(400);
    const body = JSON.stringify(await res.json());
    expect(body).not.toMatch(/[A-Za-z]:\\\\|\/etc\/|interp-base|summary\.json/i);
  });

  it('symlink/junction que sale del base ⇒ 404 genérico (realpath-prefix)', async () => {
    if (!symlinkOk) return; // FS sin symlinks — cubierto por el resto de la matriz
    const res = await summaryGET(
      new Request(sumUrl(q('zoo', 'usdcop', 'linkmodel', '2026-07-27')), { headers: ADMIN }),
    );
    expect(res.status).toBe(404);
    const body = JSON.stringify(await res.json());
    expect(body).not.toMatch(/interp-outside|secretver/i);
  });

  it('inexistente ⇒ 404 con mensaje genérico (sin eco de segmentos)', async () => {
    const res = await summaryGET(new Request(sumUrl(q('zoo', 'usdcop', 'nope', '2026-07-27')), { headers: ADMIN }));
    expect(res.status).toBe(404);
    const body = (await res.json()) as { error: { message: string } };
    expect(body.error.message).toBe('artifact not found');
  });
});

// ─────────────────────────────────── #2 · validación runtime + errores genéricos
describe('#2 — schema fail-closed, STRIP, finitos, size-cap, errores genéricos', () => {
  it('artefacto válido ⇒ 200 con envelope ok y el contenido íntegro', async () => {
    const res = await summaryGET(new Request(sumUrl(q('zoo', 'usdcop', 'ridge', '2026-07-27')), { headers: ADMIN }));
    expect(res.status).toBe(200);
    const body = (await res.json()) as { ok: boolean; data: typeof VALID_LINEAR };
    expect(body.ok).toBe(true);
    expect(body.data).toEqual(VALID_LINEAR);
  });

  it('campos desconocidos se STRIPean (no llegan al cliente)', async () => {
    const res = await summaryGET(new Request(sumUrl(q('zoo', 'usdcop', 'extras', '2026-07-27')), { headers: ADMIN }));
    expect(res.status).toBe(200);
    const raw = JSON.stringify(await res.json());
    expect(raw).not.toContain('___evil_extra');
    expect(raw).not.toContain('<script>');
  });

  it('número no finito (1e999 ⇒ Infinity) ⇒ 500 INVALID_ARTIFACT genérico', async () => {
    const res = await summaryGET(new Request(sumUrl(q('zoo', 'usdcop', 'nonfinite', '2026-07-27')), { headers: ADMIN }));
    expect(res.status).toBe(500);
    const body = (await res.json()) as { error: { code: string; message: string } };
    expect(body.error.code).toBe('INVALID_ARTIFACT');
    expect(body.error.message).toBe('invalid artifact');
  });

  it('faltan campos required ⇒ 500 INVALID_ARTIFACT (fail-closed)', async () => {
    const res = await summaryGET(new Request(sumUrl(q('zoo', 'usdcop', 'badschema', '2026-07-27')), { headers: ADMIN }));
    expect(res.status).toBe(500);
  });

  it('JSON corrupto ⇒ 500 genérico sin filtrar el error de parse', async () => {
    const res = await summaryGET(new Request(sumUrl(q('zoo', 'usdcop', 'corrupt', '2026-07-27')), { headers: ADMIN }));
    expect(res.status).toBe(500);
    const body = JSON.stringify(await res.json());
    expect(body).not.toMatch(/Unexpected token|position|JSON\.parse|EISDIR|ENOTDIR/i);
    expect(body).toContain('invalid artifact');
  });

  it('artefacto > 2MB ⇒ rechazado por size-cap ANTES de parsear', async () => {
    fsCalls.list.length = 0;
    const res = await summaryGET(new Request(sumUrl(q('zoo', 'usdcop', 'big', '2026-07-27')), { headers: ADMIN }));
    expect(res.status).toBe(500);
    expect(fsCalls.list).toContain('stat');
    expect(fsCalls.list).not.toContain('readFile'); // el cap corta antes de leer
  });
});

// ─────────────────────────────────── índice
describe('índice — solo artefactos reales bajo el base', () => {
  it('lista los summaries válidos del base y omite el symlink que escapa', async () => {
    const res = await indexGET(new Request(INDEX_URL, { headers: ADMIN }));
    expect(res.status).toBe(200);
    const body = (await res.json()) as { ok: boolean; data: { entries: { model_id: string }[] } };
    expect(body.ok).toBe(true);
    const ids = body.data.entries.map((e) => e.model_id);
    expect(ids).toContain('ridge');
    if (symlinkOk) expect(ids).not.toContain('linkmodel');
  });
});
