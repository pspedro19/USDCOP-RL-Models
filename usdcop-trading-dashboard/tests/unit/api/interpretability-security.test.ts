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

import { validateAndStrip } from '@/app/api/admin/interpretability/_lib/artifact-schema';
import { GET as indexGET } from '@/app/api/admin/interpretability/route';
import { GET as summaryGET } from '@/app/api/admin/interpretability/summary/route';

import { REAL_LINEAR } from '../../support/interp-fixtures';

const ADMIN = { 'x-user-role': 'admin' };
const FREE = { 'x-user-role': 'free' };

const INDEX_URL = 'http://t/api/admin/interpretability';
const sumUrl = (qs: string) => `http://t/api/admin/interpretability/summary?${qs}`;
const q = (surface: string, asset: string, model: string, version: string) =>
  `surface=${surface}&asset=${asset}&model_id=${model}&version=${version}`;

/**
 * Artefacto lineal VÁLIDO — leído del artefacto REAL trackeado en
 * `<repo>/data/interpretability/zoo/usdcop/ridge/**` (tests/support/interp-fixtures.ts).
 *
 * NO es una copia a mano: la copia a mano se quedó sin `artifact_id`/`provenance` cuando
 * BL-20 los añadió al schema, dejó la ruta feliz en 500 y — lo grave — convirtió los casos
 * fail-closed de abajo en verdes por el fixture rancio en vez de por su defensa.
 */
const VALID_LINEAR: Record<string, unknown> = JSON.parse(JSON.stringify(REAL_LINEAR));

let base: string; // raíz temporal de artefactos (INTERPRETABILITY_DATA_DIR)
let outside: string; // directorio HERMANO fuera del base (target de symlinks)
let symlinkOk = false;

function writeArtifact(segs: string[], content: string) {
  const dir = path.join(base, ...segs);
  realFs.mkdirSync(dir, { recursive: true });
  realFs.writeFileSync(path.join(dir, 'summary.json'), content);
}

/** Serializa `obj` con `field` = `1e999` (⇒ Infinity al parsear). Falla ruidosamente si no muta nada. */
function nonFiniteJson(obj: Record<string, unknown>, field: string): string {
  const SENTINEL = '__NON_FINITE__';
  const json = JSON.stringify({ ...obj, [field]: SENTINEL }).replace(`"${SENTINEL}"`, '1e999');
  if (!json.includes('1e999')) throw new Error(`no se pudo inyectar el no-finito en ${field}`);
  return json;
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
    // El centinela se sustituye por token, no por el literal del valor: así el caso no
    // se vuelve un no-op silencioso cuando el artefacto real cambie de `base_value`.
    nonFiniteJson({ ...VALID_LINEAR, model_id: 'nonfinite' }, 'base_value'),
  );
  // Fixture válido MENOS exactamente un campo required: el rechazo solo puede
  // atribuirse a `required`, no a un fixture incompleto por otro motivo.
  const missingRequired: Record<string, unknown> = { ...VALID_LINEAR, model_id: 'badschema' };
  delete missingRequired.base_value;
  writeArtifact(['zoo', 'usdcop', 'badschema', '2026-07-27'], JSON.stringify(missingRequired));
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

// ─────────────────────────────────── control de atribución de los casos fail-closed
/**
 * Los casos de #2 son "el artefacto VÁLIDO + UNA mutación". Eso solo prueba la defensa
 * si el artefacto base es realmente válido: con un fixture rancio, todos devuelven 500
 * por el fixture y la suite queda verde sin ejercitar nada (fue exactamente lo que pasó
 * cuando el schema ganó `artifact_id`/`provenance` y el fixture no se actualizó).
 * Este control ata el rojo al lugar correcto — el fixture — en vez de disfrazarlo de
 * defensa que funciona.
 */
describe('#0 — control: el fixture base es válido, así los fail-closed son atribuibles', () => {
  it('el artefacto REAL pasa el schema COMPARTIDO sin mutar (y sobrevive el STRIP intacto)', () => {
    expect(validateAndStrip(VALID_LINEAR)).toEqual(VALID_LINEAR);
  });

  it('cada mutación de #2 es rechazada por SU defensa y solo por ella', () => {
    // no-finito: único cambio respecto del fixture válido.
    expect(validateAndStrip(JSON.parse(nonFiniteJson(VALID_LINEAR, 'base_value')))).toBeNull();
    // required ausente: único cambio respecto del fixture válido.
    const missing: Record<string, unknown> = { ...VALID_LINEAR };
    delete missing.base_value;
    expect(validateAndStrip(missing)).toBeNull();
    // campo desconocido: NO es rechazo — se stripea y el resto llega intacto.
    const stripped = validateAndStrip({ ...VALID_LINEAR, ___evil_extra: '<script>alert(1)</script>' });
    expect(stripped).toEqual(VALID_LINEAR);
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
