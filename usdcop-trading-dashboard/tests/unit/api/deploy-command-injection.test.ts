/**
 * P0 · `/api/production/deploy` ejecutaba el manifiesto de aprobación A TRAVÉS DEL SHELL.
 * =======================================================================================
 *
 * El hallazgo (auto-red-team CLAUDE, clasificado "no arreglado" porque quitar `shell:true`
 * amenazaba el deploy local en Windows):
 *
 *     spawn('python3', [manifest.script, ...manifest.args], { shell: true })
 *
 * `script` y `args` salen del artefacto de aprobación (`data/approvals/approval_state*.json`).
 * Con `shell: true` el array se APLANA en una línea de comandos que interpreta `cmd.exe`
 * (o `/bin/sh`): cualquier `;`, `&`, `&&`, `|`, `$( )` o backtick dentro del manifiesto
 * ejecuta comandos arbitrarios con el usuario del servidor, en la ruta que promueve a
 * producción.
 *
 * Matiz honesto de dimensionamiento: escribir en `data/approvals/` ya implica un
 * compromiso previo. Pero es una escalada GRATUITA de "escribo un JSON" a "ejecuto
 * cualquier comando", y la ruta afectada es la del dinero.
 *
 * Cómo se demuestra el rojo sin ejecutar nada peligroso: el manifiesto lleva un CENTINELA
 * inofensivo (`echo` a un fichero temporal). Si el shell interpreta la cadena, el fichero
 * aparece. El script "principal" del centinela es inexistente a propósito, para que el
 * intérprete falle de inmediato y JAMÁS se dispare un reentrenamiento real.
 *
 * Invariantes que fija este fichero (`.claude/rules/approval-gates.md` §4, K-040):
 *   1. Sin shell. El proceso se lanza con argv explícito.
 *   2. Allowlist (no blacklist) de `script` (dentro del árbol `scripts/`, existente, .py)
 *      y de `args` (forma declarada).
 *   3. Fail-closed: cualquier duda ⇒ NO se ejecuta, ni local ni vía Airflow.
 *   4. El rechazo deja incidente en auditoría.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import realFs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const authMock = vi.hoisted(() => ({ protect: vi.fn() }));
vi.mock('@/lib/auth/api-auth', () => ({
  protectApiRoute: (...a: unknown[]) => authMock.protect(...a),
}));

const dbMock = vi.hoisted(() => ({ query: vi.fn() }));
vi.mock('@/lib/db/postgres-client', () => ({
  query: (...a: unknown[]) => dbMock.query(...a),
  default: { query: (...a: unknown[]) => dbMock.query(...a) },
}));

// `spawn` se OBSERVA. Por defecto devuelve un hijo falso (ningún proceso real); el test
// del centinela sustituye la implementación por el `spawn` real para probar que hoy la
// cadena inyectada llegaría al shell.
// (Ojo: la factoría NO puede usar `importOriginal` sobre un builtin — vitest devuelve
// entonces el módulo real y el `spawn` de la ruta se ejecutaría DE VERDAD.)
const procMock = vi.hoisted(() => ({ spawn: vi.fn() }));
vi.mock('child_process', () => {
  const spawn = (...a: unknown[]) => procMock.spawn(...a);
  return { spawn, default: { spawn } };
});
vi.mock('node:child_process', () => {
  const spawn = (...a: unknown[]) => procMock.spawn(...a);
  return { spawn, default: { spawn } };
});

import { POST as deployPOST } from '@/app/api/production/deploy/route';

const REPO_ROOT = path.resolve(process.cwd(), '..');
const DEPLOY_STATUS = path.join(process.cwd(), 'public', 'data', 'production', 'deploy_status.json');

const LEGIT_MANIFEST = {
  pipeline_type: 'ml_forecasting',
  script: 'scripts/pipeline/train_and_export_smart_simple.py',
  args: ['--phase', 'production', '--no-png', '--seed-db'],
  config_path: 'config/execution/smart_simple_v1.yaml',
  db_tables: ['forecast_h5_predictions'],
};

const APPROVED = (manifest: unknown) => ({
  status: 'APPROVED',
  strategy: 'smart_simple_v11',
  strategy_name: 'Smart Simple v1.1',
  backtest_recommendation: 'PROMOTE',
  backtest_confidence: 1,
  gates: [],
  deploy_manifest: manifest,
  approved_by: 'admin@x.co',
  approved_at: '2026-07-28T00:00:00Z',
  created_at: '2026-07-01T00:00:00Z',
  last_updated: '2026-07-28T00:00:00Z',
});

let tmp: string;
let statusBackup: string | null = null;

const seed = (manifest: unknown) =>
  realFs.writeFileSync(path.join(tmp, 'approval_state.json'), JSON.stringify(APPROVED(manifest), null, 2), 'utf-8');

const principal = (role: string) => ({
  authenticated: true,
  user: { id: `user-${role}`, email: `${role}@x.co`, username: role, role },
});

const deployReq = (body: Record<string, unknown> = {}) =>
  new NextRequest('http://localhost:3001/api/production/deploy', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  } as never);

/** Hijo falso: la ruta solo necesita pid, streams opcionales, `on` y `unref`. */
function fakeChild() {
  return {
    pid: 4242,
    stdout: { on: vi.fn() },
    stderr: { on: vi.fn() },
    on: vi.fn(),
    unref: vi.fn(),
  };
}

/** El `spawn` de verdad (el mock cubre ambos especificadores). Solo para el centinela. */
const realSpawn = async () =>
  (await vi.importActual<typeof import('node:child_process')>('node:child_process')).spawn;

const deployStatus = () =>
  realFs.existsSync(DEPLOY_STATUS) ? JSON.parse(realFs.readFileSync(DEPLOY_STATUS, 'utf-8')) : null;

beforeEach(() => {
  tmp = realFs.mkdtempSync(path.join(os.tmpdir(), 'deploy-cmd-'));
  process.env.APPROVALS_DATA_DIR = tmp;
  delete process.env.AUTH_BYPASS_ENABLED;
  // Sin Airflow configurado ⇒ la ruta cae al lanzamiento local (el camino auditado aquí).
  delete process.env.AIRFLOW_API_URL;
  delete process.env.AIRFLOW_API_USER;
  delete process.env.AIRFLOW_API_PASSWORD;
  delete process.env.DEPLOY_PYTHON_BIN;
  statusBackup = realFs.existsSync(DEPLOY_STATUS) ? realFs.readFileSync(DEPLOY_STATUS, 'utf-8') : null;
  vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 200 })));
  authMock.protect.mockReset();
  authMock.protect.mockResolvedValue(principal('admin'));
  dbMock.query.mockReset();
  dbMock.query.mockResolvedValue({ rows: [] });
  procMock.spawn.mockReset();
  procMock.spawn.mockImplementation(() => fakeChild());
});

afterEach(() => {
  delete process.env.APPROVALS_DATA_DIR;
  realFs.rmSync(tmp, { recursive: true, force: true });
  if (statusBackup !== null) realFs.writeFileSync(DEPLOY_STATUS, statusBackup, 'utf-8');
  else realFs.rmSync(DEPLOY_STATUS, { force: true });
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ═════════════ 1 · inyección de comandos con centinela REAL (el rojo del hallazgo) ═════

describe('P0 · el manifiesto no puede ejecutar comandos a través del shell', () => {
  it('metacaracteres en `script` NO producen el centinela y la petición se rechaza', async () => {
    const sentinel = path.join(tmp, 'pwned.txt');
    // Script inexistente + encadenado de shell: si hay shell, el intérprete falla al
    // instante (no hay reentrenamiento) y el `echo` escribe el centinela.
    const injected =
      process.platform === 'win32'
        ? `scripts/pipeline/__no_such_script__.py & echo pwned> ${sentinel}`
        : `scripts/pipeline/__no_such_script__.py ; echo pwned > ${sentinel}`;
    seed({ ...LEGIT_MANIFEST, script: injected });
    // Aquí se usa el `spawn` REAL: es la única forma honesta de probar que hoy la cadena
    // llega al shell y mañana ni siquiera se lanza.
    const spawnReal = await realSpawn();
    procMock.spawn.mockImplementation((...a: unknown[]) =>
      (spawnReal as unknown as (...x: unknown[]) => unknown)(...a),
    );

    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    await new Promise((r) => setTimeout(r, 1500)); // margen para el hijo desacoplado

    expect(realFs.existsSync(sentinel)).toBe(false); // ← el centinela NO se materializa
    expect(res.status).toBe(400);
    expect((await res.json()).success).toBe(false);
  });

  it('metacaracteres en `args` NO producen el centinela y la petición se rechaza', async () => {
    const sentinel = path.join(tmp, 'pwned-args.txt');
    const injected =
      process.platform === 'win32' ? `production & echo pwned> ${sentinel}` : `production ; echo pwned > ${sentinel}`;
    seed({ ...LEGIT_MANIFEST, script: 'scripts/pipeline/__no_such_script__.py', args: ['--phase', injected] });
    const spawnReal = await realSpawn();
    procMock.spawn.mockImplementation((...a: unknown[]) =>
      (spawnReal as unknown as (...x: unknown[]) => unknown)(...a),
    );

    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    await new Promise((r) => setTimeout(r, 1500));

    expect(realFs.existsSync(sentinel)).toBe(false);
    expect(res.status).toBe(400);
  });

  it.each([
    ['sustitución de comando', '$(whoami).py'],
    ['backticks', '`whoami`.py'],
    ['and lógico', 'scripts/pipeline/x.py && whoami'],
    ['tubería', 'scripts/pipeline/x.py | whoami'],
    ['redirección', 'scripts/pipeline/x.py > out.txt'],
    ['newline', 'scripts/pipeline/x.py\nwhoami'],
  ])('%s en `script` ⇒ rechazo sin lanzar proceso', async (_label, script) => {
    seed({ ...LEGIT_MANIFEST, script });
    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(400);
    expect(procMock.spawn).not.toHaveBeenCalled();
  });

  it('nunca se usa `shell: true` en el caso legítimo', async () => {
    seed(LEGIT_MANIFEST);
    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(200);
    expect(procMock.spawn).toHaveBeenCalledTimes(1);
    const opts = procMock.spawn.mock.calls[0][2] as { shell?: unknown };
    expect(opts.shell).toBeFalsy();
  });
});

// ══════════════════════ 2 · allowlist de ruta (traversal / fuera del árbol) ═══════════

describe('P0 · `script` fuera del árbol permitido se rechaza', () => {
  it.each([
    ['traversal relativo', '../../../etc/passwd.py'],
    ['traversal desde scripts/', 'scripts/../airflow/dags/forecast_h5_l4b_production_deploy.py'],
    ['absoluto POSIX', '/tmp/evil.py'],
    ['absoluto Windows', 'C:\\Windows\\Temp\\evil.py'],
    ['UNC', '\\\\attacker\\share\\evil.py'],
    ['fuera de scripts/ pero dentro del repo', 'airflow/dags/forecast_h5_l4b_production_deploy.py'],
    ['no es .py', 'scripts/ops/backup/restore.sh'],
    ['inexistente dentro del árbol', 'scripts/pipeline/__definitivamente_no_existe__.py'],
  ])('%s ⇒ 400 y ningún proceso', async (_label, script) => {
    seed({ ...LEGIT_MANIFEST, script });
    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(400);
    expect(procMock.spawn).not.toHaveBeenCalled();
  });

  it.each([
    ['espacio + flag encubierto', ['--phase', 'production --seed-db']],
    ['comilla', ["--phase", "produ'ction"]],
    ['variable de entorno', ['--phase', '$HOME']],
    ['no es cadena', ['--phase', 42 as unknown as string]],
    ['ruta absoluta como valor', ['--config', '/etc/shadow']],
  ])('args inválidos (%s) ⇒ 400 y ningún proceso', async (_label, args) => {
    seed({ ...LEGIT_MANIFEST, args });
    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(400);
    expect(procMock.spawn).not.toHaveBeenCalled();
  });
});

// ══════════════════════════ 3 · el camino legítimo sigue vivo ════════════════════════

describe('el deploy legítimo sigue funcionando', () => {
  it('manifiesto válido ⇒ 200, argv explícito y ruta absoluta dentro de scripts/', async () => {
    seed(LEGIT_MANIFEST);
    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(200);

    const [exe, argv, opts] = procMock.spawn.mock.calls[0] as [string, string[], Record<string, unknown>];
    expect(typeof exe).toBe('string');
    expect(exe.length).toBeGreaterThan(0);
    expect(argv[0]).toBe(path.join(REPO_ROOT, 'scripts', 'pipeline', 'train_and_export_smart_simple.py'));
    expect(argv.slice(1)).toEqual(['--phase', 'production', '--no-png', '--seed-db']);
    expect(opts.cwd).toBe(REPO_ROOT);
    expect(deployStatus()?.status).toBe('running');
  });

  it('sin manifiesto ⇒ fallback legado (mismo script, mismos args)', async () => {
    seed(undefined);
    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(200);
    const [, argv] = procMock.spawn.mock.calls[0] as [string, string[]];
    expect(argv[0]).toBe(path.join(REPO_ROOT, 'scripts', 'pipeline', 'train_and_export_smart_simple.py'));
    expect(argv.slice(1)).toEqual(['--phase', 'production', '--no-png', '--seed-db']);
  });

  it('el intérprete se resuelve por configuración explícita, no por el shell', async () => {
    process.env.DEPLOY_PYTHON_BIN = '/usr/local/bin/python3.12';
    seed(LEGIT_MANIFEST);
    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(200);
    expect(procMock.spawn.mock.calls[0][0]).toBe('/usr/local/bin/python3.12');
  });
});

// ═══════════════════════════ 4 · el rechazo deja incidente ═══════════════════════════

describe('un manifiesto rechazado deja incidente en auditoría', () => {
  it('fila en audit_log + deploy_status failed, y NO se dispara el DAG de Airflow', async () => {
    process.env.AIRFLOW_API_URL = 'http://airflow:8080';
    process.env.AIRFLOW_API_USER = 'u';
    process.env.AIRFLOW_API_PASSWORD = 'p';
    seed({ ...LEGIT_MANIFEST, script: 'scripts/pipeline/x.py; whoami' });

    const res = await deployPOST(deployReq({ strategy_id: 'smart_simple_v11' }));
    expect(res.status).toBe(400);

    const sql = dbMock.query.mock.calls.map((c) => String(c[0])).join('\n');
    expect(sql).toMatch(/INSERT INTO audit_log/i);
    expect(JSON.stringify(dbMock.query.mock.calls)).toContain('deploy_manifest_rejected');

    // Fail-closed también para el camino preferido: un manifiesto inválido no puede
    // delegarse a Airflow (el DAG ejecuta ESE MISMO manifiesto).
    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect(deployStatus()?.status).toBe('failed');
  });
});
