/**
 * Resolución SEGURA del comando de deploy (CTR-DEPLOY-CMD-001).
 * ==============================================================
 *
 * `/api/production/deploy` lanzaba el manifiesto de aprobación así:
 *
 *     spawn('python3', [manifest.script, ...manifest.args], { shell: true })
 *
 * `script` y `args` vienen del artefacto `data/approvals/approval_state*.json`. Con
 * `shell: true` Node NO pasa un argv: aplana todo en UNA línea que interpreta `cmd.exe`
 * (o `/bin/sh`), así que un `;`, `&`, `&&`, `|`, `$( )` o un backtick dentro del JSON
 * ejecuta comandos arbitrarios con el usuario del servidor — en la ruta que promueve a
 * producción. (`shell: true` estaba ahí "para resolver el PATH en Windows": eso se
 * resuelve nombrando el intérprete, no delegando en un shell.)
 *
 * Este módulo es la ÚNICA puerta por la que puede salir un comando de deploy. Espejo
 * Python (para el DAG H5-L4b, que ejecuta EL MISMO manifiesto):
 * `src/contracts/deploy_manifest.py`. Si cambias una regla aquí, cámbiala allí.
 *
 * Diseño (K-040: allowlist, nunca blacklist):
 *
 *  1. **Sin shell.** Se devuelve `{ interpreter, argv }` y el llamador hace `spawn` con
 *     array y `shell` ausente. Un metacarácter deja de ser sintaxis: como mucho sería un
 *     argumento literal… y ni siquiera llega ahí, porque (2) lo rechaza antes.
 *  2. **`script` por allowlist de forma + de ubicación + existencia**: relativo POSIX,
 *     bajo `scripts/`, segmentos `[A-Za-z0-9_][A-Za-z0-9_.-]*`, extensión `.py`, y su
 *     `realpath` DEBE quedar dentro del `realpath` de `<root>/scripts` (mata traversal,
 *     rutas absolutas, UNC y symlinks que apunten fuera) y ser un fichero regular.
 *  3. **`args` por forma declarada**: o bandera (`--phase`, `-v`) o valor "simple"
 *     (alfanumérico + `._-`), sin espacios, comillas, `$`, `/` ni `\`. Los manifiestos
 *     reales que escribe el pipeline (`--phase production --no-png --seed-db`,
 *     `--version 1.2.1`) entran; una cadena con sintaxis de shell no.
 *  4. **Fail-closed**: ante cualquier duda se devuelve `ok:false` con motivo, y el
 *     llamador audita el incidente y NO ejecuta nada (tampoco delega en Airflow: el DAG
 *     corre ese mismo manifiesto).
 *
 * Lo que este módulo NO promete: que el script permitido sea inocuo. Cualquier `.py`
 * dentro de `scripts/` puede lanzarse si el manifiesto lo nombra. La frontera es "código
 * versionado del repo" vs "cadena arbitraria del atacante"; estrechar más (una lista
 * literal de scripts de deploy) exigiría un registro por estrategia y queda declarado
 * como deuda en el reporte.
 */
import { promises as fs } from 'fs';
import path from 'path';

/** Raíz permitida, relativa a la raíz del proyecto. Nada fuera de aquí se ejecuta. */
export const ALLOWED_SCRIPT_ROOT = 'scripts';

/** Extensión permitida: el deploy es un entrypoint Python, no un shell script. */
export const ALLOWED_SCRIPT_EXT = '.py';

/** Segmento de ruta: empieza por alfanumérico o `_` ⇒ `.`/`..` quedan fuera por forma. */
const SEGMENT_RE = /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/;

/** Bandera CLI (`--phase`, `-v`). */
const FLAG_RE = /^--?[A-Za-z0-9][A-Za-z0-9-]*$/;

/** Valor CLI simple (`production`, `1.2.1`, `smart_simple_v11`). Sin `/`, `\`, espacios. */
const VALUE_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

export const MAX_ARGS = 24;
export const MAX_ARG_LEN = 128;
export const MAX_SCRIPT_LEN = 200;

/** Fallback legado cuando el artefacto no trae manifiesto (misma validación). */
export const LEGACY_SCRIPT = 'scripts/pipeline/train_and_export_smart_simple.py';
export const LEGACY_ARGS = ['--phase', 'production', '--no-png', '--seed-db'];

export interface DeployManifestInput {
  script?: unknown;
  args?: unknown;
}

export interface DeployCommand {
  /** Ejecutable a lanzar (argv[0]) — resuelto por configuración, jamás por el shell. */
  interpreter: string;
  /** Ruta ABSOLUTA y verificada del script. */
  scriptPath: string;
  /** Argumentos validados uno a uno. */
  args: string[];
  /** Ruta relativa (POSIX) que se aceptó — para logs/auditoría. */
  scriptRel: string;
  /** `true` si no había manifiesto y se usó el fallback legado. */
  legacy: boolean;
}

export type DeployCommandResolution =
  | { ok: true; command: DeployCommand }
  | { ok: false; field: 'script' | 'args' | 'manifest'; reason: string };

const deny = (field: 'script' | 'args' | 'manifest', reason: string): DeployCommandResolution => ({
  ok: false,
  field,
  reason,
});

/**
 * Intérprete de Python SIN shell.
 *
 * `shell: true` existía "para resolver el PATH en Windows". La forma explícita:
 *   1. `DEPLOY_PYTHON_BIN` (o `PYTHON_BIN`) — configuración del SERVIDOR, no del
 *      manifiesto: ruta absoluta del intérprete del entorno de deploy.
 *   2. Si no hay configuración: `python` en Windows (donde `python3` a menudo es solo el
 *      alias de la Store) y `python3` en Linux/macOS y en los contenedores.
 *
 * `spawn` sin shell YA busca en el PATH (libuv añade las extensiones de Windows), así
 * que un nombre desnudo funciona en ambos sistemas sin intérprete de comandos de por medio.
 */
export function resolvePythonExecutable(env: NodeJS.ProcessEnv = process.env): string {
  const configured = (env.DEPLOY_PYTHON_BIN || env.PYTHON_BIN || '').trim();
  if (configured) return configured;
  return process.platform === 'win32' ? 'python' : 'python3';
}

/** Valida la FORMA de la ruta relativa (sin tocar el disco). */
export function validateScriptShape(script: string): string | null {
  if (!script) return 'script vacío';
  if (script.length > MAX_SCRIPT_LEN) return `script demasiado largo (${script.length} > ${MAX_SCRIPT_LEN})`;
  if (script.includes('\\')) return 'separador de Windows / UNC no permitido (usa `/`)';
  if (script.includes('\0')) return 'byte nulo en la ruta';
  if (path.posix.isAbsolute(script) || path.win32.isAbsolute(script) || /^[A-Za-z]:/.test(script)) {
    return 'ruta absoluta no permitida';
  }
  const segments = script.split('/');
  if (segments[0] !== ALLOWED_SCRIPT_ROOT) return `debe estar bajo \`${ALLOWED_SCRIPT_ROOT}/\``;
  if (segments.length < 2) return 'falta el nombre del script';
  for (const seg of segments) {
    if (!SEGMENT_RE.test(seg)) return `segmento inválido: ${JSON.stringify(seg)}`;
  }
  if (!script.endsWith(ALLOWED_SCRIPT_EXT)) return `extensión no permitida (se exige ${ALLOWED_SCRIPT_EXT})`;
  return null;
}

/** Valida los argumentos por forma declarada. Devuelve el motivo del rechazo o `null`. */
export function validateArgs(args: unknown): string | null {
  if (args === undefined || args === null) return null; // ausente = sin argumentos
  if (!Array.isArray(args)) return 'args debe ser una lista';
  if (args.length > MAX_ARGS) return `demasiados args (${args.length} > ${MAX_ARGS})`;
  for (const raw of args) {
    if (typeof raw !== 'string') return `arg no textual: ${JSON.stringify(raw)}`;
    if (raw.length === 0 || raw.length > MAX_ARG_LEN) return `longitud de arg inválida: ${raw.length}`;
    if (!FLAG_RE.test(raw) && !VALUE_RE.test(raw)) return `arg fuera de la forma permitida: ${JSON.stringify(raw)}`;
  }
  return null;
}

/**
 * Resuelve el comando de deploy o explica por qué NO se puede ejecutar.
 * No lanza: el llamador decide (auditar + 400). Fail-closed por construcción.
 */
export async function resolveDeployCommand(
  projectRoot: string,
  manifest?: DeployManifestInput | null,
  env: NodeJS.ProcessEnv = process.env,
): Promise<DeployCommandResolution> {
  const legacy = !manifest;
  let script: string;
  let args: unknown;

  if (legacy) {
    script = LEGACY_SCRIPT;
    args = LEGACY_ARGS;
  } else {
    if (typeof manifest !== 'object') return deny('manifest', 'manifiesto no es un objeto');
    if (typeof manifest.script !== 'string') return deny('script', 'script ausente o no textual');
    script = manifest.script;
    args = manifest.args ?? [];
  }

  const shapeError = validateScriptShape(script);
  if (shapeError) return deny('script', shapeError);

  const argsError = validateArgs(args);
  if (argsError) return deny('args', argsError);

  // Ubicación REAL: el realpath del script debe caer dentro del realpath de `scripts/`.
  // Se compara sobre realpath (no sobre la ruta lógica) para que un symlink que apunte
  // fuera del árbol tampoco pase.
  let realRoot: string;
  try {
    realRoot = await fs.realpath(path.join(projectRoot, ALLOWED_SCRIPT_ROOT));
  } catch {
    return deny('script', `no existe el árbol permitido \`${ALLOWED_SCRIPT_ROOT}/\``);
  }

  const candidate = path.resolve(projectRoot, ...script.split('/'));
  let real: string;
  try {
    real = await fs.realpath(candidate);
  } catch {
    return deny('script', `el script no existe: ${script}`);
  }
  if (real !== realRoot && !real.startsWith(realRoot + path.sep)) {
    return deny('script', `el script resuelve fuera de \`${ALLOWED_SCRIPT_ROOT}/\``);
  }
  try {
    const st = await fs.stat(real);
    if (!st.isFile()) return deny('script', 'el script no es un fichero regular');
  } catch {
    return deny('script', 'no se pudo verificar el script');
  }

  return {
    ok: true,
    command: {
      interpreter: resolvePythonExecutable(env),
      scriptPath: real,
      args: (args as string[] | undefined) ?? [],
      scriptRel: script,
      legacy,
    },
  };
}
